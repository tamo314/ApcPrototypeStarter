# Design Contract — Compositional Execution Contract (Phase D / Task D-001)

**新規の提案契約。以下の域・状態・数値は、対象primitive（SORT）の修復pilotを事前登録するために
本タスクで新たに導出したものであり、既存repoにこの契約が実装済みという意味ではない。** 実装は
`src/apc/primitives/composition.py` の `execute_composition_recipe` / `CompositionRecipe` /
`CompositionLibrary`（AGENTS.mdの `PrimitiveBank` と `CompositionLibrary` 分離要件を実装済み）と、
`src/apc/environments/operations.py` の8primitive canonical registry
（`ALL_CANONICAL_OPERATIONS`、`src/apc/evaluation/unified_oracle_causal_benchmark.py:92-95`）を
変更せず前提とする。本契約は既存の合成実行経路を置き換えるものではなく、その経路が扱う入力域・
出力域・合成条件を明文化し、対象primitiveの修復範囲を限定するためのものである。

## 0. 適用範囲と非対象

対象primitive registryは深さ≤3の合成文法を持つ8つの canonical primitive
`{SELECT, COUNT, BIND, SHIFT, COPY, REVERSE, SORT, NEGATE}`
（`ALL_CANONICAL_OPERATIONS`、NRQ-005/006/007/008と同一の registry）。COMPARE / ACCUMULATE /
Branch-B novel op / Phase A.2 incremental op はこの registry に含まれず、本契約の対象外。

本タスクの初回修復対象は **SORT一つ** に固定する（後続task向けに一般形として書くが、本タスクが
authorizeするのはSORTの範囲のみ）。

## 1. 4つの状態を区別する

一つの (primitive, 入力長, 合成上の位置) について、次の4状態を**別のもの**として扱う。前者の
成立だけで後者をPASSとしない。

| 状態 | 定義 | 判定方法 | 学習・実行を必要とするか |
|---|---|---|---|
| `SHAPE_COMPATIBLE` | 前段primitiveの `output_length(L)` が後段primitiveの `is_valid_for_length` を構造的に満たす | `Operation.output_length` / `Operation.is_valid_for_length` の純関数評価のみ（`src/apc/environments/operations.py`）。モデル実行不要。 | 不要 |
| `LENGTH_SUPPORTED` | その長さが対象primitiveの**学習露出済み、または本契約で明示的に拡張registeredな有効長集合**に属する | 本契約 §3 の有効長集合表と比較 | 不要（登録の確認のみ） |
| `STANDALONE_QUALIFIED` | その長さの入力に対し、対象primitiveを**単独**実行したときのsequence EMが受入floor以上 | `execute_composition_recipe` を depth=1 で呼ぶ、または同等の単独forward。診断reset（§4条件B）は使わない。 | 必要（単独評価） |
| `COMPOSITION_QUALIFIED` | 対象primitiveが深さ≤3合成の一部として、**通常の合成経路**（前段の予測トークンをargmaxで確定し、frozen task-blind Coreで再encodeしてから次段に渡す。正解中間列の注入なし）で実行されたときに、合成セル全体のsequence EMが受入floorを満たす | `execute_composition_recipe`（`composition.py:194-299`）をそのまま呼ぶ。正解intermediateの注入（NRQ-007の条件B「診断reset」）は診断専用であり、この状態のPASS判定には使わない。 | 必要（合成評価） |

`SHAPE_COMPATIBLE` と `LENGTH_SUPPORTED` は既存artifactの再集計・静的解析で確認可能であり、本タスク
内で確定できる。`STANDALONE_QUALIFIED` と `COMPOSITION_QUALIFIED` はSORT-only学習の実行結果に
依存するため、本タスクでは**未確定**として登録する（pilot実行後にのみ判定できる）。

## 2. 入力域・出力域・引数（§3契約項目表、8 primitiveすべて）

すべて `src/apc/environments/operations.py` の `Operation` サブクラスの純関数から導出。値は
推測ではなくソースの `output_length` / `min_input_length` / `is_valid_for_length` /
`required_argument_names` をそのまま転記する。

| Primitive | `min_input_length` | 追加構造制約 | `output_length(L)` | 引数 | 引数の正規化・未定義時の扱い |
|---|---|---|---|---|---|
| SELECT | 1 | なし | `max(1, L // 2)` | `indices`: 昇順counted subset | `sample_params` はcurrent lengthから毎回再サンプル。オラクル外argumentは未定義（本契約は評価時のCorrect/Wrong/None controlのみ使用） |
| COUNT | 1 | なし | `1` | `target ∈ [0, vocab_size)` | vocabulary外の値は生成側で発生しない（`rng.randrange(vocab_size)`） |
| BIND | 2 | **偶数長のみ有効**（`is_valid_for_length`: `L>=2 and L%2==0`） | `1` | `query_key`: 配列中に実在するkeyのみ | 実在しないkeyは `sample_params` から出ない。`apply`は不一致時 `value=0` を返す（未定義入力ではなく明示的既定値） |
| SHIFT | 1 | なし | `L`（恒等） | `amount`（`apply`内で `% L` に正規化） | `amount` は `rng.randrange(L)` で生成、`apply`側で追加のmod演算により任意の整数を受理 |
| COPY | 1 | なし | `L`（恒等） | なし（parameter-free） | — |
| REVERSE | 1 | なし | `L`（恒等） | なし（parameter-free） | — |
| SORT | 1 | なし | `L`（恒等） | なし（parameter-free） | — |
| NEGATE | 1 | なし | `L`（恒等） | なし（parameter-free） | — |

**SORTは引数を持たない。** `required_argument_names = frozenset()`、`sample_params` は常に `{}`
を返し、`apply` は `tuple(sorted(sequence))`（`operations.py:279-303`）。したがってSORTの
`SYMBOLIC_REFERENCE`（§7参照）は追加実装なしに `SortOp.apply` そのものを使える。

tensor schema（トークン/値ドメイン）: `vocab_size = DEFAULT_VOCAB_SIZE = 10`
（`src/apc/environments/vocab.py:12`）、`max_sequence_length = 32`
（`DEFAULT_MAX_SEQUENCE_LENGTH`、`src/apc/primitives/primitive.py:32`）。パディングは
`apc.core.data.pad_token_sequences` / `IGNORE_INDEX = -100`
（`src/apc/core/data.py:36`、`torch.nn.functional.cross_entropy` の既定 `ignore_index` と同義）。
maskは `execute_composition_recipe` 内部で長さ情報（`current_lengths`/`output_lengths`）として
明示的に管理し、パディング位置はlossから除外される。

## 3. 有効長集合の導出（推測ではなく到達可能性から導出）

SORTの `min_input_length=1` は構造的にはあらゆる長さを許すが、**学習露出**と**合成上の到達可能性**
は別軸である。

- **学習露出済み長集合（既存カリキュラム）：** `UnifiedBenchmarkConfig.sequence_length_range = (6, 10)`
  （`src/apc/evaluation/unified_oracle_causal_benchmark.py:150`）。SORTはこの範囲でi.i.d.生成された
  入力のみで学習されてきた（`_train_single_primitive`、非parameterized分岐、
  `unified_oracle_causal_benchmark.py:536-563`）。つまり **{6,7,8,9,10}**。
- **合成上の到達可能な中間長（対象panelから導出）：** 対象7クラス（§target panel、
  `docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md`）はすべて「SORTがSELECTの直後」の形を
  持つ。`SelectOp.output_length(L) = max(1, L // 2)` を `L ∈ {6,7,8,9,10}` に適用すると
  `{3, 3, 4, 4, 5}` → **到達可能中間長 = {3, 4, 5}**。これは推測ではなく `SelectOp.output_length`
  の純関数評価そのものである。
- **本契約が登録する有効長集合：**
  `VALID_LENGTHS(SORT) = {3, 4, 5} ∪ {6, 7, 8, 9, 10} = {3, 4, 5, 6, 7, 8, 9, 10}`。
  `{3,4,5}` が `STANDALONE_QUALIFIED`/`COMPOSITION_QUALIFIED` の**新規獲得対象**、`{6,...,10}` が
  既存能力の**回帰保全対象**である。この2群を混同しない（§4のregression panelは後者専用）。

`{1, 2}` はSORTにとって構造的に`SHAPE_COMPATIBLE`だが（`min_input_length=1`）、既存の合成文法
depth≤3で「SELECT→SELECT→SORT」等を辿っても到達しない（`SELECT`を2回連続適用した最短到達長は
`max(1, max(1, L//2)//2)`; `L=6`なら`max(1,3//2)=1`)。長さ1・2の扱いは本タスクの対象panelには
現れないため、本契約は`{1,2}`を`LENGTH_SUPPORTED`として登録しない（`SHAPE_COMPATIBLE`のみ）。
将来task（D-002以降）で長さ1・2を要求する場合は、この契約を新たに拡張すること。

## 4. 合成条件（前段出力域→後段入力域）

深さ3合成 `op1 -> op2 -> op3` が `SHAPE_COMPATIBLE` であるとは、初期入力長
`L0 ∈ {6,7,8,9,10}`（生成器の既定域）に対し、
`L1 = output_length(op1, L0)`、`L2 = output_length(op2, L1)` がそれぞれ次段の
`is_valid_for_length` を満たすことである。対象7クラスすべてがこの意味で `SHAPE_COMPATIBLE`
であることはNRQ-006（`docs/research/NRQ006_REVIEW_RECORD.json`、`audit_summary`の
構造的invalid/reducible/irreducible分類）で既に確認済みであり、本契約はその結果を再検証せず
援用する。

`LENGTH_SUPPORTED` は§3の集合と比較して判定する。対象7クラスは全てSORTへの入力長が
`{3,4,5}`に属し、これは本契約で新規に`LENGTH_SUPPORTED`として登録する対象である
（現在の訓練済みモデルではまだ`STANDALONE_QUALIFIED`/`COMPOSITION_QUALIFIED`ではない —
NRQ-007のSHORT_SEQUENCE_CAPACITY_DEFICIT判定そのものがその不成立の証拠である）。

## 5. 学習露出（既存学習で実際に対象となった領域）

`_train_single_primitive`（`src/apc/evaluation/unified_oracle_causal_benchmark.py:493-601`、
非parameterized分岐 `L.536-563`）の学習batchは、毎stepごとに32例、
`seq_len ~ Uniform{6,...,10}`、各トークン `~ Uniform{0,...,9}` の完全独立生成である。
**SORTは学習中、長さ3・4・5の入力を一度も見ていない。** これは推測ではなくサンプリングコードの
直接引用である。実際にSORTの訓練に使われた既定step数は、reconstructed bundle（seed 1–4）の
構築手順（`src/apc/evaluation/learned_routing_benchmark.py:627-634`、非parameterized primitiveは
`config.bank_train_steps // 2`、既定`bank_train_steps=6000`→**3000 steps**）である。
`UnifiedBenchmarkConfig.parameter_free_train_steps=4000` という別の既定値も同じ関数系列に
存在するが、これはTask A1-B002自身の実行系列で使われた値であり、reconstructed bundle
（NRQ-005/006/007/008の評価対象そのもの）は前者（3000）で構築された。両者を混同しないこと。

## 6. 実行証拠（単独実行と連続合成実行の証拠を分離）

NRQ-007（`docs/research/STEPWISE_CAUSAL_ATTRIBUTION_NRQ007.md` §2.1、
`docs/research/NRQ007_REVIEW_RECORD.json`）は3種の証拠をすでに分離して記録している。本契約は
これをそのまま単独実行／合成実行の既存evidenceとして援用する（再実行しない）。

| 条件 | NRQ-007での名称 | 意味 | SORT-immediately-after-SELECTの7クラスでの既存値（`mean_*_step_ems`の対象primitive位置） |
|---|---|---|---|
| 単独実行（standalone） | Condition (C) | 対象primitiveだけを、対応長域の独立i.i.d.入力で単発評価 | `mean_standalone_step_ems` の該当primitiveのstep位置がおよそ0.05–0.135（`NRQ007_REVIEW_RECORD.json`の`class_attributions`より） |
| 連続合成実行（continuous） | Condition (A) | 前段の**予測**トークンをargmax・再encodeして流し込む、通常の`execute_composition_recipe`経路 | 同じ位置でおよそ0.05–0.09（`mean_continuous_step_ems`） |
| 診断reset（診断専用、PASS判定に使わない） | Condition (B) | 正解symbolic intermediateを再encodeして注入する診断専用条件 | 標準実行の成功率算出には混ぜない（§1参照） |

standaloneとcontinuousの値がほぼ一致していること（例: `SELECT->SORT->REVERSE`は
standalone 0.05 vs continuous 0.01、`NEGATE->SELECT->SORT`は standalone 0.075 vs
continuous 0.078）は、失敗が合成インターフェース（hidden-state interface）由来ではなく、
**SORTという primitive 自体が短系列で構造的に破綻している**ことを示す既存evidenceである
（NRQ-007の`SHORT_SEQUENCE_CAPACITY_DEFICIT`判定根拠と整合）。

## 7. 依存関係（Core hash / bank hash / token schema / architecture signature）

本契約は独自のhash方式を新設せず、既存の `src/apc/utils/model_bundle.py` の
`ModelBundleManifest` / `PrimitiveManifestEntry` / `canonical_state_hash` /
`compute_execution_signature` をそのまま適用する。

- **Core hash：** `ModelBundleManifest.core.canonical_state_hash`。SORT修復recipeはCoreを
  freeze/frozen（`requires_grad=False`）のまま使用し、この値を変更しない。
- **Bank hash：** SORTの重みは `PrimitiveBank` の `ModuleDict` 内で
  `_primitives.<SORTのprimitive_id>.*` として他primitiveと分離されている
  （`primitive_state_dict`、`model_bundle.py:294-306`）。修復recipeは
  この prefix 配下のテンソルのみ変更し、他のprimitive（SELECT/COUNT/BIND/SHIFT/COPY/
  REVERSE/NEGATE）の `PrimitiveManifestEntry.weights_hash` は不変であることを
  ロード前後で再計算・比較する。
- **Token schema：** `ModelBundleManifest.vocabulary.schema_hash` /
  `core.schema_hash`（一致検査は`load_bundle`が既に強制、`model_bundle.py:791-802`）。
- **Architecture signature：** SORTは `CrossPositionPrimitive`
  （`src/apc/primitives/primitive.py:334-472`）として実装され、
  `CrossPositionPrimitiveConfig(operation="SORT", d_operator=32, n_head=4,
  d_operator_ff=64, vocab_size=10, max_sequence_length=32, arg_dim=16)`
  （`src/apc/evaluation/unified_oracle_causal_benchmark.py:458-472`）で構築される。この
  configをrecipe実行前に文字列化した `architecture_signature` をSORTの
  `PrimitiveManifestEntry.architecture_signature` として固定し、修復recipeはこの値を
  変更しない（アーキテクチャ変更なし、パラメータのみ更新）。

## 8. 通常経路の保存（正解中間状態の注入によるPASS偽装を禁止）

`execute_composition_recipe`（`composition.py:194-299`）は既に「前段の`argmax`予測を
`_encode_intermediate_tokens`でtask-blind Coreに再encodeしてから次段に渡す」という設計である。
本契約はこの実装を**変更しない**。`COMPOSITION_QUALIFIED`の判定は必ずこの経路（あるいは完全に
等価な経路）を通した評価でなければならず、NRQ-007条件B（診断reset、正解intermediateの直接注入）
の結果を`COMPOSITION_QUALIFIED`のPASS根拠に使うことを禁止する。

## 9. 未完成事項（本契約が確定できないもの）

- `STANDALONE_QUALIFIED` / `COMPOSITION_QUALIFIED` はSORT-only recipeの実行結果に依存するため、
  本タスクでは`UNDETERMINED`として登録する（pilot実行前の契約はここまでで完成、実行はしない）。
- 長さ1・2はこの契約の対象panelに現れないため`LENGTH_SUPPORTED`として登録しない（§3）。
