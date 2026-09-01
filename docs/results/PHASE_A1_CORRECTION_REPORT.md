# Phase A.1 Correction 結果レポート(A1-C001〜A1-C007)

**位置づけ:** `docs/CODEX_TASKS_PHASE_A1_CORRECTION.md` の A1-C001〜A1-C007。挿入点は「A1-006の後、A1-007の前」(`docs/exec-plans/active/PHASE_A1_CORRECTION.md`)。

**このレポートの役割:** A1-C008(Correction audit / ADR)が要求する監査内容——H1a(A1-006)結果・H1b(shared-core)結果・probe結果・parameterized primitive決定・A1-007前提への変更・次タスクのブロック解除有無——を1ファイルに集約したもの。個別の設計判断は既存の `docs/DECISIONS.md` ADR-0017〜ADR-0024にすでに記録済みで、本レポートはそれらを結果ベースで串刺しにする「progress report」側の成果物(A1-C008は「ADR or short progress report」のどちらでもよいとしている)。**新規アーキテクチャ変更はこのレポートに含まない**(A1-C008のルール「Audit only」に準拠)。

**結論を先に:** A1-C001〜A1-C007の acceptance criteria はすべて PASS。`docs/EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md` section 4 の「A1-007 unblocked」4条件も全て充足。**A1-007(Oracle primitive routing)は着手可能な状態。**

---

## 1. 背景: なぜ A1-006 の後に補正フェーズが必要だったか

A1-006 (`dd060ed`) は STOP GATE として PASS したが、**1操作=1モデル**という設計で検証していた(H1a)。これは「後段の APC ルーターが必要とする能力」——1つの共有 Stable Core がタスク識別を表現し、有用なルーティング状態を持てるか——を証明しない。この補正フェーズは H1b/H1c として、その能力を A1-007 着手前に検証する。

---

## 2. タスク別結果

### A1-C001 — Explicit task specification schema(`a3ce059`)

**目的:** 出力を左右するタスク変数をすべてモデル可視にする。

**実装:** `apc.environments.task_spec.TaskSpec`/`TaskStepSpec` を追加。`Program` の model-visible な射影として、operation identity と `SELECT.indices`/`COUNT.target`/`SHIFT.amount`/`BIND.query_key`(ADR-0017 で発見された4つの隠しパラメータ)を含む。`TaskGenerator` が生成する全 `Example` に `Example.task_spec` として付与。`OracleMetadata`(K/C/N/R 評価ラベル)とは明確に分離。

**受け入れ基準:** PASS — 同一コンテンツでも task specification が異なれば正しく異なるターゲットを持てる/task spec が全隠しパラメータを決定する/model-facing path が oracle-only フィールドを読まない、の3点をテストで確認。

### A1-C002 — Mixed-operation online generator(`406665a`)

**目的:** 識別可能な混合操作オンライン学習ストリームを1本生成する。

**実装:** `apc.environments.generator.build_mixed_operation_generator` を追加。`max_depth=1` に固定した `TaskGenerator` で、`KNOWN_OPERATION_NAMES` 全8操作(ADR-0017 の4つの隠しパラメータ操作を含む)をプールする。`permute_symbols=False` がプライマリ設定。

**受け入れ基準:** PASS — 全操作が1ストリームに出現する/同一コンテンツが複数操作と正しく対応する/task spec + content のみでターゲットを再現できる、の3点をテストで確認。

### A1-C003 — Shared-core input encoding(`e693c34`)

**目的:** 既存 Stable Core が1モデルでタスク/制御情報とコンテンツ情報の両方を消費できるようにする。

**実装:** `apc.core.tokens.SharedCoreTokens`/`build_shared_core_tokens` を追加。`[BOS] [TASK_START] op arg... [TASK_END] input... [SEP] target... [EOS]` という形式でタスクセグメントを埋め込む。`apc.core.data.encode_task_spec` が変換を担当し、`encode_example`/`evaluate_exact_match` は `include_task_spec=True` でオプトイン(既定 False、既存呼び出し元は byte-for-byte 不変)。`apc.core.model.register_encode_split_probe` も追加(A1-C005 の確率プローブ用フック)。

**受け入れ基準:** PASS — 1モデルフォワードで全操作を処理できる/task状態とcontent状態が別々に取得できる/既存 A1-006 経路が再現可能なまま、の3点を確認。

### A1-C004 — Shared-core systematic-generalization gate(`976afd8`, STOP GATE, H1b)— **PASS**(ADR-0021)

**目的:** H1b「1つの共有 Stable Core が明示的タスク仕様に条件づけて複数操作・複数引数に汎化できる」を検証。

**設定:** `configs/phase_a1_shared_core_gate.yaml`、5 seeds (`0-4`)、RTX 5060 Ti、commit `e693c34`。DecoderOnlyTransformer(192d, 4層, 約1.80M params)。`build_mixed_operation_generator` の混合オンラインストリーム、`include_task_spec=True`。

**結果:**

| 指標 | 結果 | 閾値 |
|---|---|---|
| 全体 unseen exact match(平均) | **0.9946** | ≥0.95 |
| 全体 stdev / seed 最小値 | 0.0030 / 0.9922 | — |
| 各操作の平均(最低は SHIFT) | COPY 0.9996, SELECT 0.9946, COMPARE 0.9989, COUNT 0.9863, SHIFT 0.9842, BIND 0.9981, NEGATE 0.9977, ACCUMULATE 0.9982 | 各 ≥0.90 |
| seed×operation の 0.85 未満件数 | 0 | 0 |
| Negative control(task spec 除去)平均 | 0.139 | 参考 |
| Explicit − Negative control ギャップ | **0.856** | ≥0.20(有意差) |

**受け入れ基準:** 3項目すべて PASS。Negative control が大幅に劣化することで、効果が「task spec を明示したこと」自体に起因すると確認(付随的な学習設定変更の効果ではない)。

**副次観察(非ブロッキング):** ACCUMULATE の negative control スコアが 0.175 とやや高め(他の隠しパラメータ操作は <0.10)。ゲートの pass/fail には影響しないため、将来この結果が別の結論の根拠になる場合にのみ追加調査。

### A1-C005 — Task/content representation probes(`6fb92fe`, STOP GATE, H1c)— **PASS**(ADR-0022)

**目的:** H1c「`z_task`/`h_content` の分離が実際に有用な情報を含むか」を検証。

**設定:** `configs/phase_a1_task_content_probes.yaml`、3 seeds (`0-2`)、CPU、commit `976afd8`(A1-C004 と同一学習コード)。`[TASK_END]` 位置(因果的にコンテンツを見ていない)で `z_task` を、コンテンツ各位置で `h_content` を線形プローブ。

**結果:**

| プローブ | 内容 | 結果 | 閾値 |
|---|---|---|---|
| T1 | `z_task` からの operation identity | **1.0**(全seed) | ≥0.95 |
| T2 | `z_task` からの operation argument | COUNT/SHIFT/BIND: 1.0、SELECT.indices: 0.9758/0.9419/0.9592 | ≥0.90(各) |
| C1 | `h_content` からの content-token identity | 0.9810/0.9928/0.9878(平均0.9872) | ≥0.98(事前宣言の厳しめ閾値) |
| (optional leakage) | operation from content_state / content from task_state | いずれも高値(構造的に予測される因果マスキングの帰結) | 非ゲート |

**受け入れ基準:** 3 seed 全てで PASS。

**留保事項(非ブロッキング):** seed 0 の Probe C1 マージンが +0.0009 とやや薄い。0.98 は厳しめに事前宣言された閾値であり、3 seed とも独立にクリアしているため failure とは扱わない。この結果が将来的に他の結論の根拠になる場合は追加のシード数でのサニティチェックが望ましい。

### A1-C006 — Parameterized PrimitiveCall abstraction(`564d95b`)— PASS(ADR-0023)

**目的:** primitive の選択と引数を分離した実行時抽象を追加する。

**実装:** `apc.environments.primitive_call.PrimitiveCall`(`operation` + `arguments`)。`Operation.required_argument_names` を新設し、構築時に必須/余剰引数を即座に検証。`primitive_id` は `operation` のみの関数(引数値に非依存)。`SELECT` は設計ドキュメントの単純化された `{"index": 3}` ではなく、実際の `SelectOp` が使う `{"indices": [...]}` 部分集合表現を採用(ADR-0023)。

**受け入れ基準:** PASS —
- 1つの primitive family が3つ以上の異なる引数値を正しく実行(SHIFT/SELECT/COUNT/BIND それぞれ)
- 100個の `PrimitiveCall`(50+50の異なる引数値)を作っても registry 上の family 数は不変
- 引数の欠落/余剰を構築時にエラーで検出

608件の既存+新規テスト全通過、ruff/mypy クリーン。ただしこの時点では **`apc.core.execution`/`apc.primitives` への配線は未実施**(A1-C007 のスコープとして明示的に持ち越し)。

### A1-C007 — Oracle PrimitiveCall routing adapter(`9099726`)— PASS(ADR-0024)

**目的:** A1-007 のオラクルが「primitive ID だけでなく完全な実行可能コール」を渡せるように準備する。

**実装:**
- `apc.environments.generator.oracle_calls_for_example`/`oracle_call_for_example` — `Example.program` を `PrimitiveCall` チェーンに変換(Work item 1)。深さ>1(`novel_composition`)は明示的に `ValueError`(A1-008 のスコープ)。
- `apc.core.execution.apply_bank_with_oracle_calls`/`forward_logits_with_oracle_calls`/`generate_greedy_with_oracle_call`/`evaluate_exact_match_with_oracle_calls` — バッチ要素ごとに1つの oracle `PrimitiveCall` から bank primitive 選択を強制する、`Router` を一切引数に取らない新規関数群(Work item 2: 学習経路への到達可能性が構造的にゼロ)。`evaluate_exact_match_with_oracle_calls` は `oracle_call_provider` を注入可能にし、A1-007評価APIの土台となる(Work item 3)。

**受け入れ基準:** PASS —
- oracle call が primitive 実行を完全に決定する(全8操作でターゲット完全一致をテストで確認)
- A1-006由来の隠しパラメータ問題がオラクルモードで再発しない(パラメータ化4操作は引数なしでは `PrimitiveCall` 自体を構築不可能、という型レベルの保証で担保)
- A1-007のブロック解除

`tests/test_oracle_routing.py` 新規46件を含め653件全通過、ruff/mypy クリーン。

---

## 3. Routing-readiness criterion(A1-007解禁の4条件)

`docs/exec-plans/active/PHASE_A1_CORRECTION.md` section 4 は、以下4条件がすべて満たされて初めて A1-007 に着手できるとしている。

| # | 条件 | 充足根拠 | 状態 |
|---|---|---|---|
| 1 | H1b が pass する | ADR-0021(A1-C004) | ✅ |
| 2 | `z_task` が operation identity を確実に予測する | ADR-0022 Probe T1(A1-C005) | ✅ |
| 3 | 必要なタスク引数がモデル可視である | ADR-0022 Probe T2(A1-C005)+ A1-C001/A1-C003 のタスクセグメント配線 | ✅ |
| 4 | oracle routing が `PrimitiveCall` として表現できる | ADR-0023(抽象の追加, A1-C006)+ ADR-0024(実行系への配線, A1-C007) | ✅ |

**4条件すべて充足。A1-007(Oracle primitive routing)は着手可能。**

---

## 4. A1-007 に持ち越される既知の制約

このレポートはA1-C008の「Audit only」ルールに従い新規アーキテクチャ変更を含まないが、A1-007 実装時に踏まえるべき制約を明記しておく(いずれもA1-C007のスコープ外として意図的に据え置かれたもの)。

1. **ニューラル primitive はまだ引数非対応。** `apc.primitives.primitive.Primitive.forward` は `arguments` を一切受け取らない(`h + gate * B(A(h))` のまま)。`PrimitiveCall.arguments` は `OracleRoutingOutput.calls` を通じて保持されるが、低ランク変換自体はまだそれを消費しない。引数依存の primitive 実行が必要な A1-007 の実験がある場合、これは別途対応が必要。
2. **bank への operation 整列 primitive の事前登録は A1-007 自身の実験セットアップの仕事。** `apply_bank_with_oracle_calls` は `call.primitive_id`(`apc.environments.task_spec.operation_id` そのもの)を bank の lookup key としてそのまま使うため、A1-007 のベンチマークが各操作について `operation_id(name)` の id で bank primitive を登録する必要がある。
3. **複数ステップの oracle 合成実行は未実装。** `oracle_calls_for_example` は `novel_composition` 例に対して複数 `PrimitiveCall` のチェーンを返せるが、それを連結実行する機構は A1-008(Composition Library)のスコープ。
4. **ACCUMULATE の negative control やや高め(0.175)、Probe C1 seed 0 のマージンが薄い(+0.0009)。** いずれも該当ゲートの pass/fail には無関係だが、将来これらの数値が別の結論の根拠になる場合は追加調査が望ましい(ADR-0021/ADR-0022 に詳細記録済み)。

---

## 5. テスト・検証コマンド

```bash
python -m pytest -q          # 653 passed, 1 skipped(無関係な既存skip)
python -m ruff check .       # All checks passed!
python -m mypy src/apc       # Success: no issues found in 46 source files
```

Run artifacts(gitignore対象、ローカルに存在):
- `runs/phase_a1_shared_core_gate/`(A1-C004, config `configs/phase_a1_shared_core_gate.yaml`)
- `runs/phase_a1_task_content_probes/`(A1-C005, config `configs/phase_a1_task_content_probes.yaml`)

---

## 6. 参照

- `docs/CODEX_TASKS_PHASE_A1_CORRECTION.md` — A1-C001〜A1-C008 タスク定義
- `docs/exec-plans/active/PHASE_A1_CORRECTION.md` — 補正フェーズの実行計画・routing-readiness criterion
- `docs/EXPERIMENT_PLAN_PHASE_A1_CORRECTION.md` — H1b/H1c 実験計画
- `docs/DECISIONS.md` ADR-0017〜ADR-0024 — 個別の設計判断・測定結果の詳細記録
- `docs/design-docs/PARAMETERIZED_PRIMITIVE_CALLS.md` — `PrimitiveCall` 設計ドキュメント

---

## 7. 次のステップ

A1-C008(このレポートで内容的にはカバー済み)を正式クローズとするか、直接 A1-007(Oracle primitive routing)に進むかはユーザー判断。`docs/CODEX_TASKS_PHASE_A1.md` の A1-007 定義:

> **Goal:** 学習済みルーティングとは独立に primitive 実行を検証する。
> **Work:** 環境が oracle primitive ID を oracle controller に供給する。学習済みルーターは完全にバイパスされる。
> **Accept:** oracle-routed K ≥0.95; 選択された ID がメタデータと完全一致する。

本レポートの§4「持ち越される既知の制約」の1・2番(引数非対応の primitive、bank 事前登録)は A1-007 実装時に最初に対処が必要になる見込み。
