# A1-R005E-004 結果サマリー — 凍結済み高容量オペレータの上限性能

**位置づけ:** `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md` の A1-R005E-004。測定値・詳細な考察は `docs/DECISIONS.md` ADR-0041 に記録済みで、本レポートはそれを次のブランチ判断(A1-R005E-005 に進むか、A1-R005E-006 に進むか、あるいは他の対応を取るか)のために整理したもの。**新規の実験・実装はこのレポートに含まない。**

**結論を先に:** 5 seed × 4 演算(SHIFT/SELECT/COUNT/BIND)で実行し、**総合判定は FAIL**(4演算とも全基準同時クリアはできず)。ただし内容は単純な失敗ではない。全演算で causal gap が歴史的な pointwise primitive(`ConditionedPrimitive` rank=8/arg_dim=16)を大幅に上回った(**BIND: 統計的にゼロ(-0.005) → 0.586**、**SELECT: 0.049 → 0.895**、SHIFT: 0.039 → 0.618、COUNT: 0.222(最良)→ 0.399)。特に **SELECT は Correct exact match 0.919 で 0.90 の基準をクリアし、token accuracy も 0.978 で 0.98 まであと 0.002**。**BIND も Correct 0.879 で 0.90 まであと 0.021**。SHIFT の低い平均値は 5 seed 中 2 seed が operator 学習で局所解にはまった bimodal な現象(frozen core 自体は全 seed でほぼ同一)であり、上限そのものが低いという証拠ではない。COUNT が最も弱く、`None` アーム(引数なし)の精度がやや高い(0.318)のはターゲット分布自体の偏り(count=0 が 35〜53%)による可能性がある。

---

## 1. 背景: 何を検証したか

`docs/AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.md` の Active question:

- **Representation-sufficient**: `h_content` は引数条件付き計算に十分な情報を保持しており、ボトルネックは operator クラスの方。
- **Representation-insufficient**: frozen task-blind `h_content` が情報を捨てている/絡めているため、強い operator でも解けない。

A1-R005E-002(表現監査、ADR-0039)は SHIFT/SELECT の per-position 情報が強く保持されている(0.94〜1.00)一方、単一ベクトルへの系列再構成は全演算で失敗することを示した。A1-R005E-003(oracle latent operator、ADR-0040)は、frozen core 自身の pretrained decode head を content 領域の状態に再利用すると SHIFT/SELECT/BIND が(chance 未満まで)失敗する一方、**COUNT だけ decode head を新規学習済み readout に差し替えると 0.973 まで回復する**ことを示し、「decoder/interface のミスマッチ」が SHIFT/SELECT/BIND 失敗の相当部分を占めている可能性を指摘した。

A1-R005E-004 はこの2つの知見を踏まえ、「frozen `h_content` + 引数から、**新規学習した cross-position operator + 新規学習した readout**(古い decode head は使わない)で実際に解けるか」という `docs/design-docs/REPRESENTATION_OPERATOR_ISOLATION.md` セクション3の **R3(タスク上限)** を直接検証するタスク。

---

## 2. 実装したもの(アーキテクチャ概要)

`src/apc/evaluation/frozen_high_capacity_operator_benchmark.py`(新規モジュール)。

- **既存の `PrimitiveBank`/`Router`/`ConditionedPrimitive` は使わない**(スタンドアロン診断)。理由: `apc.core.execution._route_and_apply_oracle_calls` は `hidden` を `[batch * position, d_model]` にフラット化してから primitive を呼ぶため、既存の `ArgumentConditionedPrimitive` はどれも「その位置自身の状態 + 引数」しか見られない pointwise 変換に構造的に限定される(ADR-0039 の診断)。`docs/AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.md` の "No premature operator rollout" により、この診断が終わる前に `apc.core.execution` 自体を書き換えることは範囲外。
- **`HighCapacityOperator`**(新規 `nn.Module`): 引数トークン(`apc.primitives.conditioning.default_argument_encoder` で埋め込み、この operator だけに入る — content encoder には一切入らない)・凍結された content の各位置の状態・出力スロット数ぶんの学習可能な "answer-query" トークンを連結し、双方向 self-attention ブロック(既定 3 層、`d_operator=256`, `n_head=4`, `d_ff=1024`)に通す。SHIFT(出力長=入力長)・SELECT(出力長=`max(1, L//2)`)・BIND/COUNT(出力長=1)を **同一アーキテクチャ**で扱う(演算ごとの特別扱いなし)。
- 最終読み出しは新規初期化した `nn.Linear(d_operator, vocab_size)`(ADR-0040 の教訓 — 古い decode head の使い回しはしない)。演算全体を通じてこの readout ごと end-to-end で学習。
- 学習・評価はこれまでの A1-R005D 系 counterfactual gate と同じ反実仮想(counterfactual)プロトコル: 同一 content に対し出力が互いに異なる複数の引数値を用意し(`generate_high_capacity_operator_counterfactual_groups`)、Correct / effectful Wrong argument / None(引数トークンをゼロ化)の3アームで評価し、`causal_gap = correct - max(wrong, none)` を算出。
- Stable Core は各 (seed, operation) ごとに専用に事前学習(`d_model=192, n_layer=4, n_head=4, d_ff=768`, `steps=30000`)して凍結(A1-R005E-002/003 と同一予算 — 同じ frozen `h_content` を検証している)。Operator 側は 8000 ステップ。

---

## 3. 結果

5 seeds (`0`–`4`)、RTX 5060 Ti(ネイティブ Windows, `torch==2.11.0+cu128`)、総 wall-clock 約 8929 秒(約2時間29分)。アーティファクト: `runs/phase_a1_frozen_high_capacity_operator_benchmark/`。

### 3.1 全体サマリー

| Operation | Correct exact(mean, stdev, min–max) | Correct token acc. | Wrong exact | None exact | Causal gap(exact / token) | `passed` |
|---|---|---|---|---|---|---|
| SHIFT | 0.618(0.476, 0.097–0.993) | 0.952 | 0.000 | 0.0002 | **0.618** / 0.740 | false |
| SELECT | **0.919** | 0.978 | 0.0003 | 0.024 | **0.895** / 0.636 | false |
| COUNT | 0.716(0.090, 0.631–0.850) | 0.716 | 0.129 | 0.318 | 0.399 / 0.399 | false |
| BIND | 0.879(0.102, 0.708–0.956) | 0.879 | 0.032 | 0.292 | **0.586** / 0.586 | false |

**基準:** Correct exact `>= 0.90`、SHIFT/SELECT token acc. `>= 0.98`、effectful Wrong argument `<= 0.30`、None `<= 0.30`(タスク文に明記なし、これまでの gate と同じ値を踏襲)、causal gap `>= 0.50`。

**基準ごとの合否:**

| Operation | Correct exact | Token acc. | Wrong | None | Causal gap | 総合 |
|---|---|---|---|---|---|---|
| SHIFT | FAIL(0.618) | FAIL(0.952) | PASS | PASS | PASS(0.618) | FAIL |
| SELECT | **PASS**(0.919) | FAIL(0.978、あと0.002) | PASS | PASS | PASS(0.895) | FAIL |
| COUNT | FAIL(0.716) | (対象外) | PASS | FAIL(0.318、上限0.30をわずかに超過) | FAIL(0.399) | FAIL |
| BIND | FAIL(0.879、あと0.021) | (対象外) | PASS | PASS | **PASS**(0.586) | FAIL |

`argument_effect_rate = 1.000`(全演算・全 seed — グループ構成上の不変量であり測定で確認)。`meets_seed_policy = true`(5 seed、タスク文の "run >= 5 seeds" を満たす)。

### 3.2 歴史的結果(A1-R005D 系 counterfactual gate、pointwise `ConditionedPrimitive`)との比較

| Operation | 過去の最良 causal gap(pointwise primitive) | 今回(cross-position operator) | 倍率/質的変化 |
|---|---|---|---|
| BIND | **-0.005**(ADR-0036、統計的にゼロ) | **0.586** | ゼロ効果 → 明確な効果 |
| SELECT | 0.049(exact)/ 0.097(token)(ADR-0037) | 0.895(exact)/ 0.636(token) | 約18倍 / 約7倍 |
| SHIFT | 0.039(ADR-0037) | 0.618 | 約16倍 |
| COUNT | 0.222(容量スイープの最良値、ADR-0034) | 0.399 | 約1.8倍 |

### 3.3 演算ごとの深掘り

**BIND — ADR-0036 の仮説を直接裏付ける結果。** ADR-0036 は「BIND は連想検索であり、rank-8 の加算的残差には引数でどの位置に注意を向けるかを制御する構造がない」と推測していた。本タスクの operator はまさにその機構(content 全位置への self-attention)を持ち、causal gap は `-0.005`(効果なし)から `0.586` へ移動した。Correct(0.879)は 0.90 まであと 0.021、Wrong(0.032)・None(0.292)はいずれも余裕を持って基準をクリア。この診断チェーンの中で最も強い「operator クラスがボトルネックだった」という直接証拠。

**SELECT — 最も基準に近い。** Correct exact match(0.919)は基準をクリア、causal gap(0.895)も極めて大きい。Wrong(0.0003)・None(0.024)ともにほぼゼロ。`passed=false` を妨げているのは token accuracy(0.978 対 0.98、差は 0.002)のみで、これは Correct exact match 自体の seed 間ばらつき(stdev 0.083)より小さい。

**SHIFT — 平均値の低さは bimodal な学習収束の問題であり、上限の証拠ではない。** seed 別の Correct exact match: seed0 `0.097`、seed1 `0.099`、seed2 `0.992`、seed3 `0.993`、seed4 `0.909`。5 seed中2つが悪い局所解(operator 最終学習損失 `~0.305`)にはまり、残り3つはほぼ天井(`0.0002`〜`0.047`)まで収束。一方 frozen core 自体の `final_core_train_loss` は全 seed でほぼ同一(`0.2330`〜`0.2366`)——つまり operator に渡される表現は seed 間で一貫しており、失敗しているのは **operator 自身の 8000 ステップ最適化**(固定学習率、warmup/schedule なし)。「収束すれば解ける」ことは3 seed が示しており、この学習レシピが全 seed で頑健に収束していないだけの可能性が高い。

**COUNT — 最も弱い。** Correct(0.716)・causal gap(0.399)ともに基準未達。None アーム(0.318)がわずかに上限(0.30)を超えているが、これは `CountOp` のターゲット分布自体の偏り(`vocab_size=10`, 長さ6〜10の一様ランダム content に対して `count=0` の確率がおよそ35〜53%)による base-rate 的な寄与の可能性がある——引数なしでも「多数派クラス(0 付近)」を答えるだけである程度の精度が出てしまう。Wrong アーム(0.129)は Correct より明確に低く、operator が引数の identity には反応していることは確認できるが、精度が不十分。

---

## 4. ブランチ判断のための整理(`docs/design-docs/NEXT_PHASE_DECISION_MATRIX.md` との対応)

本タスク自身は分岐を選択しない(`AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.md` の "Branching discipline" — 最終判断は A1-R005E-008 かユーザー)。以下は判断材料の整理。

### Branch A(Operator / Heterogeneous Primitive)を支持する材料
- BIND: 統計的にゼロだった causal gap が 0.586 まで拡大(質的な転換)。
- SELECT: ほぼ完全に基準をクリア(token accuracy のみ 0.002 不足)。
- 4演算すべてで、過去の pointwise primitive を大幅に上回る causal gap。
- Branch A の条件のひとつ「historical low-rank conditioned primitive fails」はすでに成立している(A1-R005 retry 全体で確認済み)。

### Branch A を選ぶにはまだ弱い材料
- 4演算とも「厳密に全基準同時クリア」はしていない(`passed=true` が0件)。
- Branch A の残り条件「compact cross-position operator が materially gap を縮める」は A1-R005E-005(未実施)で検証する必要がある——今回の operator は意図的に primitive スケールを超える大きさ(`operator_param_count ~= 2.44M` 対 `core_param_count = 1,795,968`)であり、この性能がコンパクトな operator でも再現できるかは未知数。

### Branch B(Queryable Representation Learning)を支持する材料
- COUNT は causal gap 0.399 で基準(0.50)に届かず、None アームの上限超過も示している——4演算中もっとも「representation-insufficient」寄りの結果。

### Branch B を選ぶにはまだ弱い材料
- SELECT/BIND は frozen upper bound がほぼ/明確に効いており、Branch B の前提「frozen upper bound fails/is clearly limited」を4演算横断では満たしていない。
- Branch B 自体の検証(A1-R005E-006、joint task-blind representation control)は未実施。

### Branch C(Interface / Factorization Reconsideration)を支持する材料
- 今回時点では乏しい。むしろ A1-R005E-003(ADR-0040)が示した「decode head の interface ミスマッチ」問題を、新規 readout の joint training で回避したところ、SELECT/BIND で大きな改善が出た——これは interface 側の問題が(この形の対処で)相当程度解消可能であることを示唆しており、Branch C(致命的な interface/factorization 上の欠陥)を積極的には支持しない。

### Branch D(Mixed result)が現状もっとも近い読み方
- `docs/design-docs/NEXT_PHASE_DECISION_MATRIX.md` の Branch D は「演算ごとに異なる診断結果になった場合、無理に1つの primitive クラスに揃えない」ことを求めている。今回の結果はまさにこの形:
  - BIND/SELECT → Operator branch 寄り(ほぼ通過、質的な改善)
  - COUNT → Representation branch 寄り、または少なくとも Operator branch への強い反証にはならない中間結果
  - SHIFT → 判定保留(最適化次第で Operator branch 寄りになりうる、現状のデータでは結論を出せない)

---

## 5. 未解決の論点・キャビエト

1. **SHIFT の bimodality は未解決。** 学習率スケジュール(warmup 付き cosine 等)やステップ数を増やす、あるいは複数初期化からベストを選ぶといった対処で 5 seed 全てが収束するかどうかは未検証。現状の 0.618 という平均値は「真の上限」の過小評価である可能性が高い。
2. **COUNT の None アーム上振れの base-rate 仮説は厳密に検証していない。** 「引数なしで多数派クラスを答えるだけでどこまで精度が出るか」を今回の反実仮想グループ構成(3つの pairwise-distinct な出力を要求するフィルタ)のもとで正確に計算する追加分析があれば、None ceiling 超過がどこまで base-rate 由来かをより明確にできる。
3. **今回の operator は primitive スケールを大きく超える**(`~2.44M` パラメータ、frozen core 本体の `1,795,968` より大きい)。これは意図的な上限測定(`AGENTS_A1_R005E_DIAGNOSTIC_ADDENDUM.md` の "Diagnostic-only rule")であり、コンパクトな operator でどこまで再現できるかは A1-R005E-005 で初めて分かる。
4. **本タスクは `Router`/`PrimitiveBank` を一切通していない。** 学習ルーティングが正しく機能するかどうかについては何も語っていない(oracle-forced ではなく学習ルーティングの評価は、Operator branch が選ばれた場合の後続タスクの範囲)。

---

## 6. 関連ファイル

- ADR: `docs/DECISIONS.md` ADR-0041
- タスク仕様: `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md`(A1-R005E-004)
- 実験計画: `docs/EXPERIMENT_PLAN_A1_R005E_DIAGNOSTIC.md`(D-E3)
- 設計ドキュメント: `docs/design-docs/REPRESENTATION_OPERATOR_ISOLATION.md`(セクション5)
- 分岐判断マトリクス: `docs/design-docs/NEXT_PHASE_DECISION_MATRIX.md`(まだ未記入 — 記入は A1-R005E-008 の役割)
- 実行アーティファクト: `runs/phase_a1_frozen_high_capacity_operator_benchmark/`(`config.yaml`, `report.json`, `summary.json`, `system.json`, `seed_<n>/{report.json, <OP>_core_metrics.jsonl, <OP>_operator_metrics.jsonl}`)
- 実装: `src/apc/evaluation/frozen_high_capacity_operator_benchmark.py`, `scripts/frozen_high_capacity_operator_benchmark.py`, `configs/phase_a1_frozen_high_capacity_operator_benchmark.yaml`, `tests/test_frozen_high_capacity_operator_benchmark.py`

**関連する過去の ADR:** ADR-0029(元の A1-R005 STOP GATE FAIL)、ADR-0033/0034/0035(COUNT の容量/formula探索)、ADR-0036(BIND: causal gap 統計的にゼロ、連想検索仮説)、ADR-0037(SHIFT/SELECT: 複合ペナルティ)、ADR-0039(表現監査)、ADR-0040(oracle latent operator、decode head interface 問題)。
