# Phase A.1 Post-Correction 結果サマリー(A1-R001〜A1-R005)

**位置づけ:** `docs/CODEX_TASKS_PHASE_A1_POST_CORRECTION.md` の A1-R001〜A1-R005。挿入点は `docs/exec-plans/active/PHASE_A1_POST_CORRECTION.md` の冒頭(A1-C007 / correction audit の後)。個別の設計判断・測定値はすべて `docs/DECISIONS.md` ADR-0025〜ADR-0029 に記録済みで、本レポートはそれらを結果ベースで串刺しにしたもの。**新規の実験・実装はこのレポートに含まない**(既存 ADR の要約と、次の一手を検討するための考察のみ)。

**結論を先に:** A1-R001〜A1-R004 はすべて PASS(A1-R004 は STOP GATE ではなく module 実装)。**A1-R005(STOP GATE, H2c)は FAIL** — Correct 0.308(要求 ≥0.90)、causal gap 0.053(要求 ≥0.50)。`AGENTS.md` の STOP GATE discipline により、A1-R006 以降は現在ブロックされている。

---

## 1. 背景: 何を検証している一連のタスクか

`AGENTS.md` の現在の科学的問い:

> 演算固有の計算を、Stable Core の中で直接解かせるのではなく、疎で再利用可能な primitive に因果的に依存させられるか?

A1-R001〜A1-R005 はこの問いを段階的に検証する進行(`docs/exec-plans/active/PHASE_A1_POST_CORRECTION.md`):

1. task-blind content path(R001)
2. decoder leakage control(R002)
3. parameter-free primitive causality(R003)
4. argument-conditioned neural primitives(R004, module実装のみ)
5. parameterized primitive causality(R005)

各 causal primitive gate は `docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md` section 8 の **causal ablation matrix**(Correct / Wrong / None、パラメータ化された演算では追加で Wrong argument)で評価される。「Correct primitive → 高精度、Wrong primitive → 低精度、No primitive → 低精度」というパターンが揃って初めて、primitive が実際にその計算を行っていると解釈できる(`AGENTS.md` 「Causal primitive evidence rule」)。

---

## 2. タスク別結果

### A1-R001 — Task-blind content encoder path(STOP GATE, H2a)— **PASS**(ADR-0025)

**目的:** `h_content = f(content)` であって `h_content = f(task, content)` にならないことを、経験的なリーク検査ではなく **構造的に** 保証する。

**実装:** `apc.core.model.DecoderOnlyTransformer.encode_task_content_split` が、task-only トークン列と content-only トークン列を **別々の** forward pass として同じ共有重みに通す(`apc.core.data.build_task_only_tokens`/`build_content_only_tokens`)。content-only 列にはそもそも task トークンが1つも含まれないため、リークは「起きていない」ではなく「起こりようがない」。

**結果:** config `configs/phase_a1_task_blind_content_gate.yaml`、5 seeds(0-4)、CPU、commit `1ed8940`。

| 指標 | 結果 | 判定基準 |
|---|---|---|
| content-only 列への task トークン混入 | 0件(全seed) | 0 |
| 8操作全カバレッジ | ✓(各操作専用 generator で構造的に保証) | 全操作 |
| 同一 content・異なる task 間の `content_state` 最大絶対差 | **0.0**(CPU上で厳密一致) | 許容誤差以内 |
| padding invariance 最大絶対差 | 2.3e-6〜3.1e-6 | `atol=1e-5` 以内 |

**含意:** 未学習モデルでも成立する **アーキテクチャ的** 性質として検証(学習は不要 — 「最小実装で仮説を反証できるものを優先する」という `AGENTS.md` の原則どおり)。

---

### A1-R002 — Decoder leakage control(conditional STOP GATE)— **PASS**(ADR-0026)

**目的:** causal-mode の decode 経路に task 情報が迂回して流れ込んでいないことを監査し、primitive なしのベースライン(causal ablation matrix の "None" 枝)が将来の Correct 目標に対して十分低いことを確認する。

**実装:** `apc.core.execution` 内の全 causal-mode decode 関数の入力トークンを監査(`(bos,) + input_tokens + (sep,)` のみで、task segment は一切含まない)。`forward_logits_no_primitive`/`generate_greedy_no_primitive`/`evaluate_exact_match_no_primitive` を新設(bank/router/workspace パラメータ自体が存在しない = 迂回不可能)。

**結果:** config `configs/phase_a1_decoder_leakage_gate.yaml`、5 seeds(0-4)、RTX 5060 Ti、commit `6948360`。

| 指標 | 結果 | 判定基準 |
|---|---|---|
| None(task-blind decoder-only)全体 exact match(平均) | **0.1401**(stdev 0.0049) | ≤0.30、かつ将来の Correct 目標 0.95 より十分低い |
| 演算別内訳 | COPY 0.754 / ACCUMULATE 0.209 / SHIFT 0.092 / COMPARE 0.028 / NEGATE 0.013 / BIND 0.012 / COUNT 0.003 / SELECT 0.0 | — |
| 監査した decode 経路との数値一致 | 全seedで bit-identical | — |

**含意:** COPY の 0.754 は「値を見ず位置だけで解ける value-blind な恒等写像」であるため妥当な例外(ADR-0020/0021)。それ以外は軒並み低く、以後の primitive gate の "None" 枝は "既にバイパスで解けている" 状態ではないことを確認。

---

### A1-R003 — Parameter-free oracle primitive benchmark(STOP GATE, H2b)— **PASS**(ADR-0027)

**目的:** `COPY`/`NEGATE`/`COMPARE`/`ACCUMULATE`(隠しパラメータを持たない4操作)について、primitive が実際に計算を担っていることを causal ablation matrix で示す。

**実装:** (1) task-blind に事前学習した Stable Core を凍結、(2) 演算ごとに1つの `Primitive`(rank 8)を fresh な `PrimitiveBank` に登録、(3) oracle-forced routing(新設 `apply_bank_with_oracle_calls_trainable`)で primitive のみを学習、(4) Correct(正しい演算)/Wrong(固定 derangement で強制した誤った演算)/None(A1-R002 の "None" 枝を再利用)を評価。

**結果:** config `configs/phase_a1_parameter_free_primitive_gate.yaml`、5 seeds(0-4)、RTX 5060 Ti、commit `0bbb290`、総 wall-clock 3954秒(約66分)。

| Arm | 平均 | 判定基準 |
|---|---|---|
| Correct | **0.9999**(stdev 0.0002、最小0.99951) | ≥0.95 |
| Wrong | **0.0**(全seed・全演算) | ≤0.30 |
| None | 0.2482 | ≤0.30 |
| causal gap(Correct − max(Wrong, None)) | **0.7517** | ≥0.50 |

演算別 Correct: COPY 1.0 / NEGATE 1.0 / COMPARE 1.0 / ACCUMULATE 0.9996 — 外れ値なし。primitive の学習パラメータ数は12,288(frozen core 1,795,968 の 0.7%未満)。

**含意:** `AGENTS.md` の「Correct primitive → 高精度、Wrong primitive → 低精度、No primitive → 低精度」パターンが Phase A.1 Post-Correction で初めて end-to-end に実証された。Wrong が厳密に0.0であることは、primitive が「それらしい答えに寄せる」のではなく「その演算自体を計算している」ことの最も強い証拠。

---

### A1-R004 — Argument conditioning module(STOP GATEではない、module実装)— **PASS**(ADR-0028)

**目的:** neural primitive に `PrimitiveCall.arguments` を実際に消費させる仕組みを作る(`SHIFT(amount)`/`SELECT(indices)`/`COUNT(target)`/`BIND(query_key)`)。

**実装:** `apc.primitives.conditioning` を新設。

- `IntBucketArgumentEncoder`(有界整数1個 → embedding lookup、`value % num_buckets` で範囲外値も wraparound)が `SHIFT.amount`/`COUNT.target`/`BIND.query_key` を、`IndexSetArgumentEncoder`(可変長インデックス集合の平均 embedding)が `SELECT.indices` をカバー。
- `ConditionedPrimitive(Primitive)` が `docs/design-docs/CAUSAL_PRIMITIVE_EXECUTION.md` section 6 の最小構成式を実装: `e_a = encoder(argument); u = A_i(h); c = C_i(e_a); delta = B_i(phi(u + c)); out = h + gate * delta`。`a_proj`/`b_proj`/`c_proj`/`argument_encoder` は値によらず共有(family あたり1インスタンス)。
- `argument_values` は keyword-only・デフォルトなし = 引数を渡し忘れると即 `TypeError`(「ログには残るが実際には使われない引数」を構造的に禁止)。

**結果:** STOP GATE ではないため多seedベンチマークは行わず、`tests/test_primitives_conditioning.py`(42ケース)で unit level に受け入れ基準を確認: 引数を変えると出力が変わる/同一familyが3値以上を処理/bank の family数が値によって増えない/invalid(範囲外)・missing(未指定)の両方をテスト済み。

**含意:** ここまでは「引数を条件として実際に消費できる **能力**」を単体レベルで示したにとどまり、「学習によってその能力が実際に使われるようになるか」は A1-R005 の役割として明示的に残されていた。

---

### A1-R005 — Parameterized oracle primitive benchmark(STOP GATE, H2c)— **FAIL**(ADR-0029)

**目的:** `SHIFT`/`SELECT`/`COUNT`/`BIND` について、正しい family **かつ** 正しい引数のときに高精度、それ以外(誤引数/誤family/primitiveなし)で materially 低い精度になることを示す。

**実装:** A1-R003 と全く同じレシピを引数条件付き primitive に適用: task-blind に事前学習・凍結した Stable Core の上に `ConditionedPrimitive` を4つ登録し、oracle-forced routing で学習。`apc.core.execution._route_and_apply_oracle_calls` を拡張し、`ConditionedPrimitive` を `forward_from_calls` 経由で正しく駆動できるようにした(このタスクで初めて配線)。4-arm 評価: Correct(正family+正引数)/ Wrong argument(正family+「+1 mod 有効域」で決定的にずらした誤引数)/ Wrong family(固定derangementで強制した誤family)/ None。

**結果:** config `configs/phase_a1_parameterized_primitive_gate.yaml`、5 seeds(0-4)、RTX 5060 Ti、commit `4edc4c0`。学習量は A1-R003 と完全に同一(`core_train.steps=30000`, `primitive_train.steps=10000`, `primitive_rank=8`)。

| Arm | 平均 | 判定基準 | 判定 |
|---|---|---|---|
| Correct | **0.308**(stdev 0.017、min 0.290、max 0.328) | ≥0.90 | ✗ |
| Wrong argument | 0.255 | ≤0.30 | ○(が実質無意味) |
| Wrong family | 0.028 | ≤0.30 | ○ |
| None | 0.143 | ≤0.30 | ○ |
| causal gap | **0.053** | ≥0.50 | ✗ |
| family数一定(bank_size) | 4(全seed) | 一定であること | ○ |

演算別 Correct: COUNT 0.595 / BIND 0.420 / SHIFT 0.161 / SELECT 0.047。

**2つの独立した所見(ADR-0029 の Reason):**

1. **family選択は強く因果的だが、family内の引数選択はほぼ無関係。** Wrong family(0.028)が Correct(0.308)から大きく落ちる点は A1-R003 と同じパターンを再現しており、`h_content` が family 選択に対して依然として因果的な情報を運んでいることを示す。一方 Wrong argument(0.255)は Correct(0.308)からわずか約0.05しか離れておらず、「正しい family が選ばれてさえいれば、引数を間違えてもほぼ同じ出力」になっている。つまり `ConditionedPrimitive` の `c_proj`/`argument_encoder` 経路(A1-R004)は、この学習量では出力トークンを実際に左右するところまで学習されていない。
2. **演算別 Correct は出力トークン数と強く相関している。** COUNT/BIND(出力1トークン)が0.42〜0.60なのに対し、SHIFT(最大10トークンの全長回転)/SELECT(最大5トークンの可変長順序付き部分列)は0.05〜0.16。完全一致判定は全出力位置が同時に正しい必要があるため、1トークン当たりの精度が同程度でも複数トークン出力ほど exact match が複合的に不利になる。所見1と矛盾せず、SHIFT/SELECT は「引数選択の弱さ」と「複数トークンの複合ペナルティ」を二重に受けている。

**意図的に据え置いた点:** 学習量を A1-R003 と完全に同一にしたのは意図的な選択(`AGENTS.md` 「最小実装で仮説を反証できるものを優先する」)。「今回失敗した」原因を「引数条件付け・より難しい演算プール」だけに絞り込むため、同時にモデルサイズや学習量を増やすことは避けた。

---

## 3. 総合考察(次の仮説検討のために)

- **R001〜R003 の一貫した物語:** task-blind な content path(R001, 構造的に保証)→ decoder がそれを迂回できない(R002, 監査+実測)→ frozen core 上で primitive が実際に演算を担う(R003, Correct 0.9999 / Wrong 0.0)、という3段階はすべて計画通りに成立している。ここまでは **family 単位** の「どの演算を適用するか」という因果性の話。
- **R004〜R005 で新たに生じた論点:** R004 は「引数を条件として消費 **できる**」という能力(unit test レベル)を示したが、R005 はその能力が「学習によって出力に実際に反映される」ところまでは今回の設定では届かなかったことを明らかにした。言い換えると、**family 選択の因果性と、family 内での引数選択の因果性は別物であり、後者は前者と同じ学習レシピでは自動的には得られない。**
- **ADR-0029 に記録した、まだ実行していない3つの仮説:**
  - (a) 引数条件付け経路(`c_proj`/`argument_encoder`)は `a_proj`/`b_proj` より新しく学習が浅い可能性があるため、`primitive_train.steps` を増やす。
  - (b) `SHIFT`/`SELECT` は `max_sequence_length` 通りの回転/部分列を区別する必要があり、`COUNT`/`BIND` の `vocab_size` 通りの値と比べて、同じ `rank=8`/`arg_dim` では容量不足の可能性がある。
  - (c) SHIFT/SELECT については exact match に加えてトークン単位の精度も測定し、「引数が学習されていない」のか「複合ペナルティで潰れているだけ」なのかを切り分ける。
- **切り分けの優先度についての考察:** 所見1(Wrong argument ≈ Correct)は COUNT/BIND(単一トークン出力)でも一定程度観測されるはずなので、演算別に Correct と Wrong argument の差を個別に見ると、「引数が効いていない」問題と「複数トークンの複合ペナルティ」問題のどちらが支配的かを演算単位で切り分けられる可能性がある(このレポート作成時点では per-operation の Wrong argument 内訳は集計していない — 必要であれば `runs/phase_a1_parameterized_primitive_gate/seed_*/report.json` の `per_operation_wrong_argument_exact_match` から追加集計可能)。

---

## 4. 現在のブロック状況

`AGENTS.md` の STOP GATE discipline により:

- **A1-R006(unified oracle primitive benchmark)以降はブロック中。** A1-R005 が PASS するまで着手しない。
- ハイパーパラメータを増やしての再挑戦は行っていない(上記3仮説はいずれも未実行)。
- A1-R001〜A1-R004 の成果自体はこの結果によって無効化されない(R004 の unit-level 受け入れ基準はそのまま有効 — 「引数を渡すと出力が変わる」という能力自体は確認済みで、今回失われたのは「学習でその能力が使われるようになるか」の部分のみ)。

**関連ファイル:**
- ADR: `docs/DECISIONS.md` ADR-0025〜ADR-0029
- 実行アーティファクト: `runs/phase_a1_task_blind_content_gate/`, `runs/phase_a1_decoder_leakage_gate/`, `runs/phase_a1_parameter_free_primitive_gate/`, `runs/phase_a1_parameterized_primitive_gate/`
- 実装: `apc.evaluation.task_blind_content_gate`, `apc.evaluation.decoder_leakage_gate`, `apc.evaluation.parameter_free_primitive_gate`, `apc.primitives.conditioning`, `apc.evaluation.parameterized_primitive_gate`
