# A1-R005E-S002 結果サマリー — Balanced Mixed-Operation Training Gate

**位置づけ:** `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md` の A1-R005E-S002。タスク自身の Acceptance は「すべての測定項目を揃えること。ブランチ判断はまだしない (Complete all measurements; no branch claim yet)」であり、**本レポートも測定結果の整理に留め、Branch B を採用するかどうかの正式な判断はしない**(それは A1-R005E-S003/S005 の役割)。ADR は追加していない — S001 と同じ理由で、本タスクの Acceptance は全項目を満たしており、アーキテクチャ上の前提が誤っていた/曖昧だったことを示す結果でもないため(`AGENTS.md` の ADR トリガーは「前提が誤り/曖昧/実行不可能と判明した場合」であり、ルーティンな完了はこれに当たらない)。次の未使用 ADR 番号は 0044 のまま。

**結論を先に:** A1-R005E-S001 で組んだ「1つの共有 task-blind エンコーダ + 演算ごとの compact operator」を、SHIFT/SELECT/COUNT/BIND を interleave した単一の学習ループ(1つの optimizer がエンコーダと4つの operator 全部を同時に持つ)で **5 seed × 152000 step** 実際に学習させた。**サンプリング比は全 seed で寸分違わず 25/25/25/25(各演算ちょうど 38000 step)を実現**し、**task-blind 不変性は全演算・全 seed で厳密に保持**(`max_abs_diff = 0.0`)。総合判定(演算ごとの絶対閾値ベース)は **SELECT のみ `passed=true`**、SHIFT/COUNT/BIND は `passed=false` だが、COUNT/BIND の不合格理由は A1-R005E-006A/ADR-0043 で既に指摘済みの「None アーム上限 0.30 をわずかに超過」という既知のパターンのみ(Wrong・causal gap はいずれもクリア)。E-006A(演算ごとに専用エンコーダを学習)の保存済み結果と素朴に突き合わせると、**4演算すべてで Correct exact match・causal gap ともに E-006A の 99.6%〜103% の範囲に収まっており**、特に事前に最も懸念されていた SHIFT でも平均性能はほぼ同一(0.5193 対 0.5194)のうえ **seed 間ばらつきは明確に縮小**(stdev 0.036 対 0.161)している。これは非公式な突き合わせであり正式な `R_shared` 計算(A1-R005E-S003 の役割)ではないが、Branch B にとって好意的な材料に見える。

---

## 1. 背景: 何を検証したか

A1-R005E-006A(`docs/CODEX_TASKS_A1_R005E_E006_PLUS.md`)は、task-blind content encoder を compact operator と **jointly** 学習させると SHIFT/SELECT/COUNT/BIND 全演算で劇的に改善する(`R_access` 0.83〜2.24)ことを示した(ADR-0043)。しかし E-006A は **演算ごとに専用のエンコーダを1つずつ**学習していた(`(seed, operation)` ペアごとに fresh `JointStableCore` を1つ)。

`docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md` が要求する APC の本質的性質はより強く、「1つの共有表現 → 複数の疎な再利用可能 operator」である。A1-R005E-S001 はこの共有アーキテクチャの配線(1つのエンコーダ + 演算ごとの operator、oracle 選択)を組んだが学習は一切していない(Acceptance: "No milestone benchmark yet")。**A1-R005E-S002 はその配線を実際に学習させ、E-006A の利得が「共有」に耐えるかを検証するための測定を揃えるタスク。**

`docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md` セクション3の要求:

- balanced mixed-operation online training(既定 SHIFT/SELECT/COUNT/BIND 各25%)
- 反実仮想(counterfactual)グループ構成は維持

`docs/AGENTS_A1_R005E_SHARED_ENCODER_ADDENDUM.md` の Hard invariant: 「共有エンコーダは content のみを見る。operation ID・タスクトークン・引数トークン・oracle メタデータのいずれも入力してはならない」。

---

## 2. 実装したもの(学習設計)

`src/apc/evaluation/shared_encoder_mixed_operation_training_gate.py`(新規モジュール)。A1-R005E-S001 の `build_shared_encoder_architecture`/`run_shared_operator`(`apc.evaluation.shared_encoder_architecture_gate`)をそのまま再利用し、アーキテクチャ自体は再設計しない。

- **1つの `torch.optim.AdamW`** が共有エンコーダの全パラメータ + 4つの operator 全部のパラメータをまとめて保持。ある step で選択されなかった operator は forward/backward を一切通らないため `grad is None` のままで、PyTorch の `AdamW.step()` は `grad is None` のパラメータを自然にスキップする — 演算ごとに別々の optimizer を用意しなくても「選択された operator だけが更新される」性質がそのまま成り立つ(専用テストで確認済み)。
- **サンプリングは決定的な重み付きラウンドロビン**(`_build_operation_schedule`): 既定重み(各演算1)なら「4演算を1回ずつ含むサイクル」を毎回シャッフルして連結する。IID サンプリングと違い、有限の step 数でも実現比率が期待値ちょうどになる(既定重みかつ `steps` が4の倍数なら**厳密に** 25/25/25/25)。
- **`joint_train.steps` の既定値は 152000 = 4 × E-006A 自身の演算あたり予算(38000)**。ある step で選択された演算の operator だけがその step の勾配を受け取るため、4演算に均等に25%ずつ割り振ると、各演算の operator は結果として E-006A と**全く同じ回数**(38000回)だけ自分自身の更新を受ける — A1-R005E-S003 の "shared-vs-specialized" 比較を operator 側の学習予算という軸でフェアにするための設計判断。共有エンコーダ自身は毎 step 更新されるため、E-006A の個別エンコーダ(各38000更新)よりずっと多い152000更新を受ける(意図した非対称性)。5 seed 合計の総 step 数(5×152000=760000)は E-006A の総 step 数(5 seed×4演算×38000=760000)と完全に一致するよう設計しており、wall-clock も概ね一致する見込みだった。
- 反実仮想グループ生成(`generate_compact_operator_counterfactual_groups`)・Correct 引数の読み出し・ラベル構成は A1-R005E-006A のものを変更せず再利用。`step` ごとにどの演算が選ばれるかだけが変わり、各演算自身の内容ストリームは(そのオペレーションの own `step` 値で見れば)専用ループで学習した場合と同じ決定的な分布から引かれる。
- 各チェックポイント(`eval_every=8000`、進捗評価は小さい held-out batch)で **4演算全部**の Correct exact match を測定・記録(選択中の演算だけでなく)。共有エンコーダを介した干渉の可視化を狙った設計 — S003 の分析材料になる。
- 学習完了後、E-006A と同じ3アーム(Correct / effectful Wrong argument / None)の反実仮想評価を大きな未見バッチ(`num_unseen_eval_groups=1400`)で実施し、causal gap を算出。task-blind 不変性の回帰チェックも E-006A と同じ形で実施。

---

## 3. 結果

5 seeds (`0`–`4`)、RTX 5060 Ti(Windows, `torch==2.13.0+cu130`)、git commit `09ea9a8`(実行がコミットに先行するのはこのリポジトリの標準的な precedent)。総 wall-clock 約 **11752秒(約3時間15分52秒)** — E-006A 自身の約3時間29分よりやや短く、総 step 数を一致させる設計判断が意図通り機能したことを裏付ける。アーティファクト: `runs/phase_a1_shared_encoder_mixed_operation_training_gate/`(gitignore対象)。

### 3.1 全体サマリー(5 seed 平均)

| Operation | Correct exact(mean, stdev, min–max) | Correct token acc. | Wrong exact | None exact | Causal gap(exact / token) | `passed` |
|---|---|---|---|---|---|---|
| SHIFT | 0.5193(0.0363, 0.4748–0.5657) | 0.8935 | 0.0000 | 0.0001 | **0.5192** / 0.6786 | false |
| SELECT | **0.9998**(0.0004, 0.9990–1.0) | 0.9999 | 0.0000 | 0.0393 | **0.9605** / 0.5840 | **true** |
| COUNT | 0.9955(0.0057, 0.9855–0.9993) | 0.9955 | 0.0009 | 0.3247 | 0.6708 / 0.6708 | false |
| BIND | 0.9998(0.0002, 0.9995–1.0) | 0.9998 | 0.0001 | 0.3012 | 0.6986 / 0.6986 | false |

**基準(E-004/E-005/E-006A から踏襲、変更なし):** Correct exact `>= 0.90`、SHIFT/SELECT token acc. `>= 0.98`、effectful Wrong argument `<= 0.30`、None `<= 0.30`、causal gap `>= 0.50`。

**基準ごとの合否:**

| Operation | Correct exact | Token acc. | Wrong | None | Causal gap | 総合 |
|---|---|---|---|---|---|---|
| SHIFT | FAIL(0.519) | FAIL(0.894) | PASS | PASS | PASS(0.519、基準ぎりぎり) | FAIL |
| SELECT | **PASS**(1.000) | **PASS**(1.000) | PASS | PASS | **PASS**(0.961) | **PASS** |
| COUNT | PASS(0.995) | (対象外) | PASS | **FAIL**(0.325、上限0.30をわずかに超過) | PASS(0.671) | FAIL |
| BIND | PASS(1.000) | (対象外) | PASS | **FAIL**(0.301、上限0.30をごくわずかに超過) | PASS(0.699) | FAIL |

`argument_effect_rate = 1.000`(全演算・全 seed)。`meets_seed_policy = true`。`task_blind_invariant_passed = true`(全演算・全 seed、`max_abs_diff = 0.0`)。

### 3.2 サンプリング・更新回数の会計(タスク自身の "Required metrics")

全 5 seed で完全に同一:

| Operation | step count(= operator 自身の更新回数) | 実現比率 | examples seen(概算) |
|---|---|---|---|
| SHIFT | 38000 | 0.25 | ~4,901,969 |
| SELECT | 38000 | 0.25 | ~4,901,933 |
| COUNT | 38000 | 0.25 | ~4,819,150 |
| BIND | 38000 | 0.25 | ~4,491,291 |

`encoder_update_count = 152000`(全 step で共有エンコーダが更新される、演算に依らず)。`operation_step_counts` の合計は全 seed で `152000` と一致(会計の整合性を確認)。

### 3.3 E-006A(演算専用エンコーダ)との素朴な突き合わせ(非公式・S003 の代替ではない)

以下は E-006A の保存済み `summary.json`(`runs/phase_a1_joint_representation_compact_operator_probe/summary.json`)と単純に比を取っただけの参考値であり、**A1-R005E-S003 の正式な `R_shared` 計算ではない**(タスクの切り分け上、正式な比較・解釈は S003/S005 の役割)。

| Operation | E-006A Correct | S002 Correct | 比(S002/E-006A) | E-006A gap | S002 gap | 比(S002/E-006A) |
|---|---|---|---|---|---|---|
| SHIFT | 0.5194 | 0.5193 | ~99.98% | 0.5192 | 0.5192 | ~100.00% |
| SELECT | 1.0000 | 0.9998 | ~99.98% | 0.9641 | 0.9605 | ~99.62% |
| COUNT | 0.9974 | 0.9955 | ~99.81% | 0.6699 | 0.6708 | ~100.13% |
| BIND | 0.9996 | 0.9998 | ~100.03% | 0.6780 | 0.6986 | ~103.05% |

4演算すべてが `docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md` セクション6の "strong" 閾値(`R_shared >= 0.90`)を大きく上回る水準に見える(BIND の causal gap に至っては E-006A を上回っている)。事前にシェアリングが最も失敗しやすいと想定されていた SHIFT でも、平均値はほぼ変化していない。

### 3.4 演算ごとの深掘り

**SHIFT — 平均は E-006A とほぼ同一だが、seed間ばらつきは明確に縮小。** seed別 Correct exact match: `0.4945, 0.5657, 0.5419, 0.5198, 0.4748`(stdev `0.036`、範囲 `0.091`)。E-006A の同じ5 seed(`0.7388, 0.5995, 0.3140, 0.5038, 0.4410`、stdev `0.161`、範囲 `0.425`)よりも大幅にタイト。E-006A では seed ごとの最終学習損失(`0.098`〜`0.381`)が Correct exact match とおおむね相関しており(損失が低い seed ほど高精度)、初期化依存で収束先の質にばらつきが出ていたと考えられるが、S002 ではこのばらつきが縮小している——共有エンコーダが152000回更新される(E-006Aの単独エンコーダの4倍)ことが SHIFT にとってより安定した表現初期条件を与えている可能性があるが、これは推測であり検証していない。また学習末尾(step 136000→152000)でも SHIFT の累積平均損失は `0.366→0.358→0.351` と緩やかに下降し続けており、**152000 step 時点でも完全には収束していない可能性がある**(§5 参照)。

**SELECT — 唯一 `passed=true`。** E-006A でも唯一 `passed=true` だった演算であり、共有設定でもその地位を維持。Correct(0.9998)・causal gap(0.9605)ともに極めて高く、token accuracy(0.9999)も基準を大きく上回る。

**COUNT / BIND — 不合格理由は None アーム上限のみ(E-006A/ADR-0043 と同一パターン)。** どちらも Correct(0.995/0.9998)・Wrong(0.001/0.0001)・causal gap(0.671/0.699)は基準を十分にクリアしているが、None アーム(引数なし)の exact match がそれぞれ 0.325/0.301 で上限 0.30 をわずかに超過している。ADR-0043 はこれを「引数漏洩ではなく、ターゲット分布自体の base-rate 的な偏りによる可能性が高い」と推測していた(A1-R005E-006A でも全く同じパターン: COUNT 0.328、BIND 0.322)。この base-rate 仮説を正式に検証するのは A1-R005E-S004(COUNT/BIND argument-blind baseline audit)の役割であり、本タスクでは検証していない。

---

## 4. 今後の分岐判断のための整理(本タスク自身は判断しない)

`docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md` セクション12の Outcome A〜D のうち、§3.3 の非公式な突き合わせが正式な `R_shared` 計算(A1-R005E-S003)で追認されれば、**Outcome A(shared encoder broadly retains E-006A → Branch B formally supported)** に該当する可能性が高いように見える材料を整理する。ただしこれはあくまで参考情報であり、正式な判断は A1-R005E-S003(retention analysis)→ A1-R005E-S005(branch decision, `docs/results/A1_R005E_SHARED_ENCODER_GATE_RESULT.md` を作成)の役割。

### Outcome A を支持する材料
- 4演算すべてで Correct exact match・causal gap が E-006A の概ね99.6%〜103%の範囲(§3.3)。
- SELECT/COUNT/BIND は `docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md` セクション7の Branch-B 基準(">=90% retain")を明確に満たしそうに見える。
- SHIFT も同セクション7の "SHIFT may remain below final accuracy target if it stays materially above E-005 and its residual is consistent with operator mismatch" という緩和条件に該当しそうに見える(E-005 [frozen+compact] は SHIFT でほぼ0 [ADR-0043 の C00] だったのに対し、S002 は 0.519 — 大幅に上回っている)。
- task-blind 不変性は学習後も厳密に保持(共有アーキテクチャの hard invariant が実際の学習下でも壊れていない)。
- 干渉(multi-task interference)による性能崩壊は観測されなかった——`docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md` セクション5が懸念していたリスクは、少なくとも今回の balanced 25/25/25/25 設定では顕在化しなかった。

### まだ弱い、または未検証の材料
- 上記はすべて非公式な比較であり、A1-R005E-S003 自身の正式な `R_shared` 計算(E-004/E-005 との比較も含む)はまだ実行していない。
- COUNT/BIND の None ceiling 超過の base-rate 仮説は S004 で未検証。
- SHIFT は絶対閾値(Correct `>=0.90`)には遠く、`docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md` セクション11の "SHIFT follow-up trigger"(A1-R005E-S006 の条件)に該当するかどうかの正式判定はS005の役割。
- §3.4 で述べた通り SHIFT は 152000 step 時点でも完全収束していない可能性があり、今回の 0.519 という値が「共有アーキテクチャの上限」を過小評価している可能性がある(逆に、より長く学習すればさらに E-006A に近づく、あるいは追い越す可能性も否定できない)。

---

## 5. 未解決の論点・キャビエト

1. **正式な `R_shared`/分岐判断はまだ実行していない。** §3.3/§4 の数字は本タスクの担当者(このセッション)が参考として突き合わせただけであり、A1-R005E-S003/S005 の正式な出力ではない。
2. **SHIFT の収束状況は未確定。** 学習末尾でも累積平均損失が緩やかに下降し続けており(§3.4)、152000 step(operator 自身は38000回更新)が SHIFT にとって十分な予算だったかは検証していない。
3. **COUNT/BIND の None ceiling 超過の base-rate 仮説は本タスクでは検証していない。** A1-R005E-S004 の役割。
4. **進捗評価(`operation_progress_correct_exact_match`)は各チェックポイントで4演算全部を測定しているが、干渉の定量分析(演算間の勾配コサイン類似度など)は行っていない。** `docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md` セクション5が示す "optional pairwise gradient cosine" は本タスクのスコープ外(タスク文の Required metrics に含まれていない)。
5. **本タスクは `Router`/`PrimitiveBank`/`Primitive`/`PlasticWorkspace` を一切通していない。** oracle 選択のみで、学習ルーティングについては何も語っていない。

---

## 6. 関連ファイル

- ADR: なし(ルーティンな完了、`AGENTS.md` の ADR トリガーに該当しないため未作成。次の未使用 ADR 番号は 0044)
- タスク仕様: `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`(A1-R005E-S002)
- 実験計画: `docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md`
- 設計ドキュメント: `docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md`
- Addendum: `docs/AGENTS_A1_R005E_SHARED_ENCODER_ADDENDUM.md`
- 前タスク: A1-R005E-S001(アーキテクチャ配線)— `src/apc/evaluation/shared_encoder_architecture_gate.py`
- 比較対象: A1-R005E-006A(演算専用エンコーダ)— `runs/phase_a1_joint_representation_compact_operator_probe/summary.json`、`docs/DECISIONS_A1_R005E_DIAGNOSTIC.md` ADR-0043
- 実行アーティファクト: `runs/phase_a1_shared_encoder_mixed_operation_training_gate/`(`config.yaml`, `report.json`, `summary.json`, `system.json`, `seed_<n>/{report.json, mixed_training_metrics.jsonl}`)
- 実装: `src/apc/evaluation/shared_encoder_mixed_operation_training_gate.py`, `scripts/shared_encoder_mixed_operation_training_gate.py`, `configs/phase_a1_shared_encoder_mixed_operation_training_gate.yaml`, `tests/test_shared_encoder_mixed_operation_training_gate.py`(31 tests)

**次タスク(本レポートでは未着手):** A1-R005E-S003(shared-vs-specialized retention analysis、正式な `R_shared` 計算)。`AGENTS.md`: 「1つのタスクの完了は次のタスクの開始を自動的には許可しない」。
