# B-C005R3 post-D2 repair シリーズ 状況レポート (R3-001〜R3-010)

**作成日:** 2026-09-07
**目的:** R3-001〜R3-010の実装内容・実行結果・経緯を一覧化し、今後の方針判断(R3-011へ進むか、R3-010が発見したインフラ問題への対応を先に行うか、等)の材料とする。
**関連ドキュメント:** `docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md`(タスク仕様)、`docs/exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md`(全体計画)、`docs/EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md`(数値Gate定義)、`docs/DECISIONS_PHASE_B.md`(ADR-0082〜0091 本文)、`docs/DECISIONS.md`(ADR索引)。

---

## 1. このシリーズの位置づけ

第二診断(D2)完了後、ADR-0081がOption C(semantic-relation holdout再設計)を次の研究上の優先課題に指定した。本シリーズ `B-C005R3-001`〜`B-C005R3-012` は、それを受けて (1) 再現性を確立し (2) 証拠に基づく局所修正を行い (3) 最終的に新しい封印済みGate v2 (`B2_PROTOCOL_V2`) で評価する、という3段階の構成になっている。

**統治ルール(STOP rule):** `B-C005R3-0NN` は必ず1タスクずつ、ユーザーの明示的な指示によってのみ実行する。1タスクの完了は次タスクの自動実行を許可しない。本レポート作成時点で **R3-010まで完了**、R3-011以降は未着手。

**タスク系列全体表:**

| ID | 主目的 | 変更できる範囲 | 境界(Gate) |
|---|---|---|---|
| R3-001 | benchmarkの再現性修正 | seed導出・生成version・manifest | G0 |
| R3-002 | relation単位の分割 | 評価プロトコル・露出台帳 | G1 |
| R3-003 | adequacy指標・統計契約 | offline指標・schema | G2 |
| R3-004 | 同一入力上の凍結対照 | 評価runnerのみ | 実験整合性 |
| R3-005 | unsafe reuseを防ぐverifier | verifier・薄いsafety adapter | G3 |
| R3-006 | COUNT↔BIND key/scoring | 対象key/scoringのみ | 局所Gate |
| R3-007 | SELECT argument encoding | SELECTのencodingと必要最小限のhead | 局所Gate |
| R3-008 | BIND argument scoring | BIND head/compatibilityのみ | 局所Gate |
| R3-009 | SHIFT functional generalization | 別candidateの学習・安全な置換 | 局所Gate |
| **R3-010** | **修正の統合・旧機能回帰** | **integrationのみ** | **G4** |
| R3-011 | 新Gateの事前封印 | manifest・preflight(未着手) | 封印 |
| R3-012 | 新しいB2 Gate v2 | 評価・最終ADR(未着手) | G5 / STOP |

---

## 2. タスクごとの実装・結果サマリー

### R3-001 — benchmarkの再現性修正
**COMPLETE, commit `bb7ec00`, ADR-0082。G0: `INFRASTRUCTURE_OR_PROTOCOL_PASS`。**

`hash(operation) % 10000` によるプロセス間非決定性(ADR-0080の指摘)を、新モジュール `apc.utils.seed_derivation`(SHA256正規payload、versioned、per-sample-index)で修正。ディスパッチャースクリプト `scripts/run_phase_b_b2_post_d2_repair.py` を新設(`--task` 明示指定、`--all` なし)。

### R3-002 — relation単位の分割 (Option C)
**COMPLETE, commit `8009cd1`, ADR-0083。G1: `PROTOCOL_INSUFFICIENT_RELATIONS`。**

新モジュール `relation_split_protocol.py`。`hard_negative_routing_benchmark._RELATED_OPERATION` から機械的にrelationカタログを導出(4方向→3組)。**2つの核心的発見:**
1. BIND-key結合によりL3グループが実質2独立成分に減る(`SELECT-BIND`と`COUNT-BIND`が物理BIND primitiveのrouter keyを共有するため)。
2. 既存のR2修理レシピ(`train_repaired_router_and_scorer`)は構造的に `MINING_HOLDOUT_ONLY` が上限であることを実測で確認(SHIFTを訓練例から除外してもSHIFTのrouter keyが変化することを実証)。

development/validation(15-19)/sealed_v2(30-34)のseed区分を確立。`targeted_repair_regression` はブロックされないが、L3 unseen-relation transferの主張はR3-012まで不可。

### R3-003 — adequacy指標・統計契約 (Option E, spec-only)
**COMPLETE, commit, ADR-0084。G2: `INFRASTRUCTURE_OR_PROTOCOL_PASS`。**

新モジュール `functional_metrics_v2.py`。`ReferenceAdequacyState`/`CandidateVerdict`/`SearchStatus`/`ControllerAction`/`ExecutionStatus` 等のenumと `VerifiedDecisionEnvelope`。有限look厳密境界の統計契約(`tau=0.95, max_candidates=5, looks=[32,64,128,256,512], alpha=0.01`)。**Runtimeには未接続**(offline仕様のみ)。R3-004/005が使う契約をここで凍結。

### R3-004 — 同一入力上の凍結対照
**COMPLETE, commit, ADR-0085。Gate対象外、結果 `COMPARISON_ESTABLISHED`。**

新モジュール `paired_baseline_repair.py`。既存R1/R2機構(`train_repaired_router_and_scorer`, `evaluate_repair_cell`)を再利用し、development seeds(10-14)でR0/R1/R2比較。**2つの重要発見:**
1. **D2で診断された4つの失敗のうち全て development v2 では `NOT_REPRODUCED_ON_V2`**(COUNT↔BIND, SELECT L4, BIND L4)。→ R3-006/007/008に `NEEDS_SCOPE_REVIEW` フラグ。
2. **SHIFTのreference adequacyはdevelopment seedsで測定不能** — development seeds 10-14にはSHIFT用の事前学習済みContent Encoderが存在せず(sealed seeds 0-4のみ)、raw closed-loop EMは無意味(実測0%)。`UNVERIFIABLE_ON_DEVELOPMENT_PARTITION_UNTRAINED_CORE` として正直に報告。→ **これがR3-009の出発点となるブロッカー。**

### R3-005 — Safe Bounded Verification (Option E runtime wiring)
**COMPLETE, ADR-0086。G3: `G3_PASS`。**

新runtime verifier `apc.meta.adequacy_verifier.BoundedExactLookVerifier`(R3-003の契約に忠実、`meta`→`evaluation`のimport逆転は設計文書で許可済み)。旧 `SequentialAdequacyVerifier` はバイト単位で無改変のまま維持。

**3つの結果:** (1) CPU Bernoulli契約sweepでADR-0080のバグ(`p=0.949`で旧verifierが75.1%誤受理)を再現し、新verifierが0.0%で正しく拒否することを実証。(2) D2-005の履歴データreplayでも同じ乖離を確認。(3) **副次発見**: development seedsでは全4操作(SHIFT含む)がraw execution EMで測ると`REF_INADEQUATE`になる — untrained Coreの構造的帰結であり、routing/argument-scorer指標(R3-004が使った指標)には影響しない。

### R3-006 — COUNT↔BIND Key/Scoring Repair
**COMPLETE, ADR-0087。局所Gate: `VALIDATION_PASS`(当時)。**

新モジュール `count_bind_key_scoring_repair.py`。COUNT/BINDのrouter keyのみ`requires_grad=True`にする狭いscoped修正(`scoped_pairwise_margin`選択)。**中心的発見: development seedsではR0(未修正の凍結親)が既にGate閾値を満たしていた**(COUNT→BIND top-1 `0.965625`, BIND→COUNT top-1 `0.984375`)— つまり「修理対象が既に存在しない」ことを示す独自の`VALIDATION_PASS`(元のsealed partition の`KEY_SCORING_BOTTLENECK`は未解決のまま)。

> ⚠️ **R3-010がこの数字の再現性を破壊する重大な副作用を発見した(§4参照)。**

### R3-007 — SELECT Argument-Encoding Repair
**COMPLETE, ADR-0088。局所Gate: `VALIDATION_PASS`。**

`src/apc/primitives/argument_scoring.py` に実質10行の修正。**本物の欠陥を発見して修正**: `ArgumentScorer`はSELECTを独立multi-hot BCEWithLogitsLossで訓練していたが、推論時は全操作共通のsoftmaxでスコアリングしていた(ADR-0081のD2-004所見と一致)。修正: SELECTのみ`sigmoid`読み出しに変更(再学習不要、既存の重みをただ正しく読むだけ)。手作りlogitsでの特性評価により、cardinality>1でsoftmaxスコアが劇的に歪む(-0.00009等)ことを定量的に実証。

### R3-008 — BIND Argument-Scorer Generalization Repair
**COMPLETE, ADR-0089。局所Gate: `VALIDATION_PASS`。**

新モジュール `bind_argument_scorer_repair.py`。**中心的発見**: development seedsの集約argument_accuracy 0.9625 は、sealed partitionと全く同じ「rare値=0点、seen値=満点」構造を低頻度で隠していた(308 seen@1.0 + 12 rare@0.0)。**修正**: `stratified_value_coverage`(128例、観測値ごとに均等配分)でBINDのheadのみ再学習 → argument_accuracy 1.0。

### R3-009 — SHIFT Functional-Generalization Repair & Versioned Replacement
**COMPLETE, ADR-0090。局所Gate: `FAIL`(development 5 seed中3 seed committed)。**

R3-004が発見したブロッカー(development seedsにSHIFT用の事前学習済みCoreが無い)への対応として、**ユーザーに事前確認の上**development seeds(10-14)向けに本物の非sealed shared encoderを新規事前学習(既存Phase A1レシピを流用、sealedデータ・sealedコードパス一切不使用)。`PrimitiveBank.replace_primitive`(新規汎用API)によるper-seedの安全なバージョン置換トランザクションを実装。

**結果**: iid_baseline variantを採用。seed 10/11/14は`EM=1.0/REF_ADEQUATE`でCOMMIT、seed 12/13は`EM=0.918/0.787`で `REF_INADEQUATE` のためROLLBACK。平均EM `0.941 < 0.99` によりGate **FAIL**(正直に報告、閾値の言い訳をしない)。error-weighted stratification は逆に悪化させた(セード12/13の弱ストラタムは元々サンプル不足ではなかったため)。

**副作用の開示(ADR-0090)**: development seeds 10-14は今後「本物の非ランダムCore」を共有キャッシュ経由で持つことになり、R3-004〜008の将来の再実行結果に影響する、と当時開示していた。→ **R3-010がこの開示の"深刻度"を大幅に過小評価していたことを発見した(次節)。**

---

## 3. R3-010 — 統合ラダーと、シリーズ最大の発見

**COMPLETE, ADR-0091。G4 gate: `DEVELOPMENT_INTEGRATION_FAIL`。ただしこのFAILは「5つの修正を組み合わせたこと自体の不具合」ではなく、2つの新発見インフラ問題に起因することを実測で特定・切り分け済み。**

### 3.1 設計

R3-005〜R3-009の5つの修正を、同一の親checkpoint/入力の上に1つずつ積み上げるラダーを構築:

```
C0: v2入力 + 旧runtime(frozen parent router + legacy-formula scorer)
C1: C0 + new verifier (R3-005)
C2: C1 + COUNT<->BIND key/scoring repair (R3-006)
C3: C2 + SELECT argument-encoding repair (R3-007)
C4: C3 + BIND argument-scorer repair (R3-008)
C5: C4 + SHIFT versioned primitive replacement (R3-009)
```

新モジュール `paired_integration_regression.py`(実測部分は既存の各タスクの関数を無改変のまま再利用・再実行。新しいlossや閾値のチューニングは一切行っていない)。「Legacy K/C/N/R」(Phase A2の自律的plasticity/consolidationストリーム)は、既存コード(`sequential_closed_loop_benchmark.py`)が外部から修正済みbank/routerを注入する経路を持たない(コード読解で確認)ため、integration-onlyのスコープ外と判断し、代わりにnominal-lite matrix(L0-L4)+ 未修正6操作の直接実行チェックで代替。

### 3.2 実装中に修正した2つの実コードバグ

1. `evaluate_repair_cell`への`seed`引数に誤ってop-idを渡していた(実測上は無影響と判明)。
2. `argument_scorer`/`arg_lambda`を全levelで無条件使用していた → R3-004/006/008自身の慣例(L4のみ、かつjointly訓練されたrouter+scorerペアでのみ使用)に合わせてL4限定に修正。

### 3.3 発見A(最重要・深刻): 共有primitiveキャッシュと新しいCoreの不整合

**未修正6操作(COPY/REVERSE/SORT/NEGATE/SWAP_PAIRS/INVERT_HALF)のclosed_loop_exact_matchが、C0〜C5の全条件・全seedで一律 `0.0`。** これはR3-010の統合による回帰ではない(C0時点で既に壊れている)。

**根本原因(2通りの独立検証で確認、推測ではない):**
- `_build_frozen_base_system` は development seeds 10-14 に対し、R3-009が新規事前学習した本物のshared encoder(`shared_encoder.pt`, 2026-09-07作成)を読み込むようになった。
- 共有キャッシュ `primitive_bank_16.pt`(2026-09-06作成、**R3-009より前**)が供給する16 primitiveのうち10個(SELECT/COUNT/BIND/SHIFT/COPY/REVERSE/SORT/NEGATE/SWAP_PAIRS/INVERT_HALF)は、`get_or_build_16_primitive_bank`(`incremental_router_benchmark.py`)の「キャッシュが無い場合の新規構築パス」を読んでも**一切学習されない**ことを実機検証で確認(スクラッチキャッシュディレクトリを指定して新規構築させたところ、COPYですら`EM=0.0`/chanceレベルのtoken精度)。学習されるのは残り6つのPhase A2インクリメンタル操作のみ。
- つまり、現在のキャッシュの「良い重み」は、このリポジトリ内には存在しない、より早期の(Phase A1/A2の)学習パイプラインの遺産であり、当時のCore(R3-009以前のランダムだが決定論的なCore)向けに較正されたもの。R3-009が本物のCoreに差し替えたことで、この10 primitiveの較正が静かに無効化された。

**router系の指標(primitive_call_top1, argument_accuracy)はこの発見の影響を受けない**(routerは毎回新しいCoreに対して再較正されるため)。

**ユーザーへの2段階確認:** まず「キャッシュを削除して自動再構築させれば直るか」と問うたところ「共有キャッシュを新Coreに合わせて再構築する」を選択。しかし追加調査で「自動再構築パスはそもそも該当10 primitiveを学習しない」ことが判明し、真のコストは「`run_unified_oracle_causal_benchmark`相当の本格的な新規学習コード(8操作×5シード)を新たに書いて実行する」ことだと判明。この訂正済みコストを再提示した上で、**ユーザーは「R3-010を現状のまま正直に報告する」を選択**。共有キャッシュは一切変更していない。

### 3.4 発見B(発見Aとは独立): COUNT↔BIND のL3 routing自体も劣化

**routerは重みに依存しないfreshな再較正のため発見Aでは説明できない別の劣化。** R3-006自身の**無改変のdispatcherをそのまま再実行**して独立検証: 結果が`FAIL`に変わり、「R0はもう両方向の閾値を満たさない」と報告(当時の近1.0報告から反転)。選択されたvariantも`scoped_pairwise_margin`から`scoped_ce`に変化。新しいCoreの表現空間がCOUNT/BINDを分離しにくくなっている可能性。

### 3.5 その他の実測結果(C5、統合後の最終系)

- nominal-lite matrix: L0/L1は合格、L2/L3/L4のtop-1とL4 family-top1は不合格(発見Bおよび「C0-C2のrouterはreal scorerと一度もjoint較正されていない」という別の狭い合成効果による)。
- safety(false-accept率3種): 全条件でPASS、C1-C5は構造上完全に同一(検証済み)。
- legacy regression(C5 vs C0, 10操作): 技術的にはPASS(`mean_regression_pp=-6.12`)だが、これはC0も発見Aで同様に壊れているため「劣化が測定されない」だけであり、「legacy機能が正常」を意味しない、と明記して開示。
- SHIFT gate(C5独自のqueryで再測定): 平均EM 0.6。committed seeds(10/11/14)はEM=1.0でR3-009と一致。rolled-backのseeds(12/13)はEM=0.0(未学習ベースライン、R3-009自身の0.918/0.787=却下されたcandidateの数値とは別物であることを明記)。
- freeze audit: 各ラダー遷移で意図した箇所のみが変化していることを`_state_dict_hash`比較で確認済み(C0→C1は無変化、C1→C2はrouterのみ、C2→C3は無変化(formula変更のみ)、C3→C4はscorerのみ、C4→C5はbankのみ、かつrolled-back seedsは無変化)。

### 3.6 テスト・実行

`tests/test_paired_integration_regression.py`(新規28件、CPU-only)。フルリポジトリsuite: **1931 passed, 0 failed**(1903 + 28新規)。`ruff check .`/`mypy src/apc` 共にclean。実GPU実行: 61.5秒。

---

## 4. 現状の総括と、判断が必要な論点

### 4.1 何が「安全」で、何が「未解決」か

- **R3-010自身が実装した統合メカニズム(6条件ラダー構築・freeze audit・SHIFT versioned graftの再利用)は正しく動作することを検証済み。**
- **G4のFAILは、5つの修正を組み合わせたこと自体の欠陥ではない。** 発見A・発見Bという、R3-009の副作用(2026-09-07に開示されていたが、実際の深刻度はそれよりずっと大きかった)に起因する。
- development seeds 10-14の共有primitiveキャッシュ(`runs/phase_a2_bank_scaling_benchmark/seed_{10..14}/primitive_bank_16.pt`)は、**現時点で16 primitive中10個が新しいCoreと不整合**であり、これはR3-010に限らず**R3-004〜R3-008のどの milestone script を今後再実行しても**同じ問題を踏む。

### 4.2 選択肢(次の一手)

1. **R3-011へ進む**(新Gateの事前封印)。ただしR3-011/012はsealed seedsを対象とするため、development seeds特有のこの問題そのものはR3-011/012の評価には直接影響しない可能性がある(要確認)。ただし、B-C005R3-006/007/008の「development上の`VALIDATION_PASS`」という既存の完了済み結果の信頼性は、今回の発見によって疑問符が付いた状態。
2. **development seeds向けキャッシュの本格修復を新タスクとして立てる**: `run_unified_oracle_causal_benchmark`相当の新規学習コードを書き、8操作×5シードを新しいCoreに対して再学習する。実装コスト・GPUコストともに中規模(R3-009本体に近い規模感)。完了すればR3-004〜010のdevelopment上の数値をすべて「新しいCoreとの整合が取れた状態」で再検証できる。
3. **現状のまま容認し、sealed経路(R3-011/012)の結果を最終判断基準とする**: development seeds上の`VALIDATION_PASS`群(R3-006/007/008)はそもそも「sealed partitionの修理を証明するものではない」と各ADRが明記済みなので、最終的な科学的主張はR3-012のsealed Gateに委ねる、という整理も可能。
4. **R3-006/007/008を対象を変えて再実行**(development seedsではなくvalidation seeds 15-19、またはR3-011/012のsealed経路)。ただし現行タスク文書のスコープ外であり、新タスクの設計が必要。

いずれも**新規タスクとして明示的な指示が必要**(STOP ruleおよびG4失敗時の「原因機構の新タスクを別途設計し、自動で戻って複数修正を混ぜない」という規定による)。

---

## 5. 参照先

- ADR本文: `docs/DECISIONS_PHASE_B.md` の ADR-0082(R3-001)〜ADR-0091(R3-010)。
- ADR索引: `docs/DECISIONS.md`。
- 実行成果物: `runs/phase_b_b2_post_d2/r3_00{1..9,10}_*/`(いずれも`.gitignore`対象、リポジトリ非追跡)。
- R3-010の中心的発見の実装上の根拠: `src/apc/evaluation/paired_integration_regression.py` のモジュールdocstring(`_stale_primitive_cache_finding`関数のdocstringに詳細な再現手順を記載)。
