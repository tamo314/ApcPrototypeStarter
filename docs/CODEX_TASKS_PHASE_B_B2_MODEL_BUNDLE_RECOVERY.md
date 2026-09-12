> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# AI Coding Tasks — Phase B / B2 Model Bundle Recovery

> **現在地・実行順の正本:** [Phase B 再開計画](exec-plans/active/PHASE_B_RESTART.md)（2026-09-12、ADR-0127）。
> 本文は既存の仕様・作成当時の状態を保持する。現在の進捗と今回の継続指示は再開計画を参照。

**正式ID：B-C005REC-001～B-C005REC-008**  
**版：1.0 / 2026-09-07 / 状態：実装前の復旧指示**  
**最初はREC-001のみ。R3-011／012、B-C006、Task Inferenceはblocked。**

## 0. この文書で許可する作業

R3-010の失敗後に、整合したCore・PrimitiveBank・router・ArgumentScorer等を一式で固定し、必要なら欠落した生成経路を復旧する。既存一式の復元可否を先に確認し、再構築は不足dependencyに限る。既存のR3修正を新しい同一親モデル上で再評価するところまでを定義する。

**この文書は、研究上のFAILを閾値緩和で通すこと、sealedで修正を選ぶこと、新しいloss探索、Phase Aの全再実験を許可しない。** 実装はその都度ユーザーが指定した一タスクだけ。

[実行計画](exec-plans/active/PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md)、[復旧受入条件](EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md)、[ModelBundle契約](design-docs/B2_MODEL_BUNDLE_RECOVERY_CONTRACT.md)、[エージェント規則](AGENTS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_ADDENDUM.md)を合わせて読む。資料と提案の区別は[source notes](research/B2_MODEL_BUNDLE_RECOVERY_SOURCE_NOTES.md)に従う。

### 0.1 報告から引き継ぐ事実

[S1]の記録では、R3-009後の新Coreに以前のbankが組み合わさり、未修正6操作がC0～C5・全seedでEM=0となった。`get_or_build_16_primitive_bank`のcache-miss経路は16個中10個を学習しない。共有cache削除だけでは復旧しない。[S1: §3.3]

COUNT↔BINDのrouting劣化は別件として報告されている。新Coreによる表現変化は可能性であり、原因確定済みと扱わない。R3-010では修正bank/routerをsequential runnerへ注入できず、K/C/N/Rを代替試験にした。[S1: §3.1、§3.4]

現コード・checkpoint・JSONはここで独立検査されていない。レポートの記述と実装が異なる場合は差分を残す。

### 0.2 共通原則

- 全過去run／ADRを保存する。新runで旧PASS／FAILを上書きしない。
- `h_content=f(content)`とshared/task-blind Coreを維持する。operation別Coreへ分割しない。
- loaderは読込専用。学習・再較正・variant選択・random fallbackはしない。
- buildとrepair preparationは明示commandで実施し、evaluationは保存済みartifactのみ使用する。
- 同じseed／shape／timestampではなく、dependency hash・schema・学習来歴・機能検証で一式を認証する。
- candidateのidentity、機能adequacy、library全体の不足を分離する。UNCERTAINをPLASTICへ自動変換しない。
- full repository testsのPASSと、GPU上の学習済みモデルの機能PASSを分ける。
- 該当Gateが失敗したら依存タスクを停止する。報告のみのREC-008は早期終了時にも作成できる。

### 0.3 一式の状態

| 状態 | 説明 | 許可 |
|---|---|---|
| `LEGACY_UNVERIFIED` | 出自／依存関係が未確認 | 読み取り監査のみ |
| `STAGING_INCOMPLETE` | build途中・欠落あり | build workspaceのみ |
| `COHERENT_LIMITED` | 全dependency整合、学習済み。宣言済みSHIFT不足等を明記 | 許可した復旧・比較モードのみ |
| `NOMINAL_VALIDATED` | 明示capability scopeで機能検証済み | そのscopeの通常実行 |

hash一致だけで`NOMINAL_VALIDATED`へ上げない。SHIFT不足を持つbaselineをnominalとして使わず、C0の全予定指標からも除外しない。certificationはcomponent内容と別の証拠であり、過去の来歴不明を消すものではない。

---

## B-C005REC-001 — Artifact Preservation, Dependency Inventory & Restore Decision

### 目的・前提

共有cacheを触らず、既存一式の復元が可能か判断する。R3-010の完了報告を開始点にできる。学習しない。

### 確認する場所

以下は[S1]が報告した場所・symbol。実repoで定義元、caller、書込先を確定する。

```text
src/apc/evaluation/paired_integration_regression.py
  _stale_primitive_cache_finding / _build_frozen_base_system の定義・呼出し先
src/apc/evaluation/incremental_router_benchmark.py
  get_or_build_16_primitive_bank
src/apc/evaluation/sequential_closed_loop_benchmark.py
src/apc/evaluation/count_bind_key_scoring_repair.py
src/apc/evaluation/bind_argument_scorer_repair.py
src/apc/primitives/argument_scoring.py
run_unified_oracle_causal_benchmark 相当の既存生成レシピ
runs/phase_a2_bank_scaling_benchmark/seed_{10..14}/primitive_bank_16.pt
R3-009が作成したshared_encoder.ptの実path
R3-004～010のconfig / checkpoint / JSON / selected variant記録
```

### 実施

1. 本パックをdocsへ導入し、root AGENTSとR3のplan/taskに復旧分岐・block状態を追記する。旧本文を削除しない。
2. root、run、checkpointを読み取り監査する。helperを呼ぶ前にcache miss時のtraining／初期化／書込みを調べる。書込みがあるhelperを「読込テスト」として呼ばない。
3. 調査開始・終了時の既存artifact hashを保存する。必要なsnapshotは新namespaceへcopyし、元ファイルは変更しない。大容量copyが難しければhash台帳を先に作り、未snapshotを明記する。
4. 各modelについてCore、task側、decoder、token/schema、全16 primitive、router、scorer、controller、verifier、recipe/cache依存を列挙する。別名同一checkpointを独立modelと数えない。
5. **全16個のbuild coverage表**を作る。8 canonical、SWAP_PAIRS／INVERT_HALF、残り6個の実registry名について、既存学習関数、入力Core、source artifact、optimizer／step／budgetの取得先、freeze区間、出力先を示す。残り6個の名称を想像で埋めない。
6. git履歴や指定された過去run等から、整合したdevelopment一式を復元できるか確認する。作業treeを過去commitへ切替えたり、他の作業を巻き戻したりしない。旧sealedを都合のよい親として選ばない。
7. 読込だけで既に完結している許可範囲のartifactに限り、小さなdirect診断を実施できる。これはsource状態の確認であり最終機能認証ではない。raw／routing／verifierを分けてログ化する。
8. 各modelを`RESTORE_CANDIDATE / PARTIAL_REUSE_BUILD_REQUIRED / REBUILD_REQUIRED / UNRESOLVED`へ分類し、根拠と依存closureを保存する。

### 出力・Gate

`artifact_inventory.json`、`component_dependency_matrix.json`、`primitive_build_coverage.json`、`restore_decision.json`、`historical_claim_scope.json`。

**RG0**は調査の完全性で判定する。復元不能でも根拠・不足経路が明らかなら調査は完了。見つからない良い重みを捏造せず、次に必要なbuildを明示する。

**STOP：共有cache削除・再学習・REC-002自動実行は禁止。**

---

## B-C005REC-002 — Immutable ModelBundle, Strict Loading & Qualification Contract

### 目的

Coreだけ差し替わる構成、未学習bank、異なるscorerとの暗黙混在をロード時に拒否する。新API名は提案であり、既存abstractionを優先する。

### 実装

1. [ModelBundle契約](design-docs/B2_MODEL_BUNDLE_RECOVERY_CONTRACT.md)のmanifest／content hash／qualification schemaを実装する。
2. loaderはbundle manifestを明示入力とし、依存Core、task representation、decoder、vocab、argument schema、bank key-ID mapping、scoring formulaを検査する。欠落・不一致はtyped errorで停止する。
3. partial `state_dict`読込、欠落keyのrandom初期化、違うseedへのfallback、directory内の最新timestamp選択は禁止。model constructionの初期化値を完全checkpoint読込の代用にしない。
4. legacy importは新namespaceへのcopyと来歴記録に限る。UNKNOWNを自動的にtrained／clean exposureへ変換しない。機能認証はREC-004以降。
5. 内容hashとfunction certificateを分離する。formula-onlyのSELECT修正でも別execution signatureになるよう、code／formula versionをmanifestへ含める。
6. named capability scopeを要求する。diagnostic-only bundleで通常nominal実行を要求されたら拒否する。意図した未較正対照は明示diagnostic modeだけに限定する。
7. compatibility tableはrouter＋scorer＋lambda＋argument schema＋適用policyの組合せを含める。hard-negative level等のevaluation labelをruntime特徴量へ入れない。現コードがこの区別を満たせない場合は違反を報告し、勝手に全levelへscorerを広げない。
8. 新artifactはstagingで作り、検査後に新IDへpublishする。既存bundleをin-place更新しない。

### 必須テスト

Coreの1 tensor差分、primitive欠落、偽の同seed、bank key順序交換、schema不一致、scorer pair不一致、formula変更、checkpoint破損、途中write、UNKNOWN来歴、capability不足、loader内builder呼出しをそれぞれ検出する。

同一componentをbyteコピーしたものと独立学習したものを区別し、raw file hashとcanonical state hashの両方を保存する。

### 出力・境界

`bundle_schema.json`、`loader_contract_tests.json`、`compatibility_rules.json`、`legacy_import_policy.json`。**RG1 PASS**が必要。ここでは本格学習・科学性能評価を行わない。

---

## B-C005REC-003 — Complete Build DAG & Preregistered Recovery Protocol

### 目的

cacheが空でも、宣言した全16 primitiveを漏れなく構築できる経路を用意する。8操作だけのbuildを16-skill bank完成と呼ばない。

### 実装・仕様固定

1. REC-001で調べた既存レシピをreuseし、明示builderを作る。既存設計・レシピが文書で確定していれば、実行コードが消失していても、その仕様に沿う最小の学習ループの再実装を許可する。既存helperが10個を学習しないことを、negative fixtureで検出する。
2. buildを概ね「Core／decoder／schema確定 → 8 canonical → 既存2能力 → incremental6能力 → router／scorerの許可した較正 → certificate」に分ける。ただし既存canonical recipeがshared representationとoperatorのjoint学習を要する場合はそのstageを一体化し、**Coreを確定した後に残りの依存物を作る**。operator別Coreは作らない。
3. 再利用componentは正確なdependencyとrecipeが一致する場合だけ使う。新Coreへ旧primitiveを載せて、たまたまloadできたことを学習済み扱いしない。
4. 16行の各primitiveに`RESTORED_VALIDATED / TRAINED_THIS_BUILD / EXPLICIT_PARAMETER_FREE_APPROVED`の根拠を要求する。最後の分類は実設計で既に正当化されている場合に限る。未学習random operatorはどの分類にも入らない。
5. 各stageのtrainable／frozen parameters、target data、step、容量、optimizer、stop条件、dependency hashを保存する。Core作成は既存recipeの再現であり、新architectureの探索ではない。generatorのoracleは教師生成に使えてもruntime計算の代替にはしない。
6. build／calibrationとevaluationを別entry pointにする。評価関数から`get_or_build`を呼ぶ構造を新経路で禁止する。
7. 1-seed pilotはmodel seed10、cohortは10～14を事前固定。training data／runtime support／query／reference／shadowを分け、validation15～19、sealed30～34等の実mappingを監査する。seedをmodel IDへ暗黙変換しない。
8. sourceから取得したrecipe／budget、復旧用機能floor、qualification scope、data hash、scoring適用policyを`recovery_protocol.json`へ固定する。復旧専用の新規基準は[受入条件§2](EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md)を使う。
9. K/C/N/R注入試験用に、既存のdevelopment short stream、初期bank membership、recipe depth／budget、controller／replay recipe、Nの未導入状態を登録する。16-skill full bankへ既に含まれる能力をNと呼ばない。
10. レシピ不足は`RECIPE_UNAVAILABLE`。既存仕様の再実装では補えない新しい学習機構・未定ハイパーパラメータの設計が必要なら停止する。既存budgetを取得できずに「十分長く学習」としない。

### テスト・出力

全16行coverage、dependency順、stage resume、upstream変更でdownstream cache無効化、空cache、途中失敗、namespace分離、sealed guard、評価中training禁止、5モデルの独立性契約をCPU tiny fixtureで試験する。

`build_plan.json`、`recipe_inventory.json`、`training_budget_manifest.json`、`recovery_protocol.json`、`legacy_stream_manifest.json`、`build_dry_run.json`。

**RG2 PASS**で固定。まだ5-model学習を開始しない。機能floorの変更が必要なら測定前に独立した仕様改訂として記録する。

---

## B-C005REC-004 — One-Seed Restore / Clean Build & Fresh-Process Validation

### 目的

事前指定seed10で、復元または欠落buildから正常な読込までを通す。1 seedの成功を一般化の主張にしない。

### 実施

1. REC-001の復元候補があれば新namespaceへimportし、依存検査・機能検査を行う。不合格な復元候補は保存し、事前登録した再構築branchへ進む場合も別runで区別する。
2. 復元不能／不足stageだけをREC-003の固定recipeで構築する。empty namespaceが必要な経路では旧global cache参照を禁止する。
3. build metadataとraw neural executionの両方で16個を確認する。optimizerのstepが走ったことだけで性能PASSとしない。
4. direct/oracle callで全16個の絶対EM、token accuracy、loss、合法argument／length strataを測る。Wrong／None／Wrong argumentは既存causal検証の適用可能なものを実行し、意味的に同値なcallには誤ったcausal gapを要求しない。
5. router top-1とraw primitive EMを別表にする。verifierはraw executionの欠陥を隠すためのフィルタとして使わない。
6. 互換したtask encoder／router／scorerの既存の較正はbuildで完了させる。L3/L4の未解決性能はbaseline defectとして残し、新lossを試さない。
7. 学習processを終了し、別processでmanifest指定loadする。旧cacheを検索できないtest fixture、builder禁止patch、異なるworking directoryで、同じsample ID・component hash・scopeが再現することを確認する。
8. 一部成績が低ければ訓練epochを自動追加しない。pipeline不具合／機能不足／reference未確定／resource不足を分ける。

### 出力・Gate

`pilot_build_report.json`、`all_primitive_execution.json`、`fresh_process_report.json`、`qualification.json`、`side_effect_audit.json`。

**RG3**は全dependency・全16個の学習coverage・fresh-loadと、復旧用絶対性能floorを要求する。既知SHIFT不足だけは事前宣言した`COHERENT_LIMITED`として扱える。ほかのprimitiveの全滅を「既知欠陥」に追加して通さない。

復元のみなら`RESTORE_VALIDATED`、clean buildを実測した範囲は`CLEAN_BUILD_EXERCISED`と別表示。失敗なら5-model展開は停止。

---

## B-C005REC-005 — Five-Model Coherent Cohort & Frozen Parent Certification

### 目的

1-seedで成立した同じ生成契約を、固定した5モデルへ展開する。seed10のpilot artifactは同じprotocolなら再利用可能。

### 実施

- model seeds10～14をすべて対象にし、失敗seedを置換・除外しない。途中失敗で残り未実行ならplanned／executed／missingを明記する。
- 各Coreからbank／router／scorerまでのdependency closureをbundleごとに検査する。他modelのよいprimitiveを混ぜない。
- source不足で旧modelを復元できない場合は同じ固定buildレシピで作成する。同じCoreのcopy5個を独立modelと数えない。
- 全16操作、各seedの絶対性能を確認する。復旧floor／capability scopeはpilot後に変えない。
- SHIFTが不足していれば、その**実際のtrained baseline**をversion付きで保存する。R3-009の却下candidateの成績をinstalled baselineの成績として記録しない。
- immutableな`parent_cohort_manifest.json`を作る。以後REC-006／007はそこに記録したIDだけを入力にする。
- 成績を見て「validation seedsで再学習」「sealed用Coreを借用」に切替えない。

### 出力・Gate

`cohort_qualification.json`、`parent_cohort_manifest.json`、`per_model_operation_metrics.json`、`distinct_model_audit.json`。

**RG4 PASS = RECOVERY_COHORT_READY**。SHIFT限定不足を持つ場合は`nominal_ready=false`を併記し、R3-009／G4はPASSにしない。未修正15個のfloor失敗または不整合がある場合は後続の修正比較を止める。

---

## B-C005REC-006 — Explicit Runtime Injection & Fixed-Recipe Repair Preparation

### 目的

修正部品を評価時に再学習せず、互換するbundle一式を直接注入できるようにする。必要な再学習はこのtaskの**preparation区間**で固定レシピだけ実施する。

### A. 注入経路の実装

1. `sequential_closed_loop_benchmark.py`等の現契約を調べ、bundleまたはcomponent providerを明示的に渡す最小adapterを作る。旧defaultの意味は勝手に変えない。
2. frozen evaluationはloader以外のbuilder、router calibration、variant chooserを呼ばない。旧helperを再利用する場合、計算部分とget-or-build部分を分け、既存callerの後方互換を試験する。
3. 実K/C/N/Rでは、immutable初期bundleから作るisolated working copyを渡す。Nによる既定のtemporary learning／consolidation／router updateだけを許可し、共有cacheへのwriteは禁止する。
4. 初期bankのviewはREC-003で宣言したものを使う。N候補の既学習primitive／key／recipe／replayが隠れて残らないようsubset closureを検査する。来歴上の既往露出は別途開示し、未見family主張をしない。
5. 入出力schemaやevidence次元を変えず、新verifierのUNCERTAINを明示的に扱う。無理に0値を既存controllerへ入れない。wrapperでは解決しない新controller policyが必要なら停止する。

### B. R3修正の再適用・凍結

6. 新親とcompatibleな旧修正artifactはhash確認後に取り込める。incompatibleならR3の採用済みrecipeだけを明示preparationで再実行する。source・recipe取得不能なら`REPAIR_RECIPE_UNAVAILABLE`で止める。
7. R3-006の採用済み方式は当時の`scoped_pairwise_margin`を第一の固定再検証対象とする。[S1: §2] 今回の失敗を見て`scoped_ce`へ自動選択し直さない。必要な対照は事前固定し、結果が悪ければscope reviewにする。
8. SELECTは既存BCEに対するsigmoid読み出し修正をartifactのformula versionとして適用する。BINDは取得した既定`stratified_value_coverage` recipe、SHIFTは取得した既定`iid_baseline` recipeを上限budget内で再適用する。これらの再確認を新しい一般化成功と呼ばない。
9. router/scorerのpairingを準備時に固定する。共同較正が実契約上必要なら既存レシピだけで別prepared artifactを作り、変更tensor集合・適用scopeを記録する。keyだけの変更と説明しながらscorerまで変わることを禁止する。
10. C0→C5の系譜を作る。共同較正のため一段が複数機構を変えるなら追加対照を保存し、`COMPOSITE_DELTA_NOT_ISOLATED`と表示する。因果分離が成立しなければ個別修正への帰属はUNRESOLVEDとし、勝手にPASSしない。
11. SHIFT replacementは独立shadow判定・version transactionを使う。失敗時は同じ親のtrained SHIFTへrollbackし、未学習fallbackを使わない。旧3/5だけを持ち越さない。

### 出力・境界

`runtime_injection_contract.json`、`prepared_repair_lineage.json`、`pair_compatibility_matrix.json`、`repair_recheck.json`、`no_implicit_build_test.json`。

**RG5は注入・lineage・比較の成立**で判定する。局所性能FAILが残っても、それを明示してREC-007の既定評価を行うことは可能。ただしmissing artifact／不整合／不正なcontrolなら評価を始めない。探索的な新修正は実施しない。

---

## B-C005REC-007 — Artifact-Only Integration & Real K/C/N/R Regression

### 目的

REC-006で準備・凍結した一式を評価する。R3-010のnominal-lite代替ではなく、実runtimeの短縮K/C/N/Rを通す。

### 評価条件

```text
C0: 同一の復旧親 + 旧verifier／旧scorer formula
C1: C0 + R3-005 verifier
C2: C1 + 固定COUNT↔BIND修正
C3: C2 + SELECT formula修正
C4: C3 + BIND head修正
C5: C4 + 合格したSHIFT version置換（不合格seedはtrained親にrollback）
```

C0自体のraw execution性能と既知defectを先に表示する。各条件は同じ初期parent、dataset、logical candidate graph、scoring適用契約に基づく。学習・variant選択は評価関数から呼ばない。

### 実施

1. REC-003で固定したdevelopment／validation入力でnominal matrix、各pair／args層、same-identity stressを測る。旧不十分SHIFTを残す場合は対応Coreと一緒に独立fixtureとして扱い、新Coreへ不整合graftしてstressにしない。
2. raw／oracle EM、routing／args、verifier verdict、conditional／unconditional EM、coverageを分離する。未出力を分母から消さない。
3. legacy streamは登録済みK/C/N/Rを全条件で実行し、N→commit→Rの順を守る。Nの事前membership、oracle label漏れ、no-task経路、composition-before-plasticを検査する。
4. 適応があるstreamは、初期bundleを条件ごとに複製したworking stateで実行する。許可したepisode内更新をdiffとして保存する。static matrixのfreeze規則と混同しない。
5. Nが失敗した場合も予定されたRを分母から消さず、`DEPENDENCY_FAILED`として記録する。成功N条件付きrecurrenceと予定全streamの利用可能性を併記する。
6. Rではfresh process／持続stateの許可したloadを使い、再adaptation=0、temporary params=0、再commit=0を確認する。initial bundleのimmutable性とworking childの履歴を別監査する。
7. absolute baseline、after、paired deltaを全seed・全operationで保存する。両方EM=0なのに回帰なしとして合格させない。
8. [旧R3実験計画snapshot](research/evidence/R3_EXPERIMENT_PLAN_SUPPLIED.md)のG4／局所Gateをそのまま計算する。resource等で全部測れなければ部分結果にとどめる。G1不足を無かったことにしない。

### 出力・結果

`integration_recheck.json`、`absolute_legacy_capability.json`、`legacy_kcnr_recheck.json`、`same_identity_safety.json`、`cost_accounting.json`、`g4_recheck.json`。

**RG6**は評価の実施・比較成立の結果。**G4_RECHECK_PASS／FAIL／UNRESOLVED**は研究結果として別欄。旧R3-010はFAILのまま保存する。

Gate失敗は新lossを調整する理由にせず、REC-008へ未解決事項を引き継ぐ。G4が通ってもsealedへ自動移行しない。

---

## B-C005REC-008 — Recovery Handoff & Remaining-Block Decision

### 目的

復旧の成果、未達、次に実行可能な最小タスクを明示する。報告・ADR・navigation更新のみ。途中で復旧が止まった場合も実施可能。

### 報告するもの

- 何を復元し、何を再構築し、何が未実測か。全16個のcoverageと5モデルの独立性。
- 初期artifact／復旧親／各修正childのIDと再現command。shared cacheの前後hash。
- `RECOVERY_COHORT_READY`の可否と、SHIFT限定defectの有無。
- R3-006／007／008／009の新親上の結果、COUNT↔BINDの比較可能性、scorer pairの制限。
- G4、実K/C/N/R、absolute legacy、safetyとavailabilityを別欄で判定。
- G1の`PROTOCOL_INSUFFICIENT_RELATIONS`と`MINING_HOLDOUT_ONLY`を保持。復旧cohortはtargeted regression用でありunseen-relation用と主張しない。
- 資料に記録された旧FAIL／旧PASSの有効scope。新runを旧runの完全再現と呼ばない。

### 最終ラベル

```text
RECOVERY_INFRASTRUCTURE_COMPLETE_RESEARCH_BLOCKED
RECOVERY_COMPLETE_G4_RECHECK_PASS_RELATION_BLOCKED
RECOVERY_BLOCKED
```

RG0～RG6と研究Gateの一覧を併記する。完了番号だけで判定しない。全面再学習未実測の経路は別途明記する。

### STOP

R3-011／012、B-C006、Task Inferenceを実行・自動解除しない。sealedへ移るには、残るGateとrelation仕様を解決し、別の明示指示が必要。新研究の提案は測定で支持された一つの機構に限定する。

---

## 1. 推奨配置とCLI

既存packageの中だけを使う。同等のabstractionがある場合は重複作成しない。

```text
src/apc/utils/model_bundle.py                         # manifest/hash/load契約。提案
src/apc/evaluation/model_bundle_recovery.py            # audit/build/qualification。提案
src/apc/evaluation/paired_integration_regression.py    # 報告上存在。注入経路
src/apc/evaluation/sequential_closed_loop_benchmark.py # 報告上存在。注入経路
scripts/run_phase_b_b2_model_bundle_recovery.py        # 一task明示dispatcher。提案
configs/phase_b_b2_model_bundle_recovery_<task>.yaml
tests/test_model_bundle_contract.py
tests/test_recovery_build_coverage.py
tests/test_recovery_runtime_injection.py
runs/phase_b_b2_model_bundle_recovery/<task>/<run_id>/
runs/phase_b_b2_model_bundle_recovery/bundles/<bundle_id>/
```

新CLIを実装する場合の例（現時点で実在するcommandとの主張ではない）：

```bash
python scripts/run_phase_b_b2_model_bundle_recovery.py --task B-C005REC-001 --config configs/phase_b_b2_model_bundle_recovery_rec001.yaml --run-dir runs/phase_b_b2_model_bundle_recovery/rec001/run_001
```

REC-006／007の評価入口にはbundle/cohort manifestを必須にする。全学習を開始する`--all`や、latest bundleを自動探索するflagを追加しない。既存dispatcherを拡張するなら同じ契約を守り、実commandを完了報告に載せる。

## 2. 全taskの完了報告フォーマット

```text
Task ID / requested scope:
Implementation completion:
Recovery gate / research gate:
Files changed / tests / experiment commands:
Source facts / new decisions / deviations:
Input and output bundle IDs / parent lineage:
Model seeds / data seeds / distinct checkpoint count:
Primitive coverage (planned / restored / built / missing):
Absolute capability / known defects / reference unresolved:
Implicit train/build calls detected:
Original artifacts preserved:
Run artifacts / ADR:
Remaining blockers / next task (not executed):
```

RG失敗・未実施・不明を明示する。1931 tests PASSのような過去結果は新commitのtest結果に代入しない。全CPU試験が成功しても、missing neural artifactを機能PASSにしない。
