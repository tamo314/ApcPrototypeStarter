> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# AI Coding Task — Incremental Primitive Budget Calibration & RG3 Recheck

**正式ID：B-C005REC-004A**  
**版：1.0 / 2026-09-08 / 状態：追加タスクの指示書。未実装・未実行。**  
**位置づけ：REC-004のRG3 FAIL後、REC-005の前に挿入する一件の追加タスク。**

## 0. このタスクの目的と許可範囲

復旧pilotで未達だった **CYCLE_FOUR / MIRROR_HALVES / ROTATE_TRIPLETS / SWAP_ENDS** の4操作だけについて、既存レシピの有限な学習step比較を行う。Coreと合格済み12操作を固定し、機能不足の原因を局所化すると同時に、必要な場合は4操作の学習予算を改訂する。候補選択後、新しい独立queryでRG3を再判定する。

**本タスクは「1000 stepsが原因」と決めつけない。** 操作別の学習曲線、学習例への適合、独立validation、誤り位置を確認する。学習延長で解決しなければ、測定結果を保存して停止する。

一件のタスク内にA〜Eの段階を置く。ユーザーが `B-C005REC-004A` を指定した場合、各前提が成立する限り、その一件の段階A〜Eを実行できる。各段階で改めて確認を求める必要はない。ただし下記のSTOP条件で依存作業を停止し、REC-005を自動実行しない。

```text
REC-004 / ADR-0095: RG3 FAIL（保存）
    ↓
B-C005REC-004A
    A. 既存artifact・学習／評価契約・資格scopeの確認
    B. 比較protocolの固定
    C. 未達4操作だけの有限step比較
    D. validationによる候補選択と新bundle準備
    E. 独立query・fresh processによるRG3再判定
    ↓ PASSした場合のみ
REC-005が実行可能になる。ただし別の明示指示が必要。
```

R3-011／012、B-C006、Task Inferenceは本タスクの結果にかかわらず自動解除しない。relation不足、legacy streamのNとのmembership衝突、他seedのSHIFT不足は別blockとして残す。

---

## 1. 根拠・新規設計・未確認事項

### 1.1 根拠資料

- **[S1] REC-001〜004結果／ADR-0093〜0095。** 添付の原文snapshot：[`REC001_004_RESULT_SUPPLIED.md`](research/evidence/REC001_004_RESULT_SUPPLIED.md)。今回の数値はADR-0095 Evidence 4〜8とConsequencesによる。
- **[S2] 復旧タスク指示書。** 現repoの `docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`、特にREC-003〜005。配布時snapshot：[`MODEL_BUNDLE_RECOVERY_TASKS_SUPPLIED.md`](research/evidence/MODEL_BUNDLE_RECOVERY_TASKS_SUPPLIED.md)。
- **[S3] 復旧受入条件。** 現repoの `docs/EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` §2、§6、§7。配布時snapshot：[`MODEL_BUNDLE_RECOVERY_ACCEPTANCE_SUPPLIED.md`](research/evidence/MODEL_BUNDLE_RECOVERY_ACCEPTANCE_SUPPLIED.md)。
- 現repoのModelBundle契約、agent addendum、REC-003の `recovery_protocol.json` / `training_budget_manifest.json` を実装前に照合する。

S1は実行担当者の報告であり、この指示書作成時点でコード・checkpoint・JSONを独立再検証したものではない。snapshotは読解の根拠であり、現在の実API・重みの存在証明の代用ではない。不一致は黙って解消せず、差分を記録する。

### 1.2 引き継ぐ結果

pilot model seedは10。CoreとSHIFTを復元し、残り15 primitiveとrouter／ArgumentScorerを実際に学習した。全16操作のfresh-process離散予測一致が報告されている。ただし次の4操作がquery EM 0.95を下回った。[S1: ADR-0095]

| 操作 | REC-004 Correct EM | Token accuracy | このタスクの扱い |
|---|---:|---:|---|
| CYCLE_FOUR | 0.8906 | 0.9861 | 学習対象 |
| MIRROR_HALVES | 0.0098 | 0.6716 | 学習対象 |
| ROTATE_TRIPLETS | 0.9219 | 0.9902 | 学習対象 |
| SWAP_ENDS | 0.4736 | 0.9326 | 学習対象 |
| ALTERNATING_NEGATE | 1.0000 | 1.0000 | 凍結対照 |
| INCREMENT_MOD | 1.0000 | 1.0000 | 凍結対照 |

残り9非SHIFT操作はEM 0.9940〜1.0000、SHIFTは1.0000と報告された。したがって保護対象は11非SHIFT操作＋SHIFTの計12操作。

既存の `INCREMENTAL_6_BUILD` は各操作1000 stepsだった。しかし「1000 stepsで不足した」という観測だけでは、予算不足、仕様不一致、学習不足、汎化不足、表現／operatorの限界を区別できない。今回の原因ラベルは実験後に操作別に付ける。

### 1.3 今回新たに承認する設計

以下は旧結果ではなく、この追加タスクで新設する条件。

| 項目 | 新設する条件 |
|---|---|
| 正式task | B-C005REC-004A、一件のみ |
| model | seed10のREC-004と同じCore。新しいmodel seedを選ばない |
| 学習対象 | 上記4操作のみ |
| 総stepの観測点 | **1000 / 2000 / 4000 / 6000** |
| 予算上限 | 各操作、初期化から累計6000 optimizer updates。4操作合計最大24000 |
| validation | 各対象操作1024例、全stepで同じ入力。予算選択専用 |
| 候補選択 | validation sequence EM>=0.95を満たす最小の観測step |
| 最終query | 新しいREC-004A再判定用の各操作1024例。候補固定後に一度だけ評価 |
| 再判定 | 全16操作を表示し、非SHIFT15操作ごとのEM>=0.95を維持 |
| 後付け延長 | 6000超、別初期化の再試行、最終query後の再選択は不可 |

1024例・0.95の最終floorは[S2][S3]の継承であり、今回緩和しない。validation用1024例と候補選択規則は新設。query floorを満たしただけで統計的な `REF_ADEQUATE` を付与しない。[S3 §2.2]

---

## 2. 変更可能なもの／凍結するもの

### 2.1 モデル変更の許可集合

変更可能なのは対象4操作のprimitive重みのみ。対象の実physical IDとparameter prefixはregistry・manifestから取得する。名前や過去のID順から推測しない。

対象は同じphysical IDに対応する隔離candidateとして学習する。factoryがbankへ自動appendする場合は、その副作用を避ける既存の構築経路を使い、原bankの登録数やIDを変えない。

各対象を一つずつ学習し、専用optimizerのparameter ID集合がそのprimitiveの許可集合と一致することをassertする。既存trainerがbank全体をoptimizerへ渡す場合、他重みが実際に更新されないことまで検査する。単に `requires_grad=False` と書いたことをfreeze証拠にしない。

### 2.2 必ず保護するもの

- Core、Content Encoder、task-side encoder、decoder/readout、token/vocabulary、schema。
- 合格済み12操作：SELECT、COUNT、BIND、COPY、NEGATE、SWAP_PAIRS、INVERT_HALF、REVERSE、SORT、ALTERNATING_NEGATE、INCREMENT_MOD、SHIFT。
- Routerの共有重みと全key、key-ID mapping、ArgumentScorer、formula version、lambda。
- Controller、verifier、adequacy threshold、statistical contract、検索・support予算。
- 既存run、bundle、shared cache、R3-009のcommitted SHIFT原本。

parameterだけでなくbuffer・実行formula・schema・学習modeの影響も検査する。Content Encoderは構造的に `h_content=f(content)` を維持する。

### 2.3 許可される計測・文書変更

学習曲線・途中checkpoint・誤り集計・新runへのmanifest作成、既存dispatcherへの明示task追加、focused tests、ADR／navigation更新は許可する。計測のためのtrainer拡張は、計算・データ・更新順序を変えない範囲に限る。

**許可しないもの：** Core再学習、operator大型化／種類変更、損失・LR・optimizer・batch・dtype変更、training分布の再重み付け、curriculum、router再較正、引数head修正、oracleによるruntime置換、未達操作の除外、SHIFT以外への例外追加。

実際の教師・loss・indexing・loaderに契約違反が見つかった場合、学習予算変更と一緒に修正しない。再現テストと結果を保存し、該当依存段階を停止して別タスクへ引き継ぐ。

---

## 3. 段階A — Artifactと仕様の確認（新規学習なし）

### A1. 実repoを確認する

[S1]が報告した主な対象：

```text
src/apc/evaluation/model_bundle_recovery.py
src/apc/evaluation/incremental_router_benchmark.py
src/apc/evaluation/unified_oracle_causal_benchmark.py
src/apc/utils/model_bundle.py
scripts/run_phase_b_b2_model_bundle_recovery.py
scripts/rec004_fresh_process_check.py
```

`_train_single_primitive`、`bank.new_cross_position_primitive`、実際のgenerator・operation定義の定義元とcallerを調べる。既存helperを試しに呼ぶ前に、暗黙build／cache書込み／router較正がないことを確認する。

### A2. 親を固定する

REC-004が報告したmanifest：

```text
runs/phase_b_b2_model_bundle_recovery/bundles/2356543740ce566640f72767e73bd83955bc27bb825bb06c8fcffab03cf53995/manifest.json
```

上記は資料記載の確認対象。実ファイルの存在・ID・hashが一致することを検査し、configには確認済みの正確なpathを保存する。見つからなければ `PARENT_ARTIFACT_UNAVAILABLE`。最新bundle探索や別seedへfallbackしない。

親のmanifest、raw file hash、canonical state hash、全16 primitiveのversionとCore依存、training receipt、data lineageを新runへ記録する。親と共有cacheの開始／終了時hashを保存する。

### A3. 学習状態の保存状況を確認する

対象ごとに1000-step重みだけでなく、optimizer、scheduler、AMP scaler（使用時）、CPU／device乱数状態、data iterator／sample index、実LR、累計updates、初期化状態の保存状況を調べる。

- 必要状態と生成recipeが揃う：`EXACT_CONTINUATION`候補。
- 重みしかない、または生成状態を復元できない：`FRESH_PAIRED_REBUILD`。
- 既存レシピそのものが不明：`RECIPE_UNAVAILABLE`で停止。

重みだけからoptimizerを作り直して「1000→2000 stepsの厳密な継続」と呼ばない。全4操作について各一経路を、成績を見て選び直さず、証拠の有無だけで決定する。

### A4. 学習／評価の操作定義を照合する

4操作について、合法length、区間／端の扱い、端数長、padding／mask、token範囲、出力順、loss対象、decoder入力をsourceと既存testsから確定する。**operation名から具体的な変換規則を作らない。**

手で結果を追える小さな合法入力と、既存仕様が要求する境界条件で、generatorと評価対象の意味が一致するか確認する。oracleは教師生成・評価に限定し、primitiveのforwardへ入れない。

既存学習logがあれば、対象4操作のtraining loss、train-fit、validationの推移を抽出する。未保存なら `NOT_RECORDED` とし、最終EMから曲線や収束を捏造しない。

### A5. qualification scopeの確認

[S1]はRG3 FAILとnominal-mode load成功を併記する。これは要求scope次第で成立するため、直ちにbugとは断定しない。

`qualification.json`、manifest、fresh-processの `required_capabilities` を照合し、次を区別する。

1. 構造・依存・学習coverageが揃い、診断用にloadできる。
2. 非SHIFT15操作の復旧floorに合格している。
3. R3のreference契約まで満たし、あるnominal scopeが認証されている。

旧親に「全15非SHIFT操作がfloor合格」という資格を要求した場合、拒否または未認証となることを確認する。旧親の診断loadは可能でよい。

scopeが曖昧なら `QUALIFICATION_SCOPE_UNRESOLVED`。不十分なfull capabilityをnominalとして許す実契約違反なら `QUALIFICATION_CONTRACT_FAILURE`。どちらも新bundleの通常公開・RG3再判定PASSを禁止する。明示diagnostic modeで依存が正常に検査できる場合のみ、原因確認用の段階B/Cを継続できる。loader修正は本タスクに混ぜない。

### Aの出力

`parent_audit.json`、`resume_audit.json`、`operation_contract_audit.json`、`existing_learning_curve_audit.json`、`qualification_scope_audit.json`。

source／意味論不一致、学習対象外更新、Core不整合、親取得不能はtraining前にSTOP。単なる履歴log欠落はfresh経路を用いれば作業可能。

---

## 4. 段階B — 比較protocolの固定

### B1. 仮説と対照

各操作を別々に扱う。

- **H-budget：** 同じレシピを長く実行すると、独立validationで95%へ到達する。
- **H-generalization：** training例には適合するが、独立例・特定length等へ移らない。
- **H-unresolved：** trainingでも進まない、仕様不一致、最適化・表現・operatorの問題が分離できない。

1000 stepsが元々router向けの経路だったという報告を、過去APC全体で機能検証がなかったという意味に拡張しない。

### B2. step比較の契約

一つの対象operationにつき、一つの初期化／一つのtraining trajectory。累計stepが1000、2000、4000、6000になった時点のcheckpointを保存して比較する。**1000+2000+4000+6000 stepsの独立4本ではない。**

- fresh経路：6000 updates／操作、4操作なら最大24000。
- exact継続：既存1000から追加最大5000／操作。既往1000を費用・露出台帳から消さない。
- 成績が良くても悪くても、通常は6000まで同じtrajectoryを完走して曲線を得る。NaN／不整合／resource不足で止まったときは未実行を表示する。
- 別seed、再初期化、別LR、別loss、batch変更で追加探索しない。上限後も自動延長しない。

stepは**optimizer更新回数**。gradient accumulationがある場合はmicrobatch数と区別する。累計training examples、unique examples、sample-ID streamも記録する。online生成ならstep増加と新規訓練例増加が併存するため、「純粋な最適化回数だけの因果効果」と呼ばない。

### B3. LR scheduleと継続性

LR・weight decay・optimizer・loss・batch・schedulerはREC-003の実manifestとcall siteから取得する。資料の別trainerの数値を流用しない。

既存scheduleが総step数に依存する場合、1000-step条件と6000-step条件で最初の1000更新のLR列まで変わる設定にしない。既存の絶対step scheduleをそのまま延長できるか、学習前に調べる。延長規則が未定なら `SCHEDULE_EXTENSION_REQUIRED` で停止し、勝手にLRを再設計しない。

exact継続では1000時点のoptimizer／RNG／data位置とhashを照合する。fresh経路では初期stateから再生成し、新しい1000-step結果をpaired baselineとする。旧REC-004の1000-step結果は歴史比較として別列に置き、差があれば報告する。

観測用evalは独立RNGを使用し、終了後にtraining mode／RNG／buffer状態を戻す。途中評価を挟んだことが後続学習streamを変えないテストを作る。

### B4. データの役割

| role | 用途 | checkpoint選択への使用 |
|---|---|---|
| `train` | 既存レシピの学習stream | 学習のみ |
| `train_fit` | 実際に既に学習した例の固定部分集合、最大1024例 | 原因診断のみ |
| `budget_validation` | 対象4操作各1024例、全観測stepで固定 | これだけを予算選択に使う |
| `historical_query_replay` | REC-004の保存入力または同一ID再生成 | 親再現・凍結対照のみ。新holdoutと呼ばない |
| `recheck_query` | 全16操作各1024例、候補固定後の一回評価 | 使用禁止 |
| reference／shadow | 元の復旧契約で必要なscopeがある場合 | 予算選択に使わず元契約を維持 |

model seedは10のまま。data-role seedは現repoのdevelopment／validation policyに従い、明示namespace・generator version・sample indexから決定する。data seedから別modelを生成しない。旧sealed0〜4／20〜24、新sealed30〜34その他現repoでsealed指定されたdataへの学習・候補選択・新評価を禁止する。

`recheck_query`は本タスクの独立確認用であり、B2の未見relation sealed gateではない。生成recipe・分布・入力hashを学習前に登録し、モデル出力は段階Eまで見ない。

`train_fit`は本当に露出済みの例に限る。新しい同分布の例をtraining accuracyと呼ばない。既往例が追跡不能なら、その測定は `TRAIN_FIT_UNAVAILABLE` とする。

train／validation／recheck間は別role streamを使い、現在の4操作の学習streamと独立評価集合の重複をsample-IDと `(task, arguments, content)` digestで監査する。seedが違うだけで非重複と主張しない。重複処理は出力を見ない決定的手順として固定し、学習stream自体は比較途中に変更しない。履歴Core／既存12操作の学習露出を完全に確認できない範囲はUNKNOWNと開示し、「APC全学習に完全未見」という主張をしない。

### B5. 固定してから実行するもの

`budget_protocol.json` に次を保存し、hashを作る。

- parent bundle／Core ID、4対象と12保護対象のphysical ID・parameter集合。
- operation別のexact/fresh経路、初期化・optimizer・schedule・RNG・生成recipe。
- step観測点／上限、データrole、sampling分布、validation／recheck入力hash。
- 予算選択規則、metric定義、error strata、元RG3との継承／差分。
- 旧cache・source artifactの監査範囲、fresh-process device/dtype、既存再現許容差。

登録と実行の境界はコードで残す。1000-step評価後に上限や分布を書き換えない。この文書が許可する新規変更は、上記の有限step予算較正であることをADRに記録する。ADR番号は実repoの次番号を取得し、ADR-0095を改番しない。

---

## 5. 段階C — 4操作の学習曲線と誤り構造

### C1. 記録する基本値

全4操作×全観測stepについて、以下を保存する。

- 累計updates／training examples／unique examples、実LR、train loss。
- `train_fit`のsequence EM／token accuracy／loss、n、露出根拠。
- `budget_validation`のsequence EM／token accuracy／loss、分子／分母。
- per-length、境界／端数／合法条件の層別値、first-error位置。
- 実optimizer対象、Coreと12保護操作およびrouter/scorer等のhash。
- checkpointとtraining-state hash、時間、peak VRAM、NaN／gradient状態。

train-fitとvalidationが同じ単位・eval modeで比較できるようにする。token accuracyをsequence EMの代わりの合格基準にしない。

### C2. 操作別の重点観測

**CYCLE_FOUR／ROTATE_TRIPLETS：** sourceで確認したgroup境界・合法length条件ごとに、どの位置で誤るかを見る。端数groupの意味を名前から仮定しない。少数のerrorが系列EMへどう影響するかを示す。

**SWAP_ENDS：** source上の変換対象位置を確認する。length-preservingで位置対応が定義できる場合、`target_token != input_token` の位置を「観測上のchanged位置」として集計する。これを構造上の移動位置と混同しない。同値tokenで見かけ上変化しない例、identityで解ける例も全体queryから除外しない。changed位置が0ならその部分指標はnull＋分母0。

**MIRROR_HALVES：** train-fit自体が上がらないのか、validationでだけ低いのかを最初に区別する。区間分割・境界・出力順・paddingの合法仕様をsourceと照合し、仕様が一致していても低ければ、容量不足と即断せず `UNRESOLVED_WITHIN_FIXED_RECIPE` を許す。

### C3. Causal controls

既存 `_evaluate_primitive_arm` 相当の評価を使い、Correct／Wrong-family／None、および適用可能なWrong-argumentを記録する。操作schemaが許さない引数を対照として無理に作らない。

NoneとCOPYが同じ実装とは限らないので区別し、追加COPY対照は既存の合法なcallだけを使う。等価な出力になる入力へ一律のcausal gapを要求しない。新しいgap閾値を今回の成績から追加しない。

対照で選んだprimitiveの実行は許可された明示callである。Correct経路の「選んでいないprimitiveを裏で実行していない」という監査と、対照実験そのものを混同しない。

### C4. 曲線の解釈

| 観測 | 記載可能な結論 |
|---|---|
| paired 1000は低く、step増加でvalidationがfloor到達 | `BUDGET_EXTENSION_SUPPORTED_ON_PILOT` |
| fresh paired 1000が既に到達 | `ORIGINAL_FAILURE_NOT_REPRODUCED`。step不足の証拠とはしない |
| train-fit高、validation低／特定stratumだけ低 | `GENERALIZATION_GAP_OBSERVED`。修正方法は未確定 |
| train-fit／validationとも上がらない | `FIT_FAILURE_UNRESOLVED`。表現・容量が原因と断定しない |
| 教師・loss・index等の不一致を再現 | `CONTRACT_OR_IMPLEMENTATION_FAILURE`。step修正と混ぜずSTOP |
| 6000でも未達 | `BUDGET_LADDER_EXHAUSTED`。さらに延長せずSTOP |

複数ラベルを操作別に付けてよい。一つが延長で改善しても残りも同原因としない。one-seed診断であり、他seedの成功・失敗を予測事実として書かない。

---

## 6. 段階D — 候補固定とbundle準備

### D1. 選択規則

各操作について、固定1024例の `budget_validation` で **sequence EM>=0.95** を満たす最小の観測stepを選ぶ。

```text
for operation in four_targets:
    selected_step = first step in [1000, 2000, 4000, 6000]
                    with validation_sequence_EM >= 0.95
    if no such step:
        selected_step = null
```

非単調な曲線を隠さない。最初の合格stepより後で性能が下がった場合も全checkpointの結果を残す。上位EMを求めるgrid searchに変更しない。選択checkpointそのものを使い、選んだstepsで再学習して別の重みにしない。

全4操作に選択可能な候補がなければ、成功した候補も含め診断artifactとして保存し、**段階Eの正式RG3再判定を開始しない**。結果は `VALIDATION_TARGET_NOT_MET`。残る3〜1操作を親の失敗値や例外で埋めて完全復旧としない。

### D2. 予算改訂の範囲

`selected_recipe.json` に操作別の採用総step、1000-step対照、選択理由、Core依存、optimizer／データrecipeを固定する。ALTERNATING_NEGATEとINCREMENT_MODは既定1000-step recipeのまま。

REC-003の旧budget manifestとprotocolは変更せず、**新しいrecipe revisionと旧hashへの参照**を作る。将来REC-005で使う候補はこのoperation別step表であり、各seedの最終queryを見て予算を選び直す方式ではない。

### D3. Child bundle

全4候補を、同じCore・同じphysical ID・同じschemaの新しいchild bankへ組み込む。元bankの上書きや新しいprimitive IDの追加はしない。変化した4primitiveだけversionとtraining receiptを更新する。

Core、残り12primitive、router／keys、ArgumentScorer等の数値・formulaは不変。評価中の再較正は禁止。もし新bankに対してrouter/scorerを再学習しなければ実contractを満たせないなら、そこで `DEPENDENCY_REQUALIFICATION_REQUIRED` として停止する。Oracle実行が良くなったことをrouting較正の根拠に代用しない。

新bundle ID、execution signature、bank hash、必要な依存参照を再計算する。変更bankに依存するcache／recipe certificate／reference certificateは無効化する。旧資格を丸ごとcopyしない。hashだけ再計算して互換性を証明したことにもしない。

既存ModelBundle APIのstaging／diagnostic scopeを使い、**未検証のchildをnominalとして公開しない**。親のqualification原本は変更しない。qualification証拠を新artifactとして作り、現contractが求める内容hashと証拠の分離を守る。

---

## 7. 段階E — 独立queryでRG3を再判定

### E1. 再判定の入口

全4候補の重み・選択recipe・child manifest・データhashを固定し、runtimeからtrainer／builderを呼べない状態にしてから、`recheck_query` を評価する。

旧RG3は `FAIL` のまま。新しいrun IDの **`RG3_RECHECK_PASS / FAIL / UNRESOLVED / NOT_EXECUTED`** として記録する。独立queryで落ちた後に、保存済み6000-stepへ切り替えたり、queryを生成し直したりしない。

### E2. 全16操作の絶対値と凍結対照

- 全16操作のCorrect EM、token accuracy、loss、適用可能なcausal controlsを各1024 query例で測る。
- **非SHIFT15操作の各EM>=0.95** を要求する。平均だけで判定しない。
- SHIFTも全表に出し、今回の1.0という履歴と新測定を区別する。既存SHIFT限定例外は[S3]どおりで、今回拡張しない。
- 親とchildを同じqueryで評価し、4対象の差と12保護対象の不変性を表示する。4対象のparent失敗を「回帰なし」の基準に使わない。
- 保護対象12個はstate／formula不変に加え、同一入力上の予測一致を要求する。新queryで保護対象が0.95未満だった場合も隠さず、凍結が正しいことと復旧floorを満たすことを別判定する。
- Router top-1/top-k、scorerについては既存の非学習diagnosticを別欄に出す。L3/L4の未解決Gateをこのタスクで修理・免除しない。

1024例におけるEM floorは元のpoint-estimate契約。今回だけ信頼区間下限をfloorへ置換したり、token accuracyへ置換したりしない。`REF_ADEQUATE`やnominal capabilityには別の既定reference契約が必要で、契約未確定は未確定のまま。

### E3. Fresh-process検証

別process・別working directoryで、新manifestを明示指定してloadする。旧global cache探索、`get_or_build`、optimizer構築、router較正、欠落keyのrandom fallbackを動的ガードで禁止する。static文字列scanだけを実行なしの証明にしない。

同じsample-IDに対して、全16操作の離散予測が保存前と一致することを確認する。Core／bank／全componentのhash、execution signature、資格scopeも確認する。device/dtypeや既定の許容差は実行前のprotocolどおりとし、FAIL後に許容差を広げない。

診断scopeのload成功と、復旧floor合格と、nominal認証を別欄に記録する。未認証のfull scope要求を拒否するnegative testも維持する。

### E4. RG3再判定の必要条件

1. 親はseed10の固定Coreであり、全16の学習／復元根拠と依存が確認できる。
2. 対象4以外の学習・更新・formula変更がなく、保護対象12の同一入力予測が変わらない。
3. 新queryで非SHIFT15操作が各0.95以上。SHIFTは旧契約の範囲で状態を表示する。
4. 完全なload・fresh-process予測一致・旧cache不変・不正trainingなし。
5. query／referenceから候補選択への漏洩がなく、全missing／UNRESOLVEDを表示している。
6. capabilityの過大認証がなく、旧未達bundleと新結果を区別している。

いずれか不足ならPASSにしない。成功しても、**seed10の復旧pilotを改訂レシピで再確認した結果**であって、5モデルcohort・unseen family・hard-negative routing・APC全体の合格ではない。

---

## 8. 完了状態・停止規則・引き継ぎ

### 8.1 状態を三つに分けて報告する

```text
implementation_status: COMPLETE / PARTIAL
calibration_status: SELECTED_ALL / VALIDATION_TARGET_NOT_MET / BLOCKED / RESOURCE_BLOCKED
rg3_recheck: PASS / FAIL / UNRESOLVED / NOT_EXECUTED
```

コードと測定が完成しても科学性能はFAILでよい。FAILを消すために追加探索しない。

### 8.2 再開規則

| 結果 | 次の扱い |
|---|---|
| 4操作のvalidation未達 | 予算比較完了、RG3再判定未実施。REC-005はblocked |
| 新query floor未達 | RG3再判定FAIL。query後の予算変更不可。REC-005はblocked |
| dependency／scope／再現性に未解決 | 性能が高くてもRG3をPASSにしない |
| 全RG3再判定条件PASS | 改訂recipeを引き継ぎ、REC-005を実行可能とする。実行は別指示 |

REC-005ではseed10の合格childを使用できるが、残りseeds11〜14には**同じ固定したoperation別予算**を適用する。今回seed10で復元成功したSHIFTが他seedでも成功するとはしない。失敗seedの置換や、追加task内での先行5-model学習は禁止。

### 8.3 未解決結果からの提案

学習延長が効かなかった場合、次の提案は最も明確な未解決機構に一つだけ絞る。例：MIRROR_HALVESのtrain-fit失敗、SWAP_ENDSの変換位置誤り。新しいCore・大型operator・別lossを実装せず、根拠のある新タスク案として止める。

原レシピの1000-step未達、今回のvalidationによる選択、独立queryによる再確認を区別する。原runを完全再現できなかった経路はその旨を明記する。

---

## 9. 実装先とCLI

既存module／dispatcherを優先する。新しいpackageや全phaseを動かすwrapperは作らない。

```text
docs/CODEX_TASKS_PHASE_B_B2_INCREMENTAL_BUDGET_CALIBRATION.md
src/apc/evaluation/model_bundle_recovery.py                 # 既存。薄い追加入口
src/apc/evaluation/incremental_budget_calibration.py         # 分離が必要な場合のみ新設
scripts/run_phase_b_b2_model_bundle_recovery.py              # REC-004A明示dispatch
scripts/rec004_fresh_process_check.py                       # 既存。manifest入力再利用
configs/phase_b_b2_model_bundle_recovery_rec004a.yaml
tests/test_incremental_budget_calibration.py
tests/test_recovery_qualification_scope.py                  # 既存testsへの追加でも可
runs/phase_b_b2_model_bundle_recovery/rec004a/<run_id>/
```

上記新module／config名は提案。現APIを調べて最小差分にする。新規configには未確認のparent pathやoptimizer既定値を埋め込まず、段階Aで確認した値を設定する。

実装後の推奨CLI形（まだ存在するとの主張ではない）：

```bash
python scripts/run_phase_b_b2_model_bundle_recovery.py --task B-C005REC-004A --config configs/phase_b_b2_model_bundle_recovery_rec004a.yaml
```

run-dir指定は既存CLI契約に従う。旧REC-004／run_001を再利用せず、既存出力があれば上書きしない。resumeは完全state・protocol hash一致の対象runに限る。`--all`、latest自動選択、自動REC-005を追加しない。

root `AGENTS.md`、復旧task/plan、`docs/DECISIONS.md`／`docs/DECISIONS_PHASE_B.md`に追加branch・結果を記録する。旧RG3や旧文書の予算を書き換えず、改訂manifestへのリンクを追記する。次ADR番号はrepoで確認する。

---

## 10. 必須テストと受入物

### 10.1 CPU focused tests

1. 対象4操作だけのparameterをoptimizerへ渡し、Core・他12・router/scorerに更新がない。
2. 複数checkpointが同一trajectory上の累計stepであり、6000を超えない。
3. exact resumeの欠落状態を検出し、weight-only restartをexactと表示しない。
4. 分割resume／連続実行がCPU tiny fixtureで一致し、途中evalがRNG／更新順を変えない。
5. validatorは最小合格stepを選び、未達はnull。recheck-queryを選択APIが受け取れない。
6. teacherと評価の仕様一致、合法境界、changed位置ゼロ→null、token accuracyとEMの分離。
7. 新bundleは4primitive以外不変、Core不一致・旧certificate・過大capabilityを検出する。
8. 評価／fresh-loadにbuilder・optimizer・較正が入った場合、dynamic guardが失敗させる。
9. 原artifact不変、出力namespace上書き拒否、sealed access禁止、operationごとの予算revision。
10. final-query失敗後に別checkpointを自動採用しない。欠落・未実行をPASSにしない。

学習予算を延長しなければ未達になることをCPU fixtureで作っても、実primitiveの成功証拠とは呼ばない。GPU性能はmilestone runで別に測る。

### 10.2 Repo標準検証

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

実repoに新しいcanonical commandがあればそれに従う。focused tests→標準検証の順で実行し、既存suiteの過去PASSを今回の結果に代入しない。同時実行の他ジョブは無断で停止しない。未実施／resource不足は明示する。

### 10.3 成果物

```text
config.yaml / protocol.json / system.json / summary.json / metrics.jsonl
parent_audit.json / resume_audit.json / operation_contract_audit.json
existing_learning_curve_audit.json / qualification_scope_audit.json
budget_protocol.json / data_manifest.json / learning_curve.jsonl
error_breakdown.json / checkpoints/ / training_states/
selected_recipe.json / recipe_revision.json / bundle_lineage.json
all_primitive_execution.json / protected_regression.json
rg3_recheck.json / fresh_process_report.json / qualification.json
side_effect_audit.json / report.md
```

未到達段階の成果物は捏造せず、summaryに `NOT_EXECUTED` と依存理由を記録する。runごとのsource commit、device、framework version、精度mode、時間、VRAM、training updates/examplesを保存する。

plotは学習／validation曲線、操作別length／位置別誤りの必要最小限でよい。主張はplotだけに依存させず、機械可読の分子／分母を残す。

### 10.4 完了報告の書式

```text
Task: B-C005REC-004A
Implementation / calibration / RG3 recheck status:
Source report facts / protocol additions / deviations:
Parent bundle / Core / model seed:
Continuation mode per operation / exact-replay limitations:
Operation | original EM | paired 1000 | 2000 | 4000 | 6000 | selected step
Operation | selected validation EM | independent query EM | train-fit | failure label
Protected 12: hashes / same-input predictions / independent query floors:
Loader requested scope / certificate scope / unresolved qualifications:
Training updates and examples / optimizer/schedule unchanged:
Fresh-process result / no implicit training / original cache unchanged:
Tests / commands / artifacts / new ADR:
REC-005 eligible?: yes/no, not executed
R3-011/012 / B-C006 / Task Inference: blocked
```

**STOP：本タスク終了後、REC-005または新しい修正を自動開始しない。**
