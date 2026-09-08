# Recovery Acceptance Plan — Model Bundle Recovery

**版：1.0 / 2026-09-07。RG0～RG6は今回新設する復旧用受入条件であり、R3のG0～G5とは別。**

## 1. 引き継ぐものと変更しないもの

[S1]はR3-003／005の契約としてtau=0.95、最大5候補、looks=[32,64,128,256,512]、alpha=0.01を報告する。細部は現repoの`statistical_contract.json`と[旧実験計画snapshot](research/evidence/R3_EXPERIMENT_PLAN_SUPPLIED.md)を照合する。復旧の都合でlook、alpha配分、UNCERTAIN処理を変えない。

SHIFTのR3-009目標はmean query EM>=0.99、全model REF_ADEQUATE、既存性能低下<=1 pp、fresh-runtime adaptation=0等。R3-010のG4にはroutingだけでなくunconditional EM、coverage、安全性、実K/C/N/Rがある。**RGの成功ではこれらを免除しない。**

復旧中の使用範囲はdevelopmentとそのvalidation。旧sealed0～4／20～24を調整へ使わず、未評価sealed30～34の出力を測らない。relation transferの不成立は別blockとして残す。

## 2. 事前固定する復旧protocol

### 2.1 Model／data単位

- pilot model seed=10、cohort model seeds=[10,11,12,13,14]。5個の独立したCore／bank等の学習runを原則とする。
- 同じCoreのcopy、data seedの繰返し、N別evaluationを独立modelに数えない。既存artifactを復元する場合も実training identityを確認する。
- development／validation区分は現repoで照合し、data seedからmodelを生成する暗黙処理を除く。
- generatorはR3-001で実装したversion付き経路を使用する。old runのexact replayと新v2 reconstructionを別ラベルにする。

### 2.2 新設する復旧用の機能floor

以下は**今回提案する技術的baseline受入条件**。旧Gateを改変するものではなく、結果を見る前にREC-003で固定する。

1. 全16個の実registry entriesについて、依存関係・学習／復元根拠・完全loadを要求する。missing、untrained、shapeだけ一致は不合格。
2. direct/oracle-call queryは各model×operationについて原則1,024例、合法argument／length分布を事前固定する。token accuracyは補助で、primaryはsequence EM。
3. **SHIFT以外の15操作は、各model×operationでquery EM>=0.95**を要求する。macro平均だけで0点operationを隠さない。古いsourceでparameter-free等の別contractがある場合は測定前に整合させ、適用scopeを明示する。
4. SHIFTは既知の修正対象として、trained・依存一致・出力可能なbaselineの機能不足を`COHERENT_LIMITED`で保持できる。この例外はSHIFTだけに事前限定する。全16表から除外せず、EM／loss／reference状態をそのまま記録する。未学習baselineや無効argumentへのfallbackは許容しない。
5. 上記query floorはreference adequacyの統計的認定ではない。通常実行のcapability認定にはR3のreference契約を別途適用する。query EMが0.95を超えただけで`REF_ADEQUATE`と書かない。
6. 既定のreferenceは旧計画の4,096例を起点に現contractへ合わせ、対象call、M、error allocation、distributionを事前固定する。reference未確定を例数の都合で強制二分しない。

SHIFTの最終修正目標を下げるものではない。baseline floorを満たしてもverifierが十分な頻度で受理できない場合はavailability側の未達として残す。

### 2.3 予算・実行可能性

学習step、optimizer、容量は既存recipeから取得し、推測値で埋めない。取得不能ならREC-003を`RECIPE_UNAVAILABLE`で停止する。

single GPU 16 GBを前提とし、stage／candidateを逐次処理する。実行環境のversionはrepo宣言を維持する。milestone commandでのみheavy runを実施し、unit testsはCPU tiny fixtureにする。OOMや実行不足は`RESOURCE_BLOCKED`。batch等の変更が学習条件を変える場合は新run／protocol revisionとし、密かに変更しない。

## 3. RG0 — 保存・復元判断の完全性（REC-001）

PASSに必要なもの：

- 既存artifactの実在／欠落／hash／source commit／来歴不明を台帳化。
- 全16個とCore／decoder／router／scorerのdependency表。
- cache-miss／cache-hitの両経路と書込み・training副作用の監査。
- 同じfamilyでもversion／Coreが違うものを別に記録。
- 復元可能性と再構築が必要なclosureをmodel単位で分類。
- 既存artifact非変更。UNAVAILABLEを架空pathで補完していない。

復元できないという調査結果はRG0 FAILではない。必要な調査を行わずに「8操作を再学習」と決めた場合はFAIL。

## 4. RG1 — Bundle／loader contract（REC-002）

CPUの正常・異常fixtureで検査する。

| 条件 | 期待する結果 |
|---|---|
| 正しいfull bundle、許可scope、certificate一致 | load可能 |
| Core hash、argument schema、key-ID mappingの不一致 | 明示的失敗 |
| 全16中1個欠落、state_dictのmissing key | 明示的失敗。補完禁止 |
| trained情報なし、UNKNOWN dependency | nominal load拒否。監査modeのみ |
| formula-onlyの修正を旧manifestでload | execution signature不一致 |
| router/scorerの非互換pair | nominal load拒否 |
| 欠落をtriggerにbuilderを呼ぶ | testが失敗すること |
| seed10のpathをseed11とlabelし直す | 独立model検査に不合格 |
| staged checkpoint／途中write | publish・nominal load不可 |

hash／state／execution signatureの一致と、capability certificateの一致を別にテストする。既存API名をそのまま使うことより、契約を守ることを優先する。

## 5. RG2 — Complete Build Plan（REC-003）

必要条件：

- build planが実registryの全16個をcoverし、8+2+6のいずれにも明示的な学習または正当なrestore経路がある。
- Core変更がdownstream全体の再検証／無効化をtriggerする。
- 全step／capacity／recipe／data roleが固定され、未定のheavy stageがない。
- build cache keyが全upstream内容とrecipe／data／schemaを含む。
- dry-runでmissing stage、resume不一致、未学習fallbackを検出。
- K/C/N/R streamの初期membershipとN/R因果順が固定されている。
- 復旧floor・fixture・source exposureが学習／成績確認前に登録されている。

旧Coreの全面再学習を必須にはしない。既存の正しいCoreが使えるなら、そのCoreを親として必要なbank stageだけbuildする。joint recipeでCoreを更新した場合は全dependencyが新parentへ移ることを明示する。

## 6. RG3 — Pilot成立（REC-004）

全て満たすこと：

- model seed10を選び直していない。
- 全16個のcoverage／compatibilityが揃い、非SHIFT15個が§2.2のfloorを満たす。
- SHIFTはtrained状態と実EMを記録し、弱ければ`COHERENT_LIMITED`。
- 固定same sample IDでsave／fresh-load後に同じpredictionsを得る。数値許容差は実行前に固定し、parameter hash一致とGPU数値再現を区別。
- 別processのloader経由でtraining／global cache参照／random fallback=0。
- build前後の旧cache hashが一致し、旧runを更新していない。
- direct／routing／verifierのスコアを混ぜていない。

復元したbundleの機能確認と、cacheなしbuildを実行した事実は別欄。実行していないclean-build経路は未実測と表示する。

失敗原因が「学習済みだが機能floor未達」なら機能不足として止める。未学習ロードバグと同一視せず、無制限学習で補わない。

## 7. RG4 — 5モデルの復旧cohort（REC-005）

RG3の条件を全5モデルで満たす。planned=5、completed=5、distinct model=5が必要。全16×5の表を必須とし、成功seedだけの平均で判定しない。

SHIFTを除く全操作のfloorとartifact整合が満たされれば`RECOVERY_COHORT_READY`。SHIFTの未達、hard L3/L4、verifier coverageは後続研究結果として明記する。

すべてのcapabilityが十分と認証できた場合のみそのscopeを`NOMINAL_VALIDATED`にする。cohortを封印済みtest用modelと呼ばない。

## 8. RG5 — 注入と修正系譜（REC-006）

PASSに必要なもの：

- 明示bundleを入力したfrozen evaluationでbuilder／calibrator／variant selectorが呼ばれない。
- 修正prepのtrainable parameterと対象dataが固定され、その後のstate hashが保存される。
- router／scorer／formula／lambdaの組合せを識別できる。
- 変更差分が期待どおり。複合変更ならその旨を記録し、単一機構の改善とは言わない。
- 実sequential runnerがworking copyを受け取り、初期Core／bankを裏で置換しない。
- Nの許可された更新と、不正なhidden rebuildをテストで区別する。
- failed SHIFTのrollback先が整合したtrained親であり、candidateの数値とinstalled数値が区別される。

事前登録した未較正のcounterfactualをdiagnostic modeで評価することは、schema／dependency不一致の見逃しとは区別する。その対照はnominal資格を持たず、誤ったpairを正常構成として宣伝しない。

局所修正の数値FAILはRG5のengineering PASSと併存できる。そのFAILを出した構成も評価対象として保存する。一方、pairの不整合やmissing artifactはRG5 FAILであり、REC-007の本評価を開始しない。

## 9. RG6 — 比較実施と研究G4の別判定（REC-007）

### 9.1 RG6が要求するもの

- C0～C5の固定入力・parent lineageと絶対性能の表。
- nominal、same-identity stress、COUNT↔BIND、SELECT/BIND、SHIFTの各適用scope。
- K/C/N/Rを実runtimeで実行した証拠。nominal-liteで代替しない。
- reference未確定、未出力、N失敗後のR、missing rowsを集計に残す。
- fresh-process recurrence、workspace、bank changes、bounded replay更新を記録。
- 旧artifact不変、共有cache非参照、評価中の不正学習なし。

性能FAILでも完全に正しく測定できればRG6は`COMPARISON_EXECUTED`。未実施項目がある場合は`PARTIAL`で、完全実施と区別する。

### 9.2 旧G4の主要条件（要約。元契約を変更しない）

[旧実験計画](research/evidence/R3_EXPERIMENT_PLAN_SUPPLIED.md)の§7–9を正とする。

- N=128：L0–L2 full-call top-1>=0.98、L3>=0.95、L4>=0.90、top-5>=0.99。
- L4 family top-1>=0.98、各parameterized operation argument accuracy>=0.95。
- unconditional query EM>=0.95、execution coverage>=0.97。
- avoidable plastic<=0.02、adequate-solution nonreuse<=0.03、reference unresolved<=0.05。
- wrong-call／inadequate-call／same-identity unsafe acceptance各<=0.01。empty denominatorでPASSしない。
- legacy EM／routingのmean低下<=1 pp、worst-operation低下<=2 pp。absolute復旧floorも表示。
- Cの根拠なきexpansion=0、Rのadaptation／temporary／recommit=0、成功Nのcapability当たりpromotion=1、leak=0。
- SHIFT個別Gateはmean EM>=0.99、全model REF_ADEQUATE等を独立に判定する。

SELECT→BIND等の必要stratumがUNRESOLVEDなら全L3成功としない。streamへの新verifier接続に伴うcoverage不足も研究FAILとして扱う。

### 9.3 統計・サンプルの意味

各指標の分子／分母、per-model、per-operation、relation、argument、bank sizeを報告する。referenceを同じcall/version/distributionで再利用した場合は再利用回数を独立trialと数えない。

unsafe acceptance 0件は母集団率0の証明ではない。旧reference契約とinterval表示を維持し、incompatibleな旧SHIFTを新Coreへ移して安全性stressを人工的に作らない。

## 10. 復旧完了／研究再開の判定表（REC-008）

| 状況 | 復旧の扱い | 次に許可されないこと |
|---|---|---|
| 互換／学習coverage／nonSHIFT floorが未達 | `RECOVERY_BLOCKED` | 修正効果の本比較、sealed |
| RG0～RG6成立、局所repairやG4はFAIL | `RECOVERY_INFRASTRUCTURE_COMPLETE_RESEARCH_BLOCKED` | 新loss自動探索、R3-011 |
| G4再評価までPASS、G1不足が残る | `RECOVERY_COMPLETE_G4_RECHECK_PASS_RELATION_BLOCKED` | unseen-relation成功主張、R3-012 |
| 必須研究条件の解決が別途確認された | 別指示でR3-011を検討可能 | 本タスク内の自動封印・自動評価 |

どの経路でも旧R3-009／010のFAILは保存する。復旧用に作った新親モデルでの結果は、新run・新parent・新qualificationとして記録する。

## 11. 計算量と副作用

保存する最低項目：build wall-clock、training steps/examples、peak VRAM、resident/active/temporary parameters、stage別primitive forward calls、support unique count、candidate-example evaluations、recipe count/depth、decision／final execution latency、runtime N適応コスト。

static matrixとsequential adaptationの費用を別にする。CPUの検定処理時間だけをend-to-end latencyと呼ばない。効率改善の新研究はB-C006へ持ち越し、この復旧では測定範囲を正しく表示することを目的にする。
