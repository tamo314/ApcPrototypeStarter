# Phase B 終端・アーカイブ台帳 — 旧再開計画

版: 2026-09-13 / ADR-0149。`CLOSED_ARCHIVED`。

## 1. 最終状態と権限境界

**`NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`** を正式な終端として固定する。
PHASE-B-CLOSEOUTの判定は **`PHASE_B_CLOSED_NEXT_RESEARCH_CHARTER_READY`**。
[最終証拠台帳](../../results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md)を主張分類の正本、
[ADR-0148の終了監査](../../results/PHASE_B_FINAL_FALSIFICATION_SUFFICIENCY_AUDIT.md)を保存する。

`candidate_selected=null`, `child_bundle=null`, `bundle_write=false`, `RG3=NOT_EXECUTED`,
`REC-005=BLOCKED`, `G1=STOP`, `G4=BLOCKED`, `G5=BLOCKED`。
今回のsealed-data/model-output access=0。sealed_v2評価は未実行・未開封。
歴史上のREC-004 RG3 FAIL、G4 development FAIL、B2 sealed re-gate FAILは変更しない。

2026-09-12のPhase B継続指示は本終端を越える研究実行権限ではない。
Phase Bの実験待ち行列は空。RG3、REC-005〜008、R3-011/012、B-C006以降は
upstream STOPによる非実行としてarchiveし、未完了backlogへ戻さない。
独立した[Phase C research charter](../../research/PHASE_C_RESEARCH_CHARTER.md)は
`TERMINATED_CURRENT_CHARTER`（ADR-0160、C-D001AA、2026-09-13）。承認されたことは一度もなく、
実験実行も一度もない。ADR-0150〜0159のpre-execution数学的レビューで完全にfalsified/retracted
され、ADR-0160がfalsification十分性を監査した上でcharterを正式終了した。詳細は
[Phase C termination evidence ledger](../../results/PHASE_C_TERMINATION_EVIDENCE_LEDGER.md)
および本ファイル末尾の第12節を参照。architecture未選択、最初の実験未定義のまま終了。
新しい研究課題には別の独立したcharterと承認が必要。

本ファイルはリンク互換性のため旧 `active/` pathへ保存するが、運用状態はarchived。
以下の完了契約・実行順・当時の継続許可は歴史記録であり、新たな実行許可ではない。

## 2. 文書の役割

| 文書 | 今後の役割 |
|---|---|
| 本書 | 終端・アーカイブ状態、および保存された過去契約・実行記録 |
| [Phase B親計画](PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md) | B1〜B6の研究目的・最終ゲートの原契約 |
| [Post-D2計画](PHASE_B_B2_POST_D2_REPAIR.md) | R3の技術仕様・G0〜G5の原契約 |
| [Recovery計画](PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md) | REC-001〜008の技術仕様・RG0〜RG6の原契約 |
| [監査報告](../../research/PHASE_B_BLOCKER_AUDIT_2026_09_12.md) | 2026-09-12時点の履歴・不具合の証拠。後続実行結果で書き換えない |
| [Decision index](../../DECISIONS.md) | 判断の索引。科学的解釈の変更は新ADRへ追記 |
| 過去REC-004A〜ACの契約・run | 完了済みの診断/介入の記録。自動再実行する待ち行列ではない |

ファイルの削除・移動や過去番号の振り直しは行わない。細分化されたタスクをさらに積み上げず、
現行architectureの実験待ち行列を再構成しない。

## 3. 終端時の成果と非実行の区別

| 項目 | 作業の状態 | 科学的/復旧判定 | 再利用する成果 |
|---|---|---|---|
| B-C001〜003 | 完了 | B1 PASS（明示TaskSpec条件） | 未見familyでのlifecycle |
| B-C004〜005、D/R1/R2/G、D2 | 完了・診断系列は終了 | B2/re-gate FAIL | ranking/argument/adequacyの分離 |
| R3-001/003/005 | 完了 | G0/G2/G3 PASS | 再現可能な生成、指標、有限look verifier |
| R3-002 | 調査完了 | G1 relation不足 | exposure台帳・不足の証拠 |
| R3-006〜008 | 過去親上で完了・archive | 新親への資格は未証明、再実行待ちではない | scoped key、SELECT式修正、BIND coverage |
| R3-009〜010 | 完了 | SHIFT/G4 FAIL | 安全な置換・統合ladder・不整合検出 |
| REC-001〜003 | 完了 | RG0〜RG2 PASS | immutable bundle、fail-closed load、16操作build |
| REC-004 | 実施済み | RG3 FAIL | seed10親とfresh-load検証 |
| REC-004A〜AC / ENV1 | 完了した診断・介入 | MIRROR候補未採用 / RG3未再検証 | 3操作validation修正、位置bias、安定value経路、負の対照 |
| 再開: AC metric_v2 | 訂正・再分析・分割全件検証完了（§9） | 計測/再現PASS、性能FAILを保持 | 正しいkey指標、保存語彙の固定 |
| 再開: score-scale precheck | 完了 | 固定4条件すべてFAIL_STOP | 全体的な拡縮だけでは当該endpointを回復できなかった |
| 再開: REC-004AE | 完了（§7） | `INSUFFICIENT_EVIDENCE_STOP` | routing失敗は局在したが、QK対position biasの単一修復標的は未分離 |
| 再開: REC-004AF | 完了（§7A） | `INSUFFICIENT_EVIDENCE_STOP` | QK endpoint 置換は一部token recoveryを示したが、routing/marginの非退化差とposition側の可変対照がない |
| 再開: REC-004AG | 完了（§7B） | `POSITION_ROUTING_TARGET_NOT_SUPPORTED` | 座標空間transportはEM=0へ崩壊し、位置bias単独標的を反証 |
| 再開: REC-004AH | 完了（§7C） | `SCORE_DECOMPOSITION_IDENTIFIABILITY_STOP` | 構成要素識別性行列の評価で全要素が条件(d)不成立、単一修復標的の探索を停止 |
| 再開: REC-004AI | 完了（§7D） | `MINIMAL_ROUTING_CONTRACT_IDENTIFIED` | MIRROR必要条件から探索なしに単一最小routing contract（CD-DPCA）を導出 |
| 再開: REC-004AJ | 完了（§7E） | `MINIMAL_ROUTING_CONTRACT_IMPLEMENTED_AND_VALIDATED` | CD-DPCAコード実装、未学習構造7不変条件PASS、有限表現境界の確定 |
| 再開: REC-004AK | 完了（§7F） | `CD_DPCA_SERIALIZATION_AND_FRESH_LOAD_VALIDATED` | CD-DPCA/Bank厳格シリアライズ、fresh-load完全等価性、10種負例fail-closed検証 |
| 再開: REC-004AL | 完了（§7G） | `PILOT_TERMINAL_VIABILITY_NOT_MET` | 6,000 updates単一init学習パイロット。validation EM=0.793945 < 0.95 未達によりfail-closed停止。candidate/bundle未作成 |
| 再開: REC-004AN | 完了（§7H） | `LENGTH10_LOCAL_OPTIMIZATION_FAILURE_IDENTIFIED` | 13 checkpoint全軌道再評価、位置局在、transplant、勾配衝突、露出監査。失点の89.4%が位置4の偽アトラクタ固着（key 7）に局在。追加学習0 |
| 再開: REC-004AO | 完了（§7I） | `LOCAL_LOSS_GRADIENT_MISALIGNMENT_IDENTIFIED` | 13 checkpoint勾配到達性診断。ルーティング勾配ノルムは十分（0.2707）だが正解margin予測変化は84.6%で非正（端末-0.6976）かつkey0飽和飢餓。追加学習0 |
| 再開: REC-004AP | 完了（§7J） | `TOKEN_ALIASING_DILUTION_AND_LATE_SATURATION_IDENTIFIED` | 13 checkpointトークン識別性層別化診断。Strata A/B（86.9%）は正解margin改善勾配を持つが、Stratum C（13.1%）の巨大破壊勾配（12.4倍優勢）がpooled勾配を希釈・反転。追加学習0 |
| 再開: REC-004AQ | 完了（§7K） | `WARM_START_PILOT_VIABILITY_MET` | 初期500step非復元抽出warm-start単一レシピ因果パイロット。決定EM=0.9854、長さ10 EM=1.0000、位置4偽アトラクタ解消。REC-004AM検討認可 |
| 再開: REC-004AM | 完了（§7L） | `MULTI_INIT_VIABILITY_NOT_MET` | warm-start固定レシピ5独立初期化（I01..I05）再現性検証。合格1/5（I01のみ合格）、適格性規則（5/5）未達によりfail-closed停止。candidate/bundle未作成 |
| 再開: REC-004AR | 完了（§7M） | `I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED` | 5初期化matched-baseline因果複製。同時改善2/5（I01, I05）、I02/I03は負干渉。普遍的優位性は反証されI01特異的・混合効果を同定。REC-004AM FAIL保持 |
| 再開: REC-004AS | 完了（§7N） | `INITIALIZATION_DATA_INTERACTION_IDENTIFIED` | 既存130 checkpoint state・全40 routing cell/初期化・step-1固定streamのno-update監査。step-0幾何単独/early-data単独は不十分、初期化×data interactionを確認。generic initialization-only repairは未同定、pilot禁止 |
| 再開: REC-004AT | 完了（§7O） | `NO_SINGLE_OPTIMIZATION_FORMULATION_IDENTIFIED_STOP` | AO/AP/AM/AR/AS既存証跡のみのoracle-free standard-CE式識別性監査。alias collisionは同一CE observableに対し反対routing方向を要求し、単一式の一意導出に失敗。CE-only optimization-repairをSTOP |
| REC-005〜008 | archive・upstream STOPによる非実行 | RG3前提不成立 | cohort・runtime注入の原契約を保存 |
| R3-011〜012 | archive・upstream STOPによる非実行 | G1 STOP / G4 BLOCKED | 封印・B2_PROTOCOL_V2原契約を保存 |
| B-C006〜014 | archive・upstream STOPによる非実行 | B2/G5前提不成立 | B3〜B6の原計画を保存 |

**未証明の範囲（backlogではない）は三つ:** (a) MIRRORを含む整合bundleの通常実行性能、
(b) 同一親での検索/引数/SHIFT/実K/C/N/R統合、(c) 独立relationの不足とholdoutへの学習露出。
I01〜I05は同一Core上の初期化であり、5個の独立modelではない。

## 4. 歴史記録: 実行順と出口条件

以下は終了前の依存関係の保存である。全行archive済みで、実行待ちを表さない。
独立とされたG1再設計もADR-0147でSTOPし、代替family追加は未認可のまま終了した。

| 順序 | 作業 | 開始条件 | 完了後に進める範囲 |
|---|---|---|---|
| 1 | REC-004AC metric_v2訂正・再分析 (§5) | ADR-0126の不具合確認済み | 原因に基づく有限なMIRROR修復の事前登録 |
| 2 | MIRRORの一仮説修復 | 計測/入力/親の再現PASS、仮説と予算固定 | 固定recipeによる全init安定性検証 |
| 3 | MIRROR採用・REC-004 RG3再検証 | 必要な全initが固定条件で合格、I01選択規則維持 | seed10全16coverage、非SHIFT15各EM≥0.95、fresh-load |
| 4 | REC-005 | RG3 PASS | 独立Core/bank seeds10〜14で全16×5表、RG4 |
| 5 | REC-006→007 | RG4、続いてRG5 | 明示bundle注入・既定修正prep・実K/C/N/R。RG6とG4を別判定 |
| 独立 | R3-002 G1再設計 | relation単位と露出契約を事前固定 | 独立clean単位validation≥2/sealed≥2、heldout-key勾配遮断の証拠 |
| 6 | REC-008→R3-011→012 | 必須復旧/研究条件が全て成立 | 引継ぎ→事前封印→一回のG5。失敗ならSTOP |
| 7 | B-C006→007 | G5 PASS | 計算量測定・必要最小限の有界探索、B3 gate |
| 8 | B-C008〜014 | B1〜B3、以降各task gate PASS | Task Inference段階、統合open-world、最終判定 |

RG3のSHIFT例外、SHIFT研究目標mean EM≥0.99・全model REF_ADEQUATE、
G4のcoverage/unsafe reuse/旧機能回帰等は元の受入計画を維持する。
G1は新seedや同じrelationの逆方向では増やせない。新しい独立relationと、全候補CEから
holdout keyへの更新を遮断する仕組みの**両方**が必要で、MIRROR改善と混ぜない。
REC-005/006ではbuild時の語彙契約とtask-tokenの操作順序も同じ親に固定する。
今回のcontent-only runtimeの語彙固定だけで、学習済みrouterのtoken互換性を認定しない。

失敗時は判定・artifact・ADR・次に必要な最小証拠を記録して依存作業を止める。
未解決の仮説を理由に連番診断を自動生成しない。完了済みタスクを再び「未着手」に戻さない。

## 5. 実行契約: REC-004AC metric_v2

**状態: 訂正・再分析PASS、分割全件検証完了（§9）。新学習0、候補選択0、sealed0。**
目的は正解key、head recall、score規模、parameter会計の訂正と、保存endpointの同入力再分析。
既存J0/O1 forward・重み・学習scheduleを変更しない。ACの通常実行FAILは独立に保持する。

- 変更対象: AC評価集計、共有の評価専用位置指標helper、regression tests、artifact-only再分析runner。
- 正解key: 既存 `mirror_halves_position_map(n)` から導出。operationの独立 `apply` をtest oracleにする。
- recall: valid example×head単位のtop-k inclusion。tieは「correct score以上のwrong key数+1」で保守的に扱う。
- margin/probability: 正しい位置についてhead別に算出。旧mean-head-rankはrank記述統計として保持し、recallと混同しない。
- score norm: residualをbaseと同じhead shapeへ展開。padding除外、行中心化後のFrobenius比をprimaryにする。未中心化の同shape比も別欄に保存。
- parameter会計: target resident / 実行active / freeze-restored / update-eligibleを分離。legacy比とZ比の追加数を分離。
- provenance: AC run_001、I03@7500、historical@8000、Z/AA/AB/W@8000の既存weightsのみ。missing/不一致でSTOP。optimizerを構築/stepしない。
- data: AC保存manifestのcontinuity4、normal validation1024、length10 confirmation512を同じgeneratorで復元し、全manifestを照合。新評価split・RG3 queryは作らない。
- endpoint: AC@8000、historical/W/Z/AA/AB@8000を同入力で比較。AC@7500はlegacy共通部＋Keyコピー＋保存された規定zero residual初期化から再構成し、元parity条件を検証する。
- 訂正前後の比較: 同じforward tensorから旧式/new式を再集計し、予測とEMが同一であることを確認する。既存JSONのEM差は0を要求。
- 元実行とのFP32差: score/logit最大差1e-4、記述集計差1e-4を目安とし、超えた場合は記録して再現STOP。新しい性能値への差替えで回避しない。
- 予算: 既存endpointのみのforward。CPUテストはtiny fixture。single GPU16GB以内。旧run・共有cacheは前後hashで非変更確認。
- 出力: `runs/phase_b_restart/rec004ac_metric_v2/<run_id>/`。config/protocol/system/source hashes、dataset manifest、訂正表、EM/予測parity、freeze/side effects、cost、summary/report。
- チェック: focused pytest、全pytest、ruff、mypy。指定WSLが使えなければ原因を記録し、Python3.12・依存制約を満たす環境を明示する。

検証中に判明した共通障害も、この再開作業内で修正する。未見familyモジュールのimportで
操作登録数が18→23へ増えると、親Coreを44語彙ではなく49語彙で再構築していた。
strict loaderが検証したvocabulary stateを返し、その保存済みtoken schemaを明示注入する。
語彙を拡張して既存重みを加工することはしない。旧Phase Aの故障注入テストは、39語彙の
保存Coreのロード失敗から暗黙に再学習していたため、小さな整合fixture＋学習禁止へ変更する。
これらは研究結果の変更ではなく、再現・検証を成立させるための障害修復である。

成功条件は計測と再現のPASSであり、研究性能PASSではない。訂正後の正解marginとresidual作用に基づき、
score scale仮説を検証可能か判断する。未確定ならその状態を出口として記録する。

## 6. 次の介入: MIRROR score-scale precheck（事前登録）

**状態: 完了・FAIL_STOP（ADR-0129）。並行していた全件検証も完了（§9）。**
2026-09-12の実行順の調整: 過去の大型データ台帳を再構築する全件テストを待つ間に、
保存モデルを変更しない本precheckを進める。`run_003`の全manifest/予測/重み再現と、
実際に失敗していた全モジュール収集時の親ロード・schema回帰8件はPASS済み。
実験条件・予算・科学的基準は変えず、再開作業の完了判断には全件テストの結果も必ず含める。
新しいREC-004連番を作らず、順序2の修復案に対する最小の事前検証として管理する。

訂正後、正解keyのmarginは約−30.84、residualのmargin寄与は約−0.00029。
したがって「小さいresidualを増幅すれば直る」とはまだ言えない。学習を追加する前に、
**既存endpointのscoreの全体的な縮小/増幅だけで通常実行を回復できるか**を検証する。
学習中のscale変更が効くか、別のrankや非線形表現が学習可能かは、この検証の対象外。

- endpoint/データ: metric_v2で再現したAC I03@8000と同じ7組の開発入力のみ。
  `examples.json`とmanifestをhash照合し、各dataset digestを再照合。新生成・sealed参照なし。
- 固定4条件: `S = α S_base + β ΔS` の `(α,β)=(1,1),(1/8,1),(1,1000),(1/8,1000)`。
  基準、base温度変更、residual拡大、両方の2×2。1/8は約−30のmarginを数単位へ弱め、
  1000は訂正済みresidual/base比約0.00039を比較可能な桁へ上げる固定値。
  結果を見て係数や条件を追加しない。
- 全parameter/Core/他15 primitiveを凍結。optimizer構築/更新0、候補選択0、bundle書込み0。
  Q/K/V、値経路、readout、入力、paddingは同一。score係数は入力に依存しない。
  正解mapはO1と評価指標に限定し、J0には渡さない。
- 最初に `(1,1)` と既存forwardのlogit差≤1e-4・離散予測差0を全入力で確認。
  4条件でO1予測一致を要求。全体の重みhashと過去artifact hashが不変でなければ即STOP。
- 出口条件: 通常1024例EM≥0.95、length10確認512例J0 EM≥0.95、
  全length10 splitのO1 EM/位置4正解率≥0.95を**同時に**満たす条件があるかを判定。
  親RG3の通常floorを緩和しない。PASSでも採用/RG3の代用にはしない。
- 予算: 4条件×既存7datasetのforwardのみ。single GPU16GB以内。最良係数の追加探索なし。
  出力は `runs/phase_b_restart/mirror_score_scale_precheck/<run_id>/` に新規保存。
- 全条件FAILなら、このendpointの固定global scale修復をSTOPとし、ADRと残条件を記録。
  500step学習や全init展開を自動追加しない。これは学習可能性全般の否定ではない。

採用・全init展開・RG3の条件を満たすまでREC-005等に進まない。

実行結果: baseline通常EM=0.7646484375 / length10=0.080078125。
温度変更は0 / 0、residual増幅は0.0419921875 / 0、両方は0 / 0。
O1の最小EMは全条件0.998046875で予測も不変。全重み・過去source hashは不変。
基準との差は拡縮条件で悪化しており、この4条件の修復試行は終了する。
別係数や学習時のscale変更の可能性は未検証であり、一般的な不可能性とは解釈しない。

## 7. 実行契約: B-C005REC-004AE — 固定checkpointのMIRROR失敗モード局在化

**状態: 完了、`INSUFFICIENT_EVIDENCE_STOP`（ADR-0130）。これは診断だけであり、修復学習・candidate選択・RG3・sealed評価を許可しない。**
目的は、訂正済みAC I03@8000の残存J0誤りを、出力位置ごとのscore routing、oracle下流復元、
score構成要素のいずれに帰属できるかを固定入力で判定し、重複しない次の単一修復介入を
選べるかを決めることにある。

- endpoint/data: `rec004ac_metric_v2/run_003` が再現したAC I03@8000と保存済み7 development dataset
  （normal validation 1024、length10 confirmation 512、continuity4、parity fixture）。manifest、
  `examples.json`、checkpoint hashを実行前後に照合する。新生成、training/validation再分割、sealed参照はない。
- 凍結: Core、bank、AC primitiveの全tensorをfreezeし、optimizerの構築をfail-closedで禁止する。
  更新数0、parameter追加0、selection0、bundle書込み0。親bankは実行しない。
- 固定診断: (1) baseline forwardの既存EM/予測parity、(2) 正しいMIRROR position mapによる
  oracle-attention対照、(3) 出力位置×headの正解key rank/margin/top-1、(4) `QK`、position bias、
  score residualそれぞれの「正解key minus baseline top-wrong key」寄与、(5) baseline scoreから
  position bias全体、またはQK全体を除いた二つの**ablation-only** forward。後二者は係数探索・
  候補修復ではなく、失敗の所在を判定する固定反実仮想である。baseline/ablationのvalue/readoutは同一。
- failure mode: 各output tokenを `DIRECT_CORRECT`、`ROUTING_ERROR_ORACLE_RECOVERS`、
  `DOWNSTREAM_ERROR_PERSISTS_UNDER_ORACLE` に排他的分類する。oracleは評価対照だけであり、
  runtime入力・修復recipeへ埋め込まない。全位置でoracle回復し、正解keyの順位が系統的に低く、
  position-bias除去が同じroutingを悪化させずQK除去だけが悪化する、というような反証可能な交差表を保存する。
- 判定: `ROUTING_LOCALIZED` は、length10 confirmationおよびnormal内length10でoracle下流errorが
  1%以下、残存direct token errorの95%以上がoracleで回復し、正解keyのrank/marginとcomponent寄与が
  同じ出力位置群で再現し、かつ二つのablationがQK競合かposition routing不足かを一意に分ける場合のみ。
  それ以外は `INSUFFICIENT_EVIDENCE_STOP`。通常J0 EMやO1 EMの0.95 floorをこの診断で変更しない。
- 重複確認: score-only continuation/score勾配（O/Q）、score scale、pre/post-V rank4 residual、
  hard CVOF freeze、trust-region、既存の加法position biasは既に棄却・停止済みである。局在が成立した場合の
  次候補はそれらの係数再探索や後付けresidualではなく、**開始時からQK content scoreとlearned position-only
  routing scoreを役割分離し、value pathを維持する一つのrecipe**に限る。これはREC-004Dの「加法bias」や
  REC-004Zのkey/value splitと同一ではなく、実行には別の明示契約・予算・all-init接続条件が必要であり、
  本task内では実装・学習しない。
- 出力: `runs/phase_b_restart/rec004ae/<run_id>/` にprotocol、input/source hashes、dataset manifest、
  position/failure/component/ablation tables、side-effect audit、summary/reportを新規保存する。失敗時もartifactを
  残し、ADRへ判断と次の開始条件を記録する。

実行結果: `run_004` はexecution PASS、source不変、optimizer更新0、candidate選択0、RG3/sealed未実行。
length10 confirmationとnormal validation内length10では、direct token errorのoracle回復率はともに1.0、
oracle下流error率は0.0で、残存誤りをrouting側へ局在した。出力位置ごとのQK/position-bias寄与の符号も
両splitで一致する。しかしQK除去とposition-bias除去はともにsequence EM=0.0となり、両者の差はnormalと
length10でいずれも0.0であった。したがって寄与の存在は示せても、どちらを単一の修復標的にすべきかを
一意に分離できない。既に棄却済みのscore-only/gradient、global scale、compact residual、hard freeze、
trust region、加法position biasを再試行する根拠も生じなかった。判定は `INSUFFICIENT_EVIDENCE_STOP` とし、
単一修復介入は選定しない。

次タスクの開始条件は、(1) 固定endpoint上でQK競合とposition routing不足を**非退化な**反実仮想で
識別する事前登録、(2) 同一のvalue/readoutとcorrect map非流入を監査する対照、(3) 既棄却仮説との非重複、
(4) その結果が一つのrecipe、全init検証、固定I01採用、RG3へ接続する明示規則、の全てである。
これらがない限り、学習、係数追加、RG3、封印評価を開始しない。

## 7A. 実行契約: B-C005REC-004AF — matched-endpoint QK/position 因果分離

**状態: 完了、`INSUFFICIENT_EVIDENCE_STOP`（ADR-0131）。評価専用であり、repair target、学習、candidate、RG3、REC-005、sealed 評価は開始しない。**
目的は、ADR-0130 で局在した length-10 routing/value-selection boundary を、保存済み AC I03@8000
endpoint における非退化な paired counterfactual で QK-content competition と position-routing
insufficiency のどちらか一方に分離できるか判定することである。

- endpoint/data/freeze: immutable AC I03@8000 checkpoint と metric_v2 `run_003` の保存済み
  `examples.json` と manifest-identical development strata のみを読む。Core、parent bank、target
  primitive、value path、FFN、readout を凍結し、optimizer 構築・更新、parameter 追加、bundle write、
  candidate selection、architecture/coefficient search、RG3、REC-005、sealed-data read は fail-closed
  で禁止する。実行前後に checkpoint/source/data hashes を照合する。
- score/replay: runtime score を `S = S_QK + S_position_bias + S_residual` と明示分解する。
  `S_QK` は保存 Q/K score、`S_position_bias` は保存 position-bias score、`S_residual` は保存値のまま
  両 counterfactual で不変とする。QK counterfactual は対象の position bias/residual を保持して
  `S_QK` だけを control endpoint の同一 score row で置換する。position counterfactual は対象の
  `S_QK`/residual を保持して `S_position_bias` だけを置換する。softmax 後の attention から、既存の
  固定 value/output/FFN/readout path を通す。correct position map、oracle attention、正解 key index は
  J0 runtime や control selection に渡さない。oracle attention は baseline failure-mode の
  evaluation-only 再確認に限る。
- preregistered matching: control pool は split 内の全保存例であり、baseline/介入結果・target・oracle
  でフィルタしない。relation=`MIRROR_HALVES`、sequence length、output position、入力 token 和 modulo 4
  だけを matching strata とする。各 endpoint identity は split 名、例 index、入力 token tuple の SHA-256
  で昇順固定し、同一 stratum の自分以外の次の4例を circular に control rank 0--3 とする。pool が5未満なら
  STOP。matching table は model forward 前に書出す。成功例を選別する規則、正解routingに基づく事後選択、
  intervention result による control 変更は一切しない。
- strata/measurement: primary strata は normal validation 内 length-10 と length-10 confirmation、
  supplemental として normal validation 全長も別表に残す。baseline、各 control rank の QK/position
  counterfactual について sequence EM、token accuracy、baseline direct-error token recovery、head×token
  top-1 correct-key routing、correct-key probability/margin、QK/position-bias の correct-key margin contribution、
  routing-change 例数、誤答→正答と正答→誤答の token/sequence 数を保存する。correct-key map はこの集計だけで
  使用する。baseline parity は REC-004AE/AC の保存値と一致を要求する。
- fixed selection rule: component T を選ぶには、4 control rank **すべて**で両 primary strata において、
  baseline 比 correct-key top-1 routing が +0.05 以上、correct-key margin が +0.25 以上、direct-error
  recovery が 0.05 以上であり、もう一方 U よりも top-1 と direct-error recovery が各 +0.05 以上、margin
  が +0.25 以上でなければならない。T=QK のみが満たせば `QK_CONTENT_TARGET_SUPPORTED`、T=position のみが
  満たせば `POSITION_ROUTING_TARGET_SUPPORTED` とする。双方、同程度、符号不一致、control identity 依存、
  parity/hash/情報境界 fail、又はこの差未達は `INSUFFICIENT_EVIDENCE_STOP` とし、repair intervention を選ばない。
- non-overlap/next connection: これは ADR-0111--0130 の global scaling、score-only continuation、gradient
  repair、compact residual relocation、hard CVOF freeze、trust-region、duplicate additive position-bias の
  再試行ではない。selection 成立時だけ、一つの対応 repair recipe を文書化する。その recipe は実装せず、
  single-init pilot → preregistered all-init validation → fixed I01 adoption rule → candidate freeze →
  full 15 non-SHIFT operation RG3 → independent 5-model/G4 接続の出口条件を記す。selection 不成立なら
  recipe 複数比較、係数探索、architecture search へ進まず STOP する。G1/G4 は解除しない。
- output: `runs/phase_b_restart/rec004af/run_005/` に protocol、source/checkpoint/data manifests と hashes、
  preregistered matching table、baseline/counterfactual metrics、oracle evaluation audit、freeze/side-effect audit、
  summary/report を新規保存する。過去 run は上書きしない。

実行結果: qualified `run_005` は baseline と checkpoint/state hashを再現し、optimizer update=0、parameter
addition=0、candidate/bundle write=0、RG3/REC-005/sealed=0 で完了した。normal validation 内 length-10 と
length-10 confirmation の oracle direct-error recovery はともに1.0、oracle-persistent error は0.0であり、
固定 value/FFN/readout path のまま routing/value-selection boundary を評価した。4個すべての固定 QK controls は、
normal length-10 のdirect-error token recovery 0.135965--0.149123、confirmation 0.128295--0.138840、
sequence EM 0.081340→0.105263--0.124402 と 0.080078→0.095703--0.101562 を示した。一方 top-1 correct-key
routing の差は -0.001555〜+0.000488、correct-key margin差は -0.002519〜+0.003447 に留まり、事前登録した
+0.05 / +0.25 の両閾値を満たさなかった。position-bias endpointは同一length/output-position内で保存値が
例間不変であり、4 controls全てで baseline と bitwise同一の出力・routing集計になった。

従って、QK endpoint を移すと一部の誤答tokenが回復するという値選択への因果的感度は分離できたが、QK
competition を次の単一repair targetとして支持するほどの correct-key routing/margin 改善は分離できない。
position-routing insufficiency はこのcontrol designでは反証も支持もできない。position endpointに例間変動が
ないため、ゼロ効果はpositionが不要という証拠ではない。control identityに依存しない有効な優位差も示せず、
判定は `INSUFFICIENT_EVIDENCE_STOP` とする。single repair recipeは文書化せず、新しいarchitecture search、
coefficient sweep、複数recipe比較、学習pilotを開始しない。`run_001`--`run_004` は baseline aggregate の
float precision/parity guardで安全に停止した未qualified artifactであり、いずれもsource不変・optimizer
非構築で保存した。`run_005`だけをqualified resultとして参照する。

## 7B. 実行契約: B-C005REC-004AG — 非退化位置routing transport因果診断

**状態: 完了、`POSITION_ROUTING_TARGET_NOT_SUPPORTED`（ADR-0132）。評価専用であり、repair target、学習、candidate、RG3、REC-005、sealed評価は開始しない。**
目的は、REC-004AFで判明した同一長さ内の位置bias例間不変性（分散0.0）に対し、アーキテクチャ座標に基づく長さ9から長さ10への非退化transport反実仮想を用いて、位置bias単独が修復標的として支持されるか判定することである。

- endpoint/data/freeze: immutable AC I03@8000 checkpoint、metric_v2 `run_003`入力、qualified REC-004AE/AF artifactのみ使用。全パラメータ凍結、optimizer禁止（`RuntimeError`）。
- 介入: 正解token・oracleを参照せず、正規化座標 $u(k, L) = k / (L - 1)$ で長さ9から長さ10へマッピング。Frobenius差 9.500572 の非退化介入を構成。
- 実行結果: `runs/phase_b_restart/rec004ag/run_001/`。normal validation長さ10およびconfirmationの双方でsequence EMは0.08→0.00へ崩壊、top-1 routingは0.45→0.30へ悪化（$\Delta = -0.15$ vs $+0.05$閾値）、marginは$-7.15$→$-7.54$へ悪化（$\Delta = -0.38$ vs $+0.25$閾値）。位置bias単独標的は反証され、`POSITION_ROUTING_TARGET_NOT_SUPPORTED` と判定。

## 7C. 実行契約: B-C005REC-004AH — Score分解の構造的識別性レビュー

**状態: 完了、`SCORE_DECOMPOSITION_IDENTIFIABILITY_STOP`（ADR-0133）。評価・監査専用であり、学習、candidate、RG3、REC-005、sealed評価は開始しない。**
目的は、immutable AC I03@8000 scorer実装およびqualified REC-004AE/AF/AG成果物の構造的識別性をレビューし、単一構成要素に対する許容介入が修復標的を分離可能か判定することである。

- 監査範囲: 学習0、順伝播修復介入0、係数探索0。計算グラフ、事前登録識別性行列、ハッシュ台帳、機械可読報告書、ADRを記録。
- 判定規則: 4条件（(a) アーキテクチャ固有、(b) target/oracle/baseline非依存、(c) 他要素・下流経路保持、(d) OOD破壊でない局所修復摂動）を同時に満たす構成要素と許容介入が厳密に1つ存在する場合のみ `SCORE_DECOMPOSITION_IDENTIFIABLE_FOR_REPAIR`、それ以外は `SCORE_DECOMPOSITION_IDENTIFIABILITY_STOP`。
- 実行結果: `runs/phase_b_restart/rec004ah/run_001/`。
  1. $S_{QK}$: 条件(d)不成立。token依存変動はpermutationタスクに対して無相関なノイズであり、top-1 routing差（$[-0.0016, +0.0005]$）およびmargin差（$[-0.0025, +0.0034]$）は閾値未達（REC-004AF）。
  2. $S_{\text{position\_bias}}$: 同一長さ内置換は退化（$\Delta=0$）。長さ間transportは離散座標衝突によりEM=0へ崩壊するOOD破壊（REC-004AG）であり、条件(d)不成立。
  3. $S_{\text{residual}}$: 容量不足（norm比0.00039、margin寄与$-0.00029$）。全体拡縮は全floor未達（ADR-0129）で条件(d)不成立。
  4. Softmax全結合: 加法結合後のSoftmax競合結合により、単一要素の孤立した修復標的化は不可能。
- 判定: 条件を満たす要素数は0（厳密に1ではない）。二値判定規則に基づき `SCORE_DECOMPOSITION_IDENTIFIABILITY_STOP` を宣言。現行score分解における単一要素修復の試行を停止する。

## 7D. 実行契約: B-C005REC-004AI — MIRROR Routing 表現要件契約レビュー

**状態: 完了、`MINIMAL_ROUTING_CONTRACT_IDENTIFIED`（ADR-0134）。契約同定・レビュー専用であり、学習0、コード実装0、candidate0、bundle write0、RG3/REC-005/sealed評価は開始しない。**
目的は、ADR-0133の単一構成要素修復停止を受け、既存score分解の対症療法を終了し、`MIRROR_HALVES`のタスク意味論からrouting機構が満たすべき必要条件を導出し、探索なしに単一の最小アーキテクチャ契約を同定できるかを判定することである。

- 実行境界: 学習、最適化器構築、パラメータ更新・追加、重み変更、candidate作成、bundle出力、RG3、REC-005、sealed評価、ハイパーパラメータ探索、アーキテクチャ候補比較探索をfail-closedで禁止。G1/G4は独立ブロックのまま。
- タスク意味論の再構成: `MirrorHalvesOp`および`mirror_halves_position_map`から、$\pi_L(i)$ が純粋な位置置換（content無相関）、全単射・自己逆（involution）、系列長 $L$ 依存（$\lfloor L/2 \rfloor$ の整数床関数段差を含む）、および出力位置 $i$ 依存であることを全合法長 $L \in [2, 16]$ で数学的・決定論的に検証。
- 必要表現要件の導出:
  1. A. Content Invariance: 入力トークン値が変わっても routing argmax が厳密に不変であること。
  2. B. Discrete Positional Distinguishability: 連続正規化座標によるエイリアシングを排除し、離散整数位置を直交・分離して識別できること。
  3. C. Length Awareness: 系列長 $L$ および $\lfloor L/2 \rfloor$ の不連続性を忠実に条件付けできること。
  4. D. Query-Position Awareness: 各出力位置 $i$ ごとに固有のスコアベクトルを生成できること。
  5. E. Permutation Consistency: 全出力位置で重複のない $L$ 個の全単射順列を構成できること。
  6. F. Runtime Target Independence: 正解トークン、正解キー、oracle attention、donor情報を推論時に入力しないこと。
  7. G. Learnability Boundary: 表現可能性（representational sufficiency）と学習獲得性を分離し、仮説空間の表現十分性のみを審査。
- 現行アーキテクチャ監査: $S = S_{QK} + S_{\text{position\_bias}} + S_{\text{residual}}$ は、$S_{QK}$ の content ノイズ干渉（要件A破綻）、$S_{\text{position\_bias}}$ の連続正規化座標エイリアシング・不連続表現不能（要件B, C破綻）、$S_{\text{residual}}$ の極小容量、および加法 Softmax 競合結合により、原理的に必要条件を満たせないことを整理。
- 最小アーキテクチャ契約の演繹的同定: 必要条件から不可欠な機能のみを積み上げ、探索なしに単一の最小契約 `ContentDecoupledDiscretePositionalCrossAttention` (`CD-DPCA`) を同定。
  - Runtime inputs: $(h_{\text{content}}, \text{content\_lengths}, \text{output\_lengths}, \text{argument\_values}=\text{None})$。
  - 離散表現: 整数出力位置埋め込み $E_{\text{query\_pos}}(i) \in \mathbb{R}^{d_{\text{op}}}$、整数系列長埋め込み $E_{\text{length}}(L) \in \mathbb{R}^{d_{\text{op}}}$、整数入力位置埋め込み $E_{\text{key\_pos}}(j) \in \mathbb{R}^{d_{\text{op}}}$。
  - スコア生成: $q(i, L) = E_{\text{query\_pos}}(i) + E_{\text{length}}(L)$、$k(j) = E_{\text{key\_pos}}(j)$ による標準 Multi-Head QK 内積 $S_h(i, j; L) = (q_h(i, L) \cdot k_h(j)^T) / \sqrt{d_{\text{head}}}$（パディング $j \ge L$ は $-\infty$）。
  - 直交分離: $h_{\text{content}}$ はスコア生成から完全に排除され、Value 経路 $V(j) = W_v h_{\text{content}}(j) + E_{\text{val\_pos}}(j)$ にのみ供給。
  - 下流接続: REC-004AE で 100% loss-free が証明された既存の LayerNorm, FFN, Readout パイプラインへ直結。
- ハードコーディング禁止境界: タスク正解式（`mid - 1 - i` 等）や lookup table は一切埋め込まず、汎用整数インデックス $(i, j, L)$ と標準正規乱数初期化を用いた汎用順列仮説クラスを定義。
- 他 relation との静的互換性: 15 non-SHIFT 操作とのテンソル形状、凍結 Core、下流経路、および bundle シリアライズ契約の完全両立を確認。
- 判定: 8つの判定前提をすべて満たし、`MINIMAL_ROUTING_CONTRACT_IDENTIFIED` を宣言。
- 後続作業の順序限定: 次タスクは (1) REC-004AJ (実装 & 未学習構造検証) に限定され、以降 (2) シリアライズ検証 $\to$ (3) 単一init学習パイロット $\to$ (4) 全init検証 $\to$ (5) I01固定採用 $\to$ (6) 15 non-SHIFT RG3 $\to$ (7) 独立5モデル cohort の順を厳格に維持。

## 7E. 実行契約: B-C005REC-004AJ — 最小routing contract実装 & 未学習構造検証

**状態: 完了、`MINIMAL_ROUTING_CONTRACT_IMPLEMENTED_AND_VALIDATED`（ADR-0135）。実装 & 構造検証完了。学習0、candidate0、bundle write0、RG3/REC-005/sealed未実行。**
目的は、ADR-0134で演繹同定された単一の最小CD-DPCA routing pathをコード実装し、未学習状態における7つの構造的不変条件（content無相関性、離散位置・長さアドレス可能性、有限次元順列表現可能性、oracle非流入、padding masking、未学習勾配到達性、下流互換性）を厳格に検証することである。

- 実装範囲: `src/apc/primitives/primitive.py` に `ContentDecoupledDiscretePositionalCrossAttentionPrimitive` (`CD-DPCA`) および `ContentDecoupledDiscretePositionalCrossAttentionPrimitiveConfig` を実装し、`PrimitiveBank` (`src/apc/primitives/bank.py`) に登録。
  - スコア経路のcontent完全排除: クエリ $q(i, L) = E_{\text{query\_pos}}(i) + E_{\text{length}}(L) + \text{arg\_token}$、キー $k(j) = E_{\text{key\_pos}}(j)$。$h_{\text{content}}$ をスコア生成から完全に排除。
  - 離散埋め込み: 整数インデックス $i, j \in [0, L_{\text{max}}-1]$、系列長 $L \in [0, L_{\text{max}}]$ を `nn.Embedding` で直交表現。連続座標の正規化除算・グリッド衝突（REC-004AG）を排除。
  - 既存下流接続の保持: Value経路 $v(j) = W_v h_{\text{content}}(j) + E_{\text{val\_pos}}(j)$、出力射影、LayerNorm、FFN、Readout、および非SHIFT bundleインターフェースを完全維持。
  - 汎用relation条件付け境界: MIRROR固有分岐やtarget map計算を一切含めず、引数付き操作のみ汎用 `arg_encoder`/`arg_proj` を経由。
- 実行制約: 最適化器構築0、重み更新0、checkpoint変更0、candidate作成0、bundle出力0、RG3未実行、REC-005未実行、sealed非参照。G1およびG4は未解除の独立ブロックとして保持。
- 構造検証（`runs/phase_b_restart/rec004aj/run_001/`、`tests/test_rec004aj_cd_dpca_contract.py`）:
  1. Scorer Content Invariance: 入力content変更時、ルーティングスコア差 $\max |\Delta S| = 0.0$、注意重み差 $\max |\Delta A| = 0.0$。下流出力のみ有意変動（$\max |\Delta y| = 3.577$）。PASS。
  2. 離散位置・長さアドレス可能性: 全合法長 $L \in [2, 16]$ でインデックス参照成功。異なる行のpairwise距離 $> 0.1$。$L > 32$ は `ValueError` で境界遮断。PASS。
  3. 順列表現可能性: 全合法MIRROR長 $L \in [2, 16]$ で正解キー確率 $> 0.99$、runner-upマージン $> 6.0$ を構成的証明。PASS。
  4. 情報境界: ルーティング経路に入力される引数に正解token・oracle注意・教示写像が一切存在しないことをAST/リフレクションで監査。PASS。
  5. Padding Masking: 無効位置 $j \ge L$ はスコア $-\infty$、重み $0.0$。有効位置の和は $1.0$。PASS。
  6. 勾配到達性: 未学習順伝播の微分損失から、クエリ・キー・長さ埋め込みおよびQK射影の勾配ノルムがすべて正値。最適化器ステップは未実行（更新0）。PASS。
  7. 下流互換性: テンソル形状 `[batch, max(out_lengths), vocab_size]`、`PrimitiveBank` 登録・呼び出し統計、`state_dict` シリアライズ互換性を確認。PASS。
- 有限表現境界の記録: サポート長 $L \in [1, 32]$、全単射順列階数上限 $L \le \min(L_{\text{max}}, d_{\text{operator}}) = 32 \ge 16$。無限長外挿は明示的に棄却（alias回避と引き換えの離散有限表現）。
- 判定: 7基準すべてPASSにより `MINIMAL_ROUTING_CONTRACT_IMPLEMENTED_AND_VALIDATED` を宣言。
- 次タスクの順序限定: 次段階は (1) REC-004AK (シリアライズ & 新規ロード検証) に限定される。

## 7F. 実行契約: B-C005REC-004AK — CD-DPCAシリアライズ & 新規ロード厳格検証

**状態: 完了、`CD_DPCA_SERIALIZATION_AND_FRESH_LOAD_VALIDATED`（ADR-0136）。シリアライズおよびfresh-load完全等価性検証完了。学習0、candidate0、bundle write0、RG3/REC-005/sealed未実行。**
目的は、CD-DPCA (`ContentDecoupledDiscretePositionalCrossAttentionPrimitive`) および `PrimitiveBank` の最小限のシリアライズ・設定登録サポート（`to_dict` / `from_dict`、`to_manifest` / `from_manifest`、`save_artifacts` / `from_artifacts`）を実装し、immutable model bundle loader contract（`strict=True`）下で未学習インスタンスの永続化と新規構築の完全等価性、および厳格な負例チェック・情報境界を検証することである。

- 実行境界: オプティマイザ構築0、パラメータ更新0、学習0。既存checkpoint変更0、candidate採択0、ModelBundle publish/write禁止（`bundle_write = False`）。RG3未実行、REC-005未実行、sealed非参照。G1およびG4は未解除の独立ブロックとして保持。
- 実装範囲:
  - `src/apc/primitives/primitive.py`: `CD_DPCA_ARCHITECTURE_SIGNATURE = "content_decoupled_discrete_positional_cross_attention_v1"` を定義。各PrimitiveConfigおよびCD-DPCA configに厳格なキー・型・ハイパーパラメータ検証付きの `to_dict()` / `from_dict()` を追加。PrimitiveBaseに `to_config_dict()` を追加し、各クラスに `ARCHITECTURE_SIGNATURE` および `from_config_dict()` を実装。グローバルレジストリ `PRIMITIVE_TYPE_REGISTRY` およびファクトリ `build_primitive_from_config_dict()` を配備。
  - `src/apc/primitives/bank.py`: `PrimitiveBank` にマニフェスト形式 `to_manifest()`、`from_manifest()`、ファイル保存/復元 `save_manifest()`、`load_manifest()`、複合アーティファクト保存/復元 `save_artifacts()`、`from_artifacts(strict=True)` を追加。
- 厳格検証（`runs/phase_b_restart/rec004ak/run_001/`、`tests/test_rec004ak_cd_dpca_serialization.py`）:
  1. パラメータ完全等価性: 20テンソル（総数19,178 params）の名前、形状、torch.float32 dtype、canonical state hash、state ABI hashが完全一致。
  2. スコアラー出力完全等価性: 同一入力長に対するルーティングスコアがビット一致（$\max |\Delta S| = 0.0$）。
  3. 注意重み & パディングマスク完全等価性: 注意重み差 $\max |\Delta A| = 0.0$。無効パディング位置 ($j \ge L$) はスコア $-\infty$、重み $0.0$。有効位置の注意重み和は $1.0$。
  4. 順伝播ロジット完全等価性: ロジット差 $\max |\Delta y| = 0.0$。
  5. 呼び出し統計互換性: スパース呼び出しカウンタ、使用回数記録、リセット操作が元インスタンスと完全両立。
  6. 10種の厳格負例テスト（Fail-Closed）:
     - 欠落state_dictキー (`length_embedding.weight`) $\to$ `RuntimeError("Missing key(s)")`
     - 予期せぬstate_dictキー (`extra_unrecognized_tensor`) $\to$ `RuntimeError("Unexpected key(s)")`
     - 不適合テンソル形状 (`(16, 32)` vs `(32, 32)`) $\to$ `RuntimeError("size mismatch")`
     - 必須configフィールド欠落 (`operation`) $\to$ `KeyError`
     - 予期せぬconfigフィールド (`unsupported_hyperparameter`) $\to$ `ValueError`
     - 不適合演算子次元整除性 (`d_operator=30`, `n_head=4`) $\to$ `ValueError`
     - 不適合最大系列長 (`max_sequence_length=0`) $\to$ `ValueError`
     - 未知のプリミティブ型 (`NonexistentPrimitiveClass`) $\to$ `ValueError`
     - 重複プリミティブID (`primitive_id=0`) $\to$ `ValueError`
     - アーキテクチャ署名不一致 (`cross_position_v1` vs CD-DPCA) $\to$ `ValueError`
  7. ランタイム情報境界監査: AST/シグネチャ検査により、復元後のルーティングパスに正解トークン、ラベル、教示注意分布、オラクル情報が一切入力されないことを確認。
- 判定: `CD_DPCA_SERIALIZATION_AND_FRESH_LOAD_VALIDATED`。
- 次タスクの順序限定: 次段階は (3) 単一init学習パイロット（REC-004AL）に限定される。

## 7G. 実行契約: B-C005REC-004AL — CD-DPCA単一init学習パイロット

**状態: 完了、`PILOT_TERMINAL_VIABILITY_NOT_MET`（ADR-0137）。終端実行性基準（EM≥0.95）未達によりfail-closed停止。全init検証・候補採択・bundle出力は厳格に遮断。**
目的は、REC-004AKで検証されたCD-DPCA (`ContentDecoupledDiscretePositionalCrossAttentionPrimitive`) の新規ロード状態から単一初期化（I01）の学習可能インスタンスを構築し、教示オラクルなしの通常トークン損失下で `MIRROR_HALVES` 順列を学習できるかを厳格な実行境界と事前登録評価プロトコル下で審査することである。

- 実行境界 & 保護要件:
  - 凍結・保護: 親bundleのCoreおよびrouterは完全凍結 (`requires_grad = False`, hash照合済)。親bankの他15プリミティブも完全凍結。学習対象は `MIRROR_HALVES` スロット（物理ID 12）のCD-DPCA単一インスタンスのみ。
  - ソース整合性: REC-004AKアーティファクトのSHA-256ハッシュ（マニフェスト、config、bank/primitive state）を事前検証。
  - 情報境界: ランタイムルーティングに正解トークン、ラベル、教示注意分布、順列ルックアップテーブルが一切入力されないことをASTおよびリフレクションで監査。
  - 副作用厳格遮断: candidate選定0 (`candidate_selected: null`), bundle write0 (`child_bundle: null`, `bundle_write: false`), 追加init実行0, ハイパーパラメータ探索0, 予算延長0, 封印データアクセス0。RG3未実行 (`rg3: NOT_EXECUTED`), `rec005_eligible: false`。G1およびG4は未解除の独立ブロックとして保持。
- 学習設定:
  - 最適化器: AdamW (`lr = 1e-3`, `weight_decay = 0.01`, `betas = (0.9, 0.999)`), CosineAnnealingLR (`T_max = 6,000`, `eta_min = 1e-5`)。
  - 予算 & 評価頻度: 最大6,000 updates。500 stepごとにcheckpoint・訓練状態を保存し、1,024例の既存開発validation setで評価。
- 終端実行性基準: 決定ステップ 6,000 における系列完全一致精度（sequence EM） $\ge 0.95$。未達の場合は即座に fail-closed 停止し、全init検証（REC-004AM）への移行を禁止。
- 実行結果（`runs/phase_b_restart/rec004al/run_001/`）:
  1. 検証指標（ステップ 6,000）:
     - 系列EM: `0.793945` (813 / 1,024例) vs 終端基準 `>= 0.95` $\to$ `terminal_viability_met: false`（基準未達）。
     - トークン精度: `0.973254` (8,029 / 8,250トークン)。
  2. 系列長別EM内訳（ステップ 6,000）:
     - 長さ 6: `199 / 204` (EM = `0.9755`, トークン精度 = `0.9959`) $\to$ 長さ別基準クリア
     - 長さ 7: `170 / 214` (EM = `0.7944`, トークン精度 = `0.9693`)
     - 長さ 8: `178 / 194` (EM = `0.9175`, トークン精度 = `0.9897`)
     - 長さ 9: `196 / 206` (EM = `0.9515`, トークン精度 = `0.9946`) $\to$ 長さ別基準クリア
     - 長さ 10: `70 / 206` (EM = `0.3398`, トークン精度 = `0.9311`) $\to$ 主たる失点局在（長さ依存崩壊）
  3. 因果対照（ステップ 6,000）:
     - 正解制御 EM: `0.7939`、逆族制御 (`REVERSE`) EM: `0.0000`、未ルーティング制御 (`None`) EM: `0.0000`、因果ギャップ: `0.7939`（有意な演算子依存性）。
  4. 注意 & パディング診断:
     - パディングマスク検証 PASS（パディング位置スコア $-\infty$、注意重み $0.0$、有効位置重み和 $1.0$）。
     - Top-1 正解キールーティング精度: `0.8964` (1,826 / 2,037位置)。
     - runner-up に対する正解スコアマージン平均: `6.4475`。
  5. 整合性 & 副作用監査:
     - ソースハッシュ照合 PASS、情報境界監査 PASS、フリーズ監査 PASS、副作用監査 PASS。
- 判定と影響:
  - 判定: `PILOT_TERMINAL_VIABILITY_NOT_MET`。
  - 事前登録された終端実行性基準（EM $\ge 0.95$）を満たさなかったため、フェイルクローズ停止を発動。
  - 全init検証（REC-004AM）、候補採択、およびbundle出力への進行はすべて遮断される。

## 7H. 実行契約: REC-004AN CD-DPCA Length-10 Failure Localization

**状態: 完了、`LENGTH10_LOCAL_OPTIMIZATION_FAILURE_IDENTIFIED`（ADR-0138）。追加学習0、評価専用診断により失点の89.4%が位置4の偽アトラクタ固着（key 7）に局在することを確認。全init検証・候補採択・bundle出力は厳格に遮断を継続。**
目的は、REC-004ALで確認されたCD-DPCA single-init failureについて、表現不足、アーキテクチャ不足、訓練予算不足、length固有パラメータ故障、shared routing干渉、late regression、特定位置局在を区別し、単一の最適化介入を選択できるだけの証拠があるかを評価専用診断で判定することである。

- 実行境界 & 保護要件:
  - 評価専用診断: 追加学習0、`optimizer.step()` 実行0、予算延長0、新初期化0、LR/curriculum/sampling変更0、アーキテクチャ変更0。
  - 凍結・保護: 親Coreおよび15 non-MIRROR primitivesは完全凍結・不変。
  - ソース整合性: REC-004ALの全13 checkpoints (step 0..6000) のSHA-256ハッシュを事前検証。
  - 副作用遮断: `candidate_selected: null`, `child_bundle: null`, `bundle_write: false`, RG3 `NOT_EXECUTED`, `rec005_status: BLOCKED`, G1/G4 `NOT_CLEARED`。封印データアクセス0。
- 診断結果（`runs/phase_b_restart/rec004an/run_001/`）:
  1. 全13 checkpoint 軌道再評価:
     - 判定: `NEVER_LEARNED_PATTERN`。
     - 長さ10の系列EMは step 0 (0.0000) から step 6000 (0.3398) までほぼ単調に推移し、途中で高い性能領域（>=0.50）に到達した形跡は皆無。late regression は明確に反証。
  2. 位置レベル局在:
     - ステップ6000における長さ10の10出力位置別トークン精度:
       - pos 0, 1, 2, 5, 6, 7, 8: `1.0000` (各 206/206)
       - pos 9: `0.9951` (205/206, 1失点)
       - pos 3: `0.9320` (192/206, 14失点)
       - pos 4: `0.3835` (79/206, 127失点) $\to$ 全142失点中127失点（**89.44%**）が位置4に集中。位置3-4合計で**99.30%**。
     - 分類: 位置4が唯一の `consistently_failing_position`。位置0, 1, 5, 6, 7, 8, 9は `consistently_correct_positions`。
  3. 偽アトラクタ固着:
     - 長さ10の正解マッピングは $\pi_{10} = (4, 3, 2, 1, 0, 9, 8, 7, 6, 5)$。位置4の正解キーは 0。
     - 位置4は訓練初期の step 500 から最終 step 6000 に至る全チェックポイントで一貫して key 7 をトップ1選択（step 6000 で $p_7 = 0.4906$ vs $p_0 = 0.0001$, margin = $-14.21$）。
     - 他の9位置が正解キールーティングを達成する中、位置4のみが偽アトラクタに初期からトラップされ続けた。
  4. 状態移植診断:
     - ベース: step 6000。過去チェックポイントからの移植を実施:
       - 介入A ($E_{\text{length}}[10]$ 移植): 最大EM = 0.3495（回復なし）
       - 介入B (Shared routing state 移植): 最大EM = 0.3689（回復なし）
       - 介入C (Full routing 移植サニティ): 最大EM = 0.3641（回復なし）
     - 判定: `NO_TRAJECTORY_LOCALIZATION`。過去のどの時点でも長さ10は正しく学習されていなかったため、過去軌道からの移植では回復しない。
  5. 勾配競合診断:
     - 長さ10と他長さ（6-9）の shared routing 勾配の余弦類似度: step 0 で `+0.4190`、step 6000 で `-0.0090`（直交、逆平行ではない）。
     - 強い勾配干渉（`SHARED_ROUTING_GRADIENT_INTERFERENCE_IDENTIFIED`）は反証。
  6. 訓練露出監査:
     - 全6,000 steps (192,000例) 中、長さ10は 38,254例（19.92%）、382,540トークン、更新露出度 99.93% で他長さと完全均等。データ不足は反証。
- 判定と影響:
  - 判定: `LENGTH10_LOCAL_OPTIMIZATION_FAILURE_IDENTIFIED`。
  - 次期最適化修復の標的が「位置4の偽アトラクタ脱出・境界ルーティング最適化」という単一メカニズムに一意に特定された。
  - ただし本タスク内での学習再試行・新パイロット実行は認可されない。全init検証、候補採択、bundle出力は厳格に遮断を継続。

## 7I. 実行契約: REC-004AO CD-DPCA 位置4偽アトラクタ勾配到達性診断

**状態: 完了、`LOCAL_LOSS_GRADIENT_MISALIGNMENT_IDENTIFIED`（ADR-0139）。追加学習0、評価専用診断によりルーティング勾配ノルムは十分（0.2707）だが正解margin変化率は84.6%で非正（端末-0.6976）かつkey0飽和飢餓を確認。全init検証・候補採択・bundle出力は厳格に遮断を継続。**
目的は、長さ10 / 出力位置4で確認された持続的偽アトラクタ（key 7）について、標準token-output cross-entropy lossから正解ルーティングへ戻すための勾配信号が実際に到達しているかを評価専用で判定することである。

- 実行境界 & 保護要件:
  - 評価専用診断: 追加学習0、`optimizer.step()` 実行0、予算延長0、新初期化0、LR/temperature/curriculum/sampling変更0、アーキテクチャ変更0。
  - 凍結・保護: 親Coreおよび15 non-MIRROR primitivesは完全凍結・不変。
  - ソース整合性: REC-004ALの全13 checkpoints (step 0..6000) のSHA-256ハッシュを事前検証。
  - 副作用遮断: `candidate_selected: null`, `child_bundle: null`, `bundle_write: false`, RG3 `NOT_EXECUTED`, `rec005_status: BLOCKED`, G1/G4 `NOT_CLEARED`。封印データアクセス0。
- 診断結果（`runs/phase_b_restart/rec004ao/run_001/`）:
  1. 位置4ルーティング軌道（全13 checkpoint）:
     - step 0: top-1 key 3, $p(0) = 0.0500$, $p(7) = 0.0951$, margin $-0.5964$, entropy $2.2740$, 精度 $0.0777$。
     - step 500: 偽アトラクタ key 7 へ急激に突入（$p(7) = 0.4048$, $p(0) = 0.00184$, margin $-6.4724$, entropy $1.4966$）。
     - step 1000..6000: step 500 以降の全チェックポイントで key 7 に完全に固着。margin は $-14.2144$ まで単調悪化、entropy は $1.1221$ まで低下、端末 $p(0) = 1.15 \times 10^{-4}$、端末精度 $0.3835$。
  2. 局所勾配到達性と一次方向微分:
     - ルーティングパラメータ全体の勾配ノルムは訓練全体を通して十分な大きさで存在（step 6000 で $\|g_{\text{loss, routing}}\| = 0.2707$、step 3500 で最大 $1.314$；$W_k = 0.2031$, $W_q = 0.1299$, $k_7 = 0.0463$）。勾配自体は到達している。
     - しかし、一次予測正解マージン変化 $\Delta M \propto - g_{\text{margin}} \cdot g_{\text{loss}}$ は、全13チェックポイント中11件（**84.6%**）で非正であり、端末 step 6000 でも負（**$-0.6976$**、余弦類似度 $-0.1116$）。通常のトークン損失勾配降下は正解マージンを改善せず、偽アトラクタを強化する方向に働く。
  3. Key 0 に対する Softmax 勾配飢餓:
     - 共有ルーティングパラメータに大きな勾配が流れる一方、正解キー埋め込み $E_{\text{key\_pos}}[0]$ に直接届く勾配ノルムは step 0 の $1.37 \times 10^{-3}$ から step 3000 で $7.87 \times 10^{-5}$、step 6000 で $7.24 \times 10^{-4}$ へと急減し、key 7 比で 64倍〜1500倍の飢餓状態に陥る。
     - これは softmax の連鎖律 $\nabla_{S(4, 0)} \mathcal{L} \propto p(0) \approx 10^{-4}$ による飽和減衰が直接原因。
  4. 例単位の一貫性:
     - 長さ10の全206検証例（step 6000）において、57.8% が負のマージン変化を示し、中央値は $-1.2380$、平均値は $-0.6976$。
     - 位置4誤答127例においても中央値は $+0.0346$（ほぼゼロ）、49.6% が負、50.4% が正で、正方向への駆動成分は完全に相殺・消失。飢餓比率 $\|g_{k0}\| / \|g_{k7}\| = 0.0063$。
  5. 対照位置との比較:
     - 長さ10位置3: 精度 0.932, 正解key 1, 予測マージン変化 $+2.4685$（正方向余弦 $+0.0875$）。
     - 長さ10位置0: 精度 1.000, 正解key 4, 予測マージン変化 $+0.6618$（正方向余弦 $+0.2949$）。
     - 長さ8位置4: 精度 1.000, 正解key 7, 予測マージン変化 $+0.4894$（正方向余弦 $+0.4550$）。
     - 長さ9位置4: 精度 1.000, 正解key 8, 精度 1.000。
     - 位置4のみが持続的偽アトラクタ、極端な負マージン（$-14.21$）、および負の勾配アライメント（$-0.6976$）を同時に示す。
- 判定と影響:
  - 判定: `LOCAL_LOSS_GRADIENT_MISALIGNMENT_IDENTIFIED`。
  - ルーティングパラメータ全体の勾配は消失（starvation）しておらず十分な大きさを持つが、標準トークン損失勾配の正解マージン方向成分が非正（misalignment）であり、偽アトラクタを脱出できない。同時に、正解キー単独への勾配は softmax 飽和により極度に飢餓している。
  - 単純な anti-saturation ヒューリスティック単体への安易な移行は禁止され、標準トークン損失という研究境界のままで解決可能な単一 optimization formulation のレビューが必要とされる。
  - 本タスク内での新規学習パイロットは認可されず（`next_learning_pilot_authorized: false`）、全init検証（REC-004AM）、候補採択、ModelBundle出力は厳格に遮断を継続。

## 7J. 実行契約: REC-004AP CD-DPCA 位置4トークン識別性層別化勾配アライメント診断

**状態: 完了、`TOKEN_ALIASING_DILUTION_AND_LATE_SATURATION_IDENTIFIED`（ADR-0140）。追加学習0、評価専用診断によりStrata A/B（86.9%）では正解margin改善勾配が存在する一方、Stratum C（13.1%）の巨大破壊勾配（12.4倍優勢）がpooled勾配を希釈・反転させ、端末ではkey 0飽和飢餓が加わる二段階メカニズムを同定。全init検証・候補採択・bundle出力は厳格に遮断を継続。**
目的は、ADR-0139で報告された損失勾配不整合が、標準損失固有の表現幾何不具合なのか、パラメータ空間での反転なのか、それともトークン重複（token aliasing）によるcredit dilutionなのかを、トークン識別性に基づき事前固定した3層別（Stratum A: 唯一正解、B: 他キー重複かつkey 7非重複、C: key 7重複）においてスコア空間勾配 $dL/dS(4, j)$ とパラメータ空間勾配アライメント $-g_{\text{margin}} \cdot g_{\text{loss}}$ を測定して峻別することである。

- 実行境界 & 保護要件:
  - 評価専用診断: 追加学習0、`optimizer.step()` 実行0、予算延長0、新初期化0、LR/温度/サンプリング変更0、アーキテクチャ変更0。
  - 凍結・保護: 親Coreおよび15 non-MIRROR primitivesは完全凍結・不変。
  - ソース整合性: REC-004ALの全13 checkpoints (step 0..6000) のSHA-256ハッシュを事前検証。
  - 副作用遮断: `candidate_selected: null`, `child_bundle: null`, `bundle_write: false`, RG3 `NOT_EXECUTED`, `rec005_status: BLOCKED`, G1/G4 `NOT_CLEARED`。封印データアクセス0。
- 診断結果（`runs/phase_b_restart/rec004ap/run_001/`）:
  1. 長さ10検証例（206例）の層別分布:
     - Stratum A（unique target）: 77例（37.38%）。正解key 0にのみtarget tokenが存在。
     - Stratum B（aliased other）: 102例（49.51%）。他キーにも存在するが偽アトラクタkey 7には不在。
     - Stratum C（aliased key 7）: 27例（13.11%）。key 7にもtarget tokenが存在。
     - 相互排他性・網羅性を検証（$77 + 102 + 27 = 206$）。非重複・他重複例が全体の 86.89% を占める。
  2. 訓練中期ダイナミクス（Steps 500..5000）:
     - 全体の 86.89% を占める Strata A および B では、標準トークン損失からパラメータ空間へ【正解マージンを改善する正の勾配】（$-g_{\text{margin}} \cdot g_{\text{loss}} > 0$）が支配的に供給されていた（Stratum A: 7/9 checkpointで正、Stratum B: 8/9 checkpointで正）。
     - onsetの step 500 において、Stratum A は $+9.1296$（54.5%正）、Stratum B は $+11.5596$（61.8%正）と強力な正解脱出シグナルを出していた。
     - しかし、Stratum C（13.11%）が $-31.0566$（96.3%負、平均絶対値 $28.52$ vs Stratum A $2.30$、**12.4倍の圧倒的優勢度**）という巨大な負の誤学習シグナルを放出したため、全体平均（pooled）が負（$-3.83$）へ引きずり倒されていた。
     - スコア空間でも、Stratum C は $\partial \mathcal{L} / \partial S(4, 7) = -4.7 \times 10^{-3}$ となり、key 7 を選んでも正解トークンが得られることによる偽の成功シグナル（credit dilution）が実証された。
  3. 端末 Step 6000 ロックイン:
     - 長期の誤結合により key 0 確率が $p(0) \approx 1.15 \times 10^{-4}$ に飽和し、key 0 埋め込みへの勾配が key 7 比で 1000倍以上飢餓。
     - この極端な飽和下で、Stratum A でも正解マージン変化が $-0.7266$（中央値 $-0.9921$、55.8%負）へと減衰・反転した。Stratum B は $+0.8263$（52.0%正）、Stratum C は $-6.3716$（100%負）。
  4. 対照位置比較:
     - 長さ10位置3（正解key 1, 対抗7）: 93.2%精度、正解margin変化率は訓練全般で正。
     - 長さ10位置0、長さ8位置4、長さ9位置4: 100%精度、一貫して正のアライメント。
- 判定と影響:
  - 判定: `TOKEN_ALIASING_DILUTION_AND_LATE_SATURATION_IDENTIFIED`。
  - ADR-0139の「損失勾配不整合」は、標準損失固有の表現幾何不具合ではなく、**トークン重複によるcredit assignmentの錯覚（Token-Aliasing Credit Dilution）**が主因であり、末期に**Softmax飽和飢餓**が加わって固定化された二段階メカニズムとして解釈が限定・確定された。
  - 新規学習パイロットは認可されず（`next_learning_pilot_authorized: false`）、全init検証（REC-004AM）、候補採択、およびModelBundle出力は厳格に遮断を継続。

## 7K. 実行契約: B-C005REC-004AQ — CD-DPCA系列内非復元抽出warm-start単一レシピ因果パイロット

**状態: 完了、`WARM_START_PILOT_VIABILITY_MET`（ADR-0141）。系列完全一致精度 0.9854（1009/1024例）、長さ10 EM 1.0000（206/206例）を達成し、終端実行性基準（EM≥0.95）および位置4偽アトラクタ解消を完全クリア。全init検証（REC-004AM）の検討が認可。候補採択・bundle出力・RG3・REC-005は厳格に遮断を継続。**
目的は、ADR-0138〜0140で特定された「初期500ステップにおけるトークン重複（Stratum C）によるcredit dilutionが位置4の偽アトラクタ固着を引き起こす」という因果仮説を検証するため、初期500ステップのみ系列内非復元抽出（pairwise-distinct）を適用する単一介入を行い、通常トークン損失のみでCD-DPCAが終端実行性基準（EM≥0.95）を達成し位置4の偽アトラクタを脱出できるかを実証することである。

- 実行境界 & 厳格対照要件:
  - REC-004AL (I01) との完全一致: step-0 model/canonical hash (`045d85cae86d54ce1caca1947a805f2f424df55c34fba4cde11cafa6bbec49dc`)、AdamW (`lr = 0.0008, weight_decay = 0.0001, grad_clip = 1.0`)、CosineAnnealingLR (`T_max = 1000, eta_min = 1e-5`, mechanical extension)、バッチサイズ（32例/step）、総6,000 updates、評価頻度（500 stepごと全13 checkpoints）、固定開発validation（1,024例）をビット・プロトコル一致。
  - 唯一の介入: steps 1–500 では各入力系列のトークンを非復元抽出して系列内を pairwise-distinct にし、steps 501–6000 では REC-004AL の元 sampler と per-step seed 式 (`ibc._generate_step_training_examples`) へ正確に戻す。500-step 境界は事前固定（sweep禁止）。全 length・全 position へ同じ規則を適用（特定 length や position の選別禁止）。
  - 損失関数 & アーキテクチャ: 通常トークン損失 (CE) のみ。oracle/teacher routing loss、補助損失、エントロピー/温度変更、LR変更は禁止。新規 MIRROR temporary primitive (id 12) のみを更新し、Core および 15 non-MIRROR primitives は完全凍結。
  - 副作用厳格遮断: candidate選定0 (`candidate_selected: null`), bundle write0 (`child_bundle: null`, `bundle_write: false`), RG3未実行 (`rg3: NOT_EXECUTED`), `rec005_eligible: false`。G1/G4は未解除の独立ブロックとして保持。
- 実行結果（`runs/phase_b_restart/rec004aq/run_001/`）:
  1. 検証指標（ステップ 6,000）:
     - 系列EM: `0.985352` (1009 / 1024例) vs ベースライン `0.793945` (+0.1914)。終端基準（>=0.95）をクリア。
     - トークン精度: `0.998168` (8235 / 8250トークン) vs ベースライン `0.973254`。
  2. 系列長別EM内訳（ステップ 6,000）:
     - 長さ 6: `196 / 204` (EM = `0.9608`, トークン精度 = `0.9935`)
     - 長さ 7: `210 / 214` (EM = `0.9813`, トークン精度 = `0.9973`)
     - 長さ 8: `194 / 194` (EM = `1.0000`, トークン精度 = `1.0000`)
     - 長さ 9: `203 / 206` (EM = `0.9854`, トークン精度 = `0.9984`)
     - 長さ 10: `206 / 206` (EM = **`1.0000`**, トークン精度 = **`1.0000`**) $\to$ REC-004ALの 0.3398 から完全回復！
     - 全5系列長が 0.95 の基準を達成。
  3. 位置4ルーティング軌道 & 偽アトラクタ完全解消:
     - step 0: top-1 key 3, margin $-0.60$, 精度 $0.0777$。
     - step 500 (warm-start完了時): top-1 key **0** (正解), $p(0) = 0.4083$, $p(7) = 0.1627$, margin **$+0.55$**, 精度 $0.3495$（REC-004ALでは key 7 に margin $-6.47$ で固着していた）。
     - step 1000 (通常データ復帰後): top-1 key 0, margin **$+2.97$**, 精度 $0.8252$。
     - step 1500..6000: margin は $+3.59$ から **$+5.38$** へ単調拡大。step 2000 以降は精度 **`1.0000`** (206/206) を完全維持。
     - 端末 step 6000: 206例中206例（100.0%）が正解 key 0 を選択。偽アトラクタ key 7 の選択数は **0件**（0.0%）。
  4. 層別勾配アライメント軌道:
     - warm-start 中 (steps 1..500) は非復元抽出により Stratum C の破壊的偽報酬がゼロとなり、正解アトラクターへの引き込みが成功。
     - 通常復元抽出データへの復帰後 (steps 501..6000) も、一度形成された正解アトラクターが壊れることなく維持され、Stratum C の破壊的勾配はほぼゼロ（$-0.0050$）へと収束。
  5. 他位置への回帰なし:
     - 全系列長・全出力位置においてトークン精度が 0.96 以上を維持（失点は全体でわずか 15 トークン）。
  6. 因果対照（ステップ 6,000）:
     - 正解制御 EM: `0.9854`、逆族制御 (`REVERSE`) EM: `0.0000`、未ルーティング制御 (`None`) EM: `0.0000`、因果ギャップ: `0.9854`（極めて有意な演算子依存性）。
  7. 整合性 & 副作用監査:
     - パディングマスク検証 PASS、フリーズ監査 PASS、副作用監査 PASS。
- 判定と影響:
  - 判定: `WARM_START_PILOT_VIABILITY_MET`。
  - 単一介入（初期500ステップのpairwise-distinctサンプリング）のみで、位置4の偽アトラクタが解消され、終端実行性基準（EM≥0.95）が完全に達成された。
  - 次段階として、REC-004AM（この warm-start 単一レシピを用いた独立5初期化の検証）の計画・実行を検討することが認可される。
  - 候補採択・bundle出力・RG3・REC-005は未認可であり、厳格に遮断を継続。

## 7L. 実行契約: B-C005REC-004AM — 固定warm-start全5初期化再現性検証

**状態: 完了、`MULTI_INIT_VIABILITY_NOT_MET`（ADR-0143）。事前登録された5独立初期化（I01..I05）において、合格は1/5（I01のみ完全合格、I02〜I05は偽アトラクタ固着または他位置回帰により不合格）。事前登録適格性基準（全5/5合格）未達によりフェイルクローズ停止。候補採択・bundle出力・RG3・REC-005は厳格に遮断を継続。**
目的は、ADR-0141 (REC-004AQ) で成功した系列内非復元抽出warm-start単一レシピ（AdamW, lr=0.0008, weight_decay=0.0001, grad_clip=1.0, CosineAnnealingLR T_max=1000 eta_min=1e-5, 32例/step, 6,000 updates, 初期500 steps pairwise-distinct warm-start）をビット不変のまま固定し、事前登録された独立5初期化（I01..I05）において全5/5が終端実行性基準（EM≥0.95、位置4偽アトラクタ解消、他位置非回帰、因果ギャップ≥0.90）を満たすかを検証することである。

- 実行境界 & 厳格対照要件:
  - ADR-0141 (REC-004AQ) レシピのビット完全固定: 最適化器、スケジューラ、バッチサイズ、6,000 updates/init（総予算30,000 updates）、評価頻度（500 stepごと全13 checkpoints/init）、固定開発validation（1,024例）。
  - データストリームの全init完全一致: 各更新ステップで全5 initが全く同一の訓練バッチを受け取る（`data_stream_identical_across_inits: true`）。初期500 stepsは非復元抽出、steps 501..6000は元の復元サンプラー。
  - 損失関数 & アーキテクチャ: 通常トークン損失 (CE) のみ。oracle/teacher routing loss、補助損失は禁止。新規 MIRROR temporary primitive (id 12) のみを更新し、Core および 15 non-MIRROR primitives は完全凍結。
  - 事前登録初期化 (I01..I05): I01 (seed 20260912), I02 (seed 20260913), I03 (seed 20260914), I04 (seed 20260915), I05 (seed 20260916)。
  - 事前登録適格性規則: 全5/5が独立して合格基準をクリアすること（1/5や4/5は不合格・停止）。
  - 副作用厳格遮断: candidate選定0 (`candidate_selected: null`), bundle write0 (`child_bundle: null`, `bundle_write: false`), RG3未実行 (`rg3: NOT_EXECUTED`), `rec005_eligible: false`。G1/G4は未解除の独立ブロックとして保持。
- 実行結果（`runs/phase_b_restart/rec004am/run_001/`）:
  1. マルチ初期化再現性サマリー:
     - 合格初期化数: **1 / 5**（I01 のみ合格、I02〜I05 は不合格）。
     - 適格性規則判定: `False`。
     - 実行判定: `FAIL` / `MULTI_INIT_VIABILITY_NOT_MET`。
     - 平均決定系列EM: `0.950195`（I01: 0.9854, I02: 0.8623, I03: 0.9629, I04: 0.9658, I05: 0.9746）。
     - 総更新数: 30,000 updates (6,000 updates x 5 inits)、実行時間: 205.15s。
  2. 初期化別内訳（ステップ 6,000）:
     - `I01` (seed 20260912): 系列EM `0.985352` (1,009/1,024), 長さ10 EM `1.0000`, 位置4 top key 0 (margin +5.38, acc 1.0000), 因果ギャップ 0.9854 $\to$ **PASS**（ADR-0141の100%ビット完全再現）。
     - `I02` (seed 20260913): 系列EM `0.862305` (883/1,024), 長さ10 EM `0.5340`, 位置4 top key **5** (margin -10.90, acc 0.6068) $\to$ **FAIL**（対抗key 5偽アトラクタ固着、系列EM < 0.95）。
     - `I03` (seed 20260914): 系列EM `0.962891` (986/1,024), 長さ10 EM `0.8689`, 位置4 top key **3** (margin -8.95, acc 0.8689) $\to$ **FAIL**（対抗key 3偽アトラクタ固着、位置4精度 < 0.95）。
     - `I04` (seed 20260915): 系列EM `0.965820` (989/1,024), 長さ10 EM `0.9903`, 位置4 top key 0 (margin +5.19, acc 1.0000) だが、長さ8位置3トークン精度が `0.8711` (< 0.90) に劣化 $\to$ **FAIL**（他位置回帰）。
     - `I05` (seed 20260916): 系列EM `0.974609` (998/1,024), 長さ10 EM `0.9806`, 位置4 top key **1** (margin -1.22, acc 0.9951) $\to$ **FAIL**（位置4で正解key 0ではなく対抗key 1が優勢）。
  3. 系列長別EM内訳（初期化間比較）:
     - 長さ 6 EM: I01=0.9608, I02=0.9412, I03=0.9804, I04=0.9951, I05=0.9657
     - 長さ 7 EM: I01=0.9813, I02=0.9579, I03=0.9953, I04=0.9673, I05=0.9579
     - 長さ 8 EM: I01=1.0000, I02=0.9742, I03=0.9948, I04=0.8711, I05=0.9794
     - 長さ 9 EM: I01=0.9854, I02=0.9078, I03=0.9757, I04=1.0000, I05=0.9903
     - 長さ 10 EM: I01=1.0000, I02=0.5340, I03=0.8689, I04=0.9903, I05=0.9806
   4. 整合性 & 副作用監査:
      - AST情報境界監査 PASS、フリーズ監査全5 initで PASS、副作用監査 PASS。
- 判定と影響:
  - 判定: `MULTI_INIT_VIABILITY_NOT_MET`。
  - I01 においては warm-start 効果が完全に再現されたが、初期重みの違いによって異なる局所幾何バイアス（I02のkey 5、I03のkey 3、I05のkey 1、I04の長さ8位置3干渉）が生じ、500ステップの一律 warm-start のみでは全初期化における大域的真のアトラクター収束を保証できないことが判明した。
  - 事前登録適格性基準に基づき、パイプラインはフェイルクローズ停止し、候補採択およびモデルバンドル作成は厳格に遮断を継続（`candidate_selected: null`, `child_bundle: null`, `bundle_write: false`）。
  - RG3、REC-005、G1、およびG4も未解除・遮断を継続。

## 7M. 実行契約: B-C005REC-004AR — CD-DPCA系列内非復元抽出warm-start 5初期化 matched-baseline 因果複製

**状態: 完了、`I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED`（ADR-0144）。matched-baseline 因果対照完了。REC-004AM の不合格判定は過去の科学的証拠として保持（遡及的変更なし）。候補採択・bundle出力・RG3・REC-005は厳格に遮断を継続。**
目的は、REC-004AMのwarm-start実行（I01..I05）を固定介入群とし、REC-004AL適格I01通常サンプラー実行を対照群として再利用した上で、未取得の初期化I02..I05についてのみ同一初期化・同一最適化・同一データシードのfresh baseline control（全6,000 updates通常復元抽出サンプラー）を実行し、全5対でmatched-baseline因果対照を実施することである。warm-start効果が普遍的優位性を持つか、初期化特異的／混合効果であるかを因果的に同定・判定する。

- 実行境界 & 厳格対照要件:
  - 操作因子はステップ1〜500のサンプラー（通常復元抽出 vs 非復元抽出）のみ。
  - 初期化重み・ハッシュ（I01..I05）、AdamW (`lr=0.0008, weight_decay=0.0001, grad_clip=1.0`)、CosineAnnealingLR (`T_max=1000, eta_min=1e-5`)、バッチサイズ32、6,000 updates、cadence 500（13 checkpoints）、per-step seed式 (`_derive_local_seed(20260912, step, 'train:MIRROR_HALVES')`)、1,024例開発検証セットを完全ビット一致。
  - 主判定規則: 4/5以上の初期化で全体EMおよび最悪位置精度が同方向に向上し、初期化間平均変化量が正 $\to$ `WARM_START_CAUSAL_SUPERIORITY_REPLICATED`。それ以外 $\to$ `I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED`。
  - レシピ変更、追加arm、追加初期化、ハイパーパラメータ探索、チェックポイント事後選別は禁止。
  - 副作用厳格遮断: candidate選定0 (`candidate_selected: null`), bundle write0 (`child_bundle: null`, `bundle_write: false`), RG3未実行 (`rg3: NOT_EXECUTED`), `rec005_eligible: false`。G1/G4は未解除の独立ブロック。
- 実行結果（`runs/phase_b_restart/rec004ar/run_001/`）:
  1. 主判定結果:
     - 同時改善初期化（全体EM向上 かつ 最悪位置精度向上）: **2 / 5**（`I01`, `I05`）。判定基準（$\ge 4/5$）未達。
     - 実行判定: `PASS`。
     - 主判定: **`I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED`**。
  2. マッチドペア詳細比較 ($\Delta = \text{Warm} - \text{Base}$):
     - `I01` (seed 20260912): Base EM `0.793945` $\to$ Warm `0.985352` ($\Delta = \mathbf{+0.191406}$); 最悪位置 `10:4 (0.3835)` $\to$ `6:2 (0.9608)` ($\Delta = \mathbf{+0.577289}$)。
     - `I02` (seed 20260913): Base EM `0.968750` $\to$ Warm `0.862305` ($\Delta = \mathbf{-0.106445}$); 最悪位置 `9:3 (0.9417)` $\to$ `10:4 (0.6068)` ($\Delta = \mathbf{-0.334951}$)。
       ※ ベースライン通常サンプラーでstep 500にて正解アトラクタ key 0 を自発獲得していたが、warm-start適用により位置4がkey 4へ逸脱し、最終的に対抗key 5偽アトラクタへ不可逆崩壊。
     - `I03` (seed 20260914): Base EM `0.992188` $\to$ Warm `0.962891` ($\Delta = \mathbf{-0.029297}$); 最悪位置 `10:4 (0.9757)` $\to$ `10:4 (0.8689)` ($\Delta = \mathbf{-0.106796}$)。
     - `I04` (seed 20260915): Base EM `0.965820` $\to$ Warm `0.965820` ($\Delta = \mathbf{+0.000000}$); 最悪位置 `8:3 (0.9588)` $\to$ `8:3 (0.8711)` ($\Delta = \mathbf{-0.087629}$)。
     - `I05` (seed 20260916): Base EM `0.939453` $\to$ Warm `0.974609` ($\Delta = \mathbf{+0.035156}$); 最悪位置 `10:3 (0.8932)` $\to$ `7:2 (0.9626)` ($\Delta = \mathbf{+0.069413}$)。
  3. 全5対要約統計:
     - 全体系列EM変化: 平均 $+0.018164$, 中央値 $+0.000000$, 範囲 $0.297852$ ([-0.106445, +0.191406]), 改善数 2/5 (40.0%)
     - 最悪位置精度変化: 平均 $+0.023465$, 中央値 $-0.087629$, 範囲 $0.912241$ ([-0.334951, +0.577289]), 改善数 2/5 (40.0%)
     - 長さ10系列EM変化: 平均 $+0.072816$, 中央値 $+0.048544$, 改善数 3/5 (60.0%)
     - 位置4トークン精度変化: 平均 $+0.044660$, 中央値 $+0.029126$, 改善数 3/5 (60.0%)
     - 位置4マージン変化: 平均 $+6.659074$, 中央値 $+4.921659$, 改善数 4/5 (80.0%)
     - 因果ギャップ変化: 平均 $+0.018164$, 中央値 $+0.000000$, 改善数 2/5 (40.0%)
  4. アトラクタダイナミクスと機構解明:
     - ベースライン通常サンプラー自体が2/5のseed（I02, I05）においてstep 500で自発的に正解アトラクタ key 0 を獲得可能であり、初期トークン重複が全初期化幾何において致命的障壁になるわけではないことを実証。
     - warm-startはI01で顕著な改善（+0.1914 EM, +0.5773 worst-pos acc）をもたらす一方、I02等では逆に破壊的干渉を生じさせ（-0.1064 EM, -0.3350 worst-pos acc）、普遍的優位性は反証された。
- 判定と影響:
  - 判定: `I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED`。
  - REC-004AMの不合格判定（`MULTI_INIT_VIABILITY_NOT_MET`）は科学的証拠として確定・維持され、遡及的変更は行わない。
  - 候補採択・bundle出力・RG3・REC-005は厳格に遮断を継続（`candidate_selected: null`, `child_bundle: null`, `bundle_write: false`）。
  - RG3、REC-005、G1、およびG4も未解除・遮断を継続。

## 7N. 実行契約: B-C005REC-004AS — CD-DPCA Initialization-Geometry × Early-Data Basin Susceptibility Audit

**状態: 完了、`INITIALIZATION_DATA_INTERACTION_IDENTIFIED`（ADR-0145）。post-mortemのみ。学習、optimizer update、新seed、candidate採択、bundle write、RG3、REC-005、sealed accessはすべて0/未実行。**

- 既存REC-004AL/AM/ARのみを読み、I01..I05のbaseline/warm-start各13 checkpoint（計130状態）と、長さ6..10・全有効output positionの40 routing cellを固定metricsで監査した。correct mapはpost-forward metricと`M=S(correct)-max S(wrong)`のdirectional derivativeだけに用い、初期化・sampler・loss・repairへは渡していない。
- step-0幾何単独ではmatched armの異なるterminal basinを決定できず、初期vulnerabilityも5 seed一貫でない。step-0→500のarm分岐cell数はI01..I05で6, 9, 5, 5, 9。I02 `10:4`は同一step-0 top-1 key 6から、baselineがstep500/terminalともcorrect key 0、warm-startがstep500 key 4、terminal key 5へ遷移した。
- 固定step-1 CEの全cell `-∇M·∇L` はsampler差で符号が変わるcellをI01..I05で10, 13, 18, 19, 12確認。I02 `10:4`はbaseline `-0.05480`、warm-start `-0.01970`（差`+0.03510`）で、その後のopposite basin transitionと整合する。sampler作用はseed一方向ではない。
- native gradient寄与と、I02 base←I01 donorのforward-only一群移植をquery/key-position、length、Wq、Wk、QK biasへ分解。top-1変化cellは23, 37, 32, 34, 35, 0/40で単一familyへの局在は不成立。generic oracle-free initialization constraintを一意に導けないため、initialization-only learning pilotとarchitecture/init sweepをSTOPする。
- 証拠: `runs/phase_b_restart/rec004as/run_001/`。source checkpoint SHA-256前後一致、`optimizer_updates=0`, `candidate_selected=null`, `child_bundle=null`, `bundle_write=false`, `RG3=NOT_EXECUTED`, `REC-005=BLOCKED`, sealed access=0。

## 7O. 実行契約: B-C005REC-004AT — Oracle-Free Standard-CE Optimization-Formulation Identifiability Audit

**状態: 完了、`NO_SINGLE_OPTIMIZATION_FORMULATION_IDENTIFIED_STOP`（ADR-0146）。artifact-only監査は実行PASSだが、CE-only optimization-repair方向はfail-closed STOP。学習、optimizer update、新seed、sampler/init/architecture変更、candidate採択、bundle write、RG3、REC-005、sealed accessはすべて0/未実行。**

- ADR-0138〜0145とAO/AP/AM/AR/AS artifactだけを固定入力にした。式の導出前に許可した情報はstandard token-output CE、model outputs、入力系列のoracle-free統計、per-example CE gradientsのみ。correct-key mapはpost-forward/backwardのmargin方向評価に限定し、式、loss、sampler、初期化には渡していない。
- ASで固定されたI01..I05、両stream、全40 routing cellのstep-0/early forward/autograd記録を監査した。stream間の`-∇M·∇L`符号変化はI01..I05で10, 13, 18, 19, 12 cell、step-0→500 basin分岐は6, 9, 5, 5, 9 cellであり、5-seed不変のoracle-free方向は存在しない。ARのmatched結果も同時改善2/5（I01, I05）、I02/I03負干渉である。
- collision audit: APのI01・長さ10・出力位置4・step500 Stratum Cは27/206例（13.1%）でtarget tokenがkey 0とkey 7の双方にある。同一input/target/logit/CE/CE-gradientという許容observableの下で、key-0 margin改善はkey 7を相対的に下げる必要がある一方、key-7 margin改善はkey 7を上げる必要があり、方向が反対である。key 0を選ぶ情報はcorrect-key mapであり、導出に使えない。実測`dL/dS(4,7)=-0.00471519`は`dL/dS(4,0)=-0.00002454`より約192倍で、CE descentのkey-0 margin予測は`-0.00469064`、Stratum C parameter-margin変化は`-31.0566`（負96.3%）。
- 従って係数探索・候補比較なしに一意なfixed optimization formulationは導けない。式を固定せず、条件付きの全40 cell（initial vulnerable、terminal failure、stable controlを含む）margin-directed no-update検証は`NOT_EXECUTED_PRECONDITION_UNMET`、将来6000-step I01..I05 pilotの事前登録は不可。証拠: `runs/phase_b_restart/rec004at/run_001/`。既存source manifestのSHA一致を継承し、`optimizer_updates=0`, `candidate_selected=null`, `child_bundle=null`, `bundle_write=false`, `RG3=NOT_EXECUTED`, `REC-005=BLOCKED`, sealed access=0。
## 7P. Execution Contract: B-C005R3-002R — Single-Family G1 Strict-Holdout Feasibility

**Status: complete, `G1_RELATION_TRANSFER_STOP` (ADR-0147).** The task preregistered exactly one parent Phase-B family, `sealed_local_neighborhood`, before outcome measurement; no alternative family, coefficient, production model, REC-004 artifact, candidate, bundle, RG3, REC-005, G4, or sealed model output was used.

- B-C002 controls passed for all three family operations: deterministic oracle and same-length checks, explicit/few-shot identifiability, and existing-bank/depth-2 symbolic novelty validity (`best direct EM=0.0`, `best depth-2 EM=0.0`, `tau=0.90`).
- Conservative relation coupling merges the family members into one clean, alias-free component; it does not count differently named members as independent relation-transfer evidence without a learned-parameter boundary. The resulting counts are validation `0/2` and sealed_v2 `1/2`, so relation-count sufficiency fails.
- On a development-seed-10 throwaway router, existing full-class CE gave held-out keys max gradients `0.04504218` and `0.02629104`, update/state drift; relation-scoped CE gave held-out max gradients exactly `0.0`, exact no-update and no optimizer state, while in-scope keys retained nonzero gradients `0.17509708` and `0.17509702`.
- Evidence: `runs/phase_b_b2_post_d2/r3_002r_single_family_g1/run_002/`. `run_001` is preserved preliminary evidence; it is not overwritten. Final execution boundary: production model training/modification `0`; sealed model outputs inspected `0`; throwaway-router setup updates `1`, denominator comparison updates `2`; `candidate_selected=null`, `child_bundle=null`, `bundle_write=false`, `rg3=NOT_EXECUTED`, `rec005=BLOCKED`.
- Decision: STOP G1 relation transfer. Do not add or compare a second family within this task; all downstream REC-004/RG3/REC-005/G4 blocks remain independently blocked.
## 8. 実行記録
- 2026-09-12: 本計画へ状態を集約。過去文書を仕様/証拠へ位置づけ直した（ADR-0127）。

- 2026-09-12: REC-004AC metric_v2を開始。過去の監査文書・変更途中のファイルを保持。
- 2026-09-12: metric_v2 `run_003` PASS。保存語彙を固定したruntimeでも、元の全manifest・EM・重みhashが一致（ADR-0128）。通常EM=0.7646484375、length10 J0=0.080078125、O1=1.0。RG3は未実行。
- 2026-09-12: `mirror_score_scale_precheck/run_001` 実行完了、研究判定FAIL_STOP（ADR-0129）。4条件固定、学習0、候補選択0。
- 2026-09-12: REC-004AE `run_004` 実行完了、`INSUFFICIENT_EVIDENCE_STOP`（ADR-0130）。length10の誤りはoracleで回復するrouting誤りだが、QKとposition biasを単一標的へ分離できなかった。修復学習・係数探索・RG3・sealedは0。
- 2026-09-12: REC-004AF qualified `run_005` 実行完了、`INSUFFICIENT_EVIDENCE_STOP`（ADR-0131）。4固定matched controlsによるQK endpoint置換は一部direct-error tokenを回復したが、routing/marginの非退化改善を示さず、同層のposition endpointは例間不変であった。repair target、recipe、学習、RG3、sealedは0。
- 2026-09-12: REC-004AG `run_001` 実行完了、`POSITION_ROUTING_TARGET_NOT_SUPPORTED`（ADR-0132）。長さ9からの非退化位置transportはEM=0へ崩壊し、位置bias単独標的は反証された。
- 2026-09-12: REC-004AH `run_001` 実行完了、`SCORE_DECOMPOSITION_IDENTIFIABILITY_STOP`（ADR-0133）。計算グラフと識別性行列の評価で全要素が条件(d)不成立。現行score分解における単一修復標的の探索を停止。
- 2026-09-12: REC-004AI `run_001` 実行完了、`MINIMAL_ROUTING_CONTRACT_IDENTIFIED`（ADR-0134）。タスク意味論から探索なしに単一最小アーキテクチャ契約（CD-DPCA）を導出。学習0、実装0、candidate0、RG3未実行。
- 2026-09-12: REC-004AJ `run_001` 実行完了、`MINIMAL_ROUTING_CONTRACT_IMPLEMENTED_AND_VALIDATED`（ADR-0135）。CD-DPCAのコード実装と未学習構造7基準PASS。学習0、candidate0、bundle0、RG3未実行。
- 2026-09-12: REC-004AK `run_001` 実行完了、`CD_DPCA_SERIALIZATION_AND_FRESH_LOAD_VALIDATED`（ADR-0136）。CD-DPCAおよびPrimitiveBankの厳格シリアライズ、fresh-load完全等価性、10種負例fail-closed、情報境界を検証。学習0、candidate0、bundle0、RG3未実行。
- 2026-09-12: REC-004AL `run_001` 実行完了、`PILOT_TERMINAL_VIABILITY_NOT_MET`（ADR-0137）。単一init（I01）学習パイロットで検証系列EM=0.793945（813/1024）となり、終端実行性基準（EM≥0.95）未達によりフェイルクローズ停止。全init検証・候補採択・bundle出力は厳格に遮断。G1/G4ブロック保持。
- 2026-09-13: REC-004AN `run_001` 実行完了、`LENGTH10_LOCAL_OPTIMIZATION_FAILURE_IDENTIFIED`（ADR-0138）。13 checkpoint全軌道再評価、位置局在、transplant、勾配衝突、露出監査を実施。失点の89.4%（位置3-4合計で99.3%）が出力位置4の偽アトラクタ固着（key 7）に局在することを確認。追加学習0、candidate0、bundle0、RG3未実行。
- 2026-09-13: REC-004AO `run_001` 実行完了、`LOCAL_LOSS_GRADIENT_MISALIGNMENT_IDENTIFIED`（ADR-0139）。13 checkpoint勾配到達性診断を実施。ルーティング勾配ノルムは十分（0.2707）だが正解マージン予測変化は84.6%で非正（端末-0.6976、cos=-0.1116）、key 0への勾配はsoftmax飽和により極小飢餓。追加学習0、candidate0、bundle0、RG3未実行。
- 2026-09-13: REC-004AP `run_001` 実行完了、`TOKEN_ALIASING_DILUTION_AND_LATE_SATURATION_IDENTIFIED`（ADR-0140）。13 checkpointトークン識別性層別化勾配診断を実施。非重複/他重複例（86.9%）では正解改善勾配が供給される一方、key 7重複例（13.1%）の12.4倍巨大破壊勾配が全体を反転させ、端末でsoftmax飢餓が固定化するメカニズムを特定。追加学習0、candidate0、bundle0、RG3未実行。
- 2026-09-13: REC-004AQ `run_001` 実行完了、`WARM_START_PILOT_VIABILITY_MET`（ADR-0141）。初期500ステップの系列内非復元抽出warm-start単一レシピ因果パイロットを実行。決定ステップ6000で系列EM=0.985352（1009/1024例）、長さ10 EM=1.0000（206/206例）、位置4トークン精度=1.0000（206/206例、top-1 key 0, margin +5.38）を達成し、終端実行性基準（≥0.95）を大幅クリア。全init検証（REC-004AM）の検討が認可。候補採択・bundle出力・RG3・REC-005は未認可・遮断を継続。
- 2026-09-13: REC-004AM `run_001` 実行完了、`MULTI_INIT_VIABILITY_NOT_MET`（ADR-0143）。ADR-0141のwarm-start固定レシピで事前登録5初期化（I01..I05、総30,000 updates）を実行。I01はEM=0.9854で完全合格・再現したが、I02〜I05は局所偽アトラクタ固着や他位置回帰により不合格（合格1/5、平均系列EM=0.950195）。事前登録適格性規則（5/5合格）未達によりフェイルクローズ停止。候補採択・bundle出力・RG3・REC-005は厳格に遮断を継続。
- 2026-09-13: REC-004AR `run_001` 実行完了、`I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED`（ADR-0144）。REC-004AM warm-start介入群とI01..I05 matched baseline対照群（I02..I05 fresh baseline 4 runs, 24,000 updates）による因果複製を実施。同時改善は2/5（I01, I05）に留まり、I02/I03ではwarm-startが負干渉を引き起こすことを解明。普遍的優位性は反証され、REC-004AMのFAILを科学的証拠として保持。candidate0、bundle0、RG3未実行。
- 2026-09-12: 全2,514ケースの分割検証・ruff・mypy・文書/差分確認を完了。今回の整理・修正・有限precheckを閉じる。Phase B全体やRG3の完了ではない。
- 2026-09-13: REC-004AS `run_001` 実行完了、`INITIALIZATION_DATA_INTERACTION_IDENTIFIED`（ADR-0145）。既存130 checkpoint states、40 routing cells/初期化、固定step-1 standard/distinct streamのno-update監査で、step-0幾何単独とearly data単独は不十分、initialization×data interactionを確認。generic initialization-only repairは一意に未同定のためlearning pilot、architecture/init sweepをSTOP。source hash不変、optimizer update/candidate/bundle/RG3/sealedは0。

- 2026-09-13: REC-004AT `run_001` 実行完了、`NO_SINGLE_OPTIMIZATION_FORMULATION_IDENTIFIED_STOP`（ADR-0146）。ADR-0138〜0145とAO/AP/AM/AR/AS既存artifactだけを監査。Stratum C（27/206）の同一CE observableに対するkey0/key7反対routing方向collision、全40 cellの5-seed early-sign/basin不一致、ARのmixed causal effectにより、係数探索・候補比較なしの固定standard-CE最適化式は一意に導出不能。CE-only optimization-repairをSTOPし、formula固定・全cell条件検証・6000-step pilotは未実行。optimizer update/new seed/training/sampler/init/architecture/candidate/bundle/RG3/sealedは0。
## 9. 歴史記録: 当時の検証と停止点

本節の検証は過去runの記録。closeoutでpytest等を再実行した記録ではない。現在の終端は§1を参照。

| 検証 | 結果 | 証拠・制約 |
|---|---|---|
| pytest 全収集ケース | 分割実行で2,514件PASS | Windows先行977件＋再開1,536件＋WSL1件。全node IDの和集合と全収集IDが一致 |
| REC-004AJ〜AR 回帰テスト | 62件PASS | REC-004AJ/AK/AL/AN/AO/AP/AQ/AM/ARの全62件PASS |
| Windows単一プロセスの全件実行 | 異常終了、PASSではない | 長いXデータ検証中のPythonアクセス違反。原因未確定。既通過分を保存し、残りを再開 |
| ruff check . | PASS | 終了コード0（全ファイル通過） |
| mypy src/apc | PASS | 169 source files、終了コード0 |
| 文書・差分 | PASS | ローカルリンク存在確認、git diff --check |

全ケースの検証範囲は満たしたが、単一プロセスの安定性を認定したとは扱わない。
WindowsはPython 3.12.13、WSLは3.12.14。Windows絶対パスの既存bundleはnative環境で読み、
WSLへ移したのはartifactパスに依存しない `test_rec004x_dataset_disjointness` 1件だけ。
残りの再開実行は全モジュールを収集したうえで実施し、操作登録数が増える条件も保持した。
チェックログと全ID台帳は `runs/phase_b_restart/verification_coverage_result.json`、
`verification_coverage_plan.json`、`verification_events.jsonl`、`final_*.log` に保存した。
研究の新学習0とは§5/6の研究実行を指し、検証用tiny fixtureの学習を含む全テストの更新数を指さない。

**この時点の停止点はREC-004ATによるoracle-free standard-CE optimization-formulation識別性STOP（`NO_SINGLE_OPTIMIZATION_FORMULATION_IDENTIFIED_STOP`）、REC-004ASのinteraction、REC-004ARのmixed effect、およびREC-004AMの不合格（`MULTI_INIT_VIABILITY_NOT_MET`）によるフェイルクローズ停止である。initialization-only pilot、architecture/init sweep、CE-only optimization-repairはSTOPし、candidate adoption、bundle promotion、RG3、REC-005、G1、G4はBLOCKEDを維持する。**
REC-004AMのwarm-start群（I01..I05）と厳格に対照されたベースライン群（I01..I05）の全5対比較により、warm-startが全体EMおよび最悪位置精度を同時に改善したのは 2 / 5 初期化（I01: $\Delta\text{EM}=+0.1914$, I05: $\Delta\text{EM}=+0.0352$）に留まり、I02（$\Delta\text{EM}=-0.1064$, $\Delta\text{WorstPosAcc}=-0.3350$）やI03（$\Delta\text{EM}=-0.0293$, $\Delta\text{WorstPosAcc}=-0.1068$）では逆に破壊的干渉を生じさせることが因果的に実証された。
また、ベースラインの通常復元抽出サンプラー自体が2/5の初期化（I02, I05）においてステップ500で自発的に正解アトラクタ key 0 を獲得しており、トークン重複は全初期化幾何において普遍的な致命障壁ではないことが判明した。
この結果、普遍的因果優位性は反証され、主判定 `I01_SPECIFIC_OR_MIXED_EFFECT_IDENTIFIED` が確定した。REC-004AMの不合格判定（`MULTI_INIT_VIABILITY_NOT_MET`）は科学的証拠として確定・維持される。
事前登録契約に基づき、候補採択（candidate adoption）、モデルバンドル出力（child bundle write）、15 non-SHIFT RG3 再検査、および 5モデル cohort REC-005 は依然として未認可であり、厳格に遮断を継続（`candidate_selected: null`, `child_bundle: null`, `bundle_write: false`, `rg3: NOT_EXECUTED`, `rec005_eligible: false`）する。独立した研究ブロック G1 および G4 も未解除のまま保持される。



## 10. PHASE-B-FINAL — Falsification Sufficiency & Research-Termination Audit

**状態: 完了。`NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`（ADR-0148）。**

本節は、ADR-0074〜0147、全active execution plan、および既存run manifest/reportだけを用いた証拠台帳監査である。新規training、optimizer update、seed、relation family、repair candidate、sealed-data/model-output access はすべて0である。完全な台帳、7項目の評価、未実行gateの依存判定、および読んだartifactのpathは[`docs/results/PHASE_B_FINAL_FALSIFICATION_SUFFICIENCY_AUDIT.md`](../../results/PHASE_B_FINAL_FALSIFICATION_SUFFICIENCY_AUDIT.md)に保存する。

- **結論の範囲:** 明示TaskSpecのB1 PASSは保持する一方、hard-negative B2/re-gate FAIL、CD-DPCAの5初期化適格性FAIL（1/5）、matched causal効果の混合（同時改善2/5）、oracle-free CE-only repairの識別不能STOP、およびG1 relation-count STOPにより、現行Phase-B architectureは主張したopen-world/semantic task-inference lifecycleを支持しない。これはAPC一般、異なるrelation registry、または将来の別設計の不可能性を主張しない。
- **未実行gate判定:** RG3、REC-005/RG4、REC-006〜008/RG5〜RG6、G4、R3-011、R3-012/G5、B-C006〜014は、candidate/bundle資格又はG1/G4/B2/G5を前提とするため、上流STOPにより科学的に不要かつ現行契約・本監査境界では不可能である。結論を変え得る、独立して実行可能な既存契約内の実験は0件である。
- **維持するblock:** `candidate_selected: null`、`child_bundle: null`、`bundle_write: false`、`rg3: NOT_EXECUTED`、`rec005: BLOCKED`、G4/G5 `BLOCKED`、sealed access=0。G1は`G1_RELATION_TRANSFER_STOP`のまま維持する。
- **終了条件:** 本監査は研究終了の証拠記録であり、既存FAILをPASSへ変更せず、STOPを迂回しない。新しい研究を再開するには、本結論を上書きしない別の明示的な研究契約と事前承認が必要である。

## 11. PHASE-B-CLOSEOUT

ADR-0149により証拠台帳・hash manifest・検証記録を固定し、Phase Bをarchiveした。
科学的結果はADR-0148のnegative terminationのまま。復旧/研究gateをPASSへ変えない。
[最終証拠台帳](../../results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md)、
[freeze manifest](../../results/PHASE_B_CLOSEOUT_FREEZE_MANIFEST.json)、
[closeout検証](../../results/PHASE_B_CLOSEOUT_AUDIT.json)が成果物。
独立したPhase C charterはレビュー可能な状態であり、承認済み・実行可能という意味ではない。
architecture/optimizer実装、学習、candidate、dataset/relation生成、seed、pilot、sealed評価は0。

## 12. PHASE-C-TERMINATION — C-D001AA Falsification Sufficiency & Charter Termination Audit

**状態: 完了。`PHASE_C_CURRENT_CHARTER_FALSIFICATION_SUFFICIENT`（ADR-0160）。
Phase C charter status: `TERMINATED_CURRENT_CHARTER`。**

独立したPhase C charter（`READY_FOR_REVIEW_NOT_APPROVED`のまま一度も承認されず、
研究実行も一度も認可されなかった）は、ADR-0150〜0159のpre-execution数学的レビューにより
以下の順で完全にfalsified/retractedされた: Contract v1 falsified (ADR-0151) → 記述子による
identifiability回復 (ADR-0153) → 同じ記述子がH-C1のestimandを0-parameter deterministic
reductionへtrivialize (ADR-0157) → continuous groundingも$B_{\text{det\_emb}}$に支配され
trivial (ADR-0158) → blind manifold/codebook residualもImpossibility-Dominance Dilemmaで
identifiability impossibilityとdeterministic dominanceの二分法に閉じ、`H-C1-Residual`が
formally retracted (ADR-0159)。

C-D001AA（[監査文書](../../phase_c/PHASE_C_C_D001AA_FALSIFICATION_SUFFICIENCY_AND_CHARTER_TERMINATION_AUDIT.md)）
はこの記録を新しいrepair・architecture・training contract・residual hypothesisを一切考案せずに
監査し、以下を確認した:
- ADR-0152とADR-0154はそれぞれ後続ADR-0153・ADR-0155により明示的にretractedされている。
- 現在のoracle boundaryはADR-0155/0156の5次元criterionのまま、ADR-0159まで無修正で維持。
- descriptor-based identifiabilityはH-C1を検証したのではなく、estimandを置き換えた
  ($H(Z\mid X,D)=0$になった時点でlearning問題が消滅する)。
- impossibility claimの範囲は監査済みのrepresentation/group-symmetry classに厳密限定され、
  APC一般の不可能性、あらゆるtask information contractの無意味化、nonlinear/stochastic/
  interactive/future formulationsの不可能性、routing-based architecture一般の不可能性は
  一切主張しない。
- 現在のcharter内に結論を変え得る独立タスクは残っていない（C-D002・architecture実装・
  optimizer trial・追加seed・新relation family・新dataset・coefficient search・descriptor
  variant・anchor count variant・sealed evaluationのいずれも`OUTSIDE_CURRENT_CHARTER`）。
- relation inventory不足（validation 1/2、sealed 1/2、ADR-0150）は独立の必要条件FAILとして
  維持されるが、これを満たしてもH-C1/residualが自動復活するわけではない。
- charter threshold・oracle boundary・initialization数・relation要件はいずれも緩和していない。
- sealed access=0、candidate=null、C-D002=NOT_AUTHORIZEDを維持。

終了条件10項目（H-C1不成立、Contract v1 falsified、Contract v1.1がestimand trivialize、
continuous residual dominated、blind residual closed、lawful residualなし、relation
inventory独立FAIL、research execution未認可、sealed access=0、結論変更taskなし）を
すべて満たしたため、`PHASE_C_CURRENT_CHARTER_FALSIFICATION_SUFFICIENT`を宣言し、
Phase C charterを`TERMINATED_CURRENT_CHARTER`へ移行した。charterは
「original hypothesis保持・未承認・未実験実行・pre-execution数学的レビューでfalsified/
retracted」という状態のhistorical artifactとして保存される。

成果物: [終端監査文書](../../phase_c/PHASE_C_C_D001AA_FALSIFICATION_SUFFICIENCY_AND_CHARTER_TERMINATION_AUDIT.md)、
[termination evidence ledger](../../results/PHASE_C_TERMINATION_EVIDENCE_LEDGER.md)、
[termination audit record](../../results/PHASE_C_TERMINATION_AUDIT.json)。
本節のためのarchitecture/optimizer実装、学習、candidate作成、dataset/relation生成、
seed実行、pilot、sealed評価はすべて0。このtask内でPhase D相当のhypothesisは一切作成していない。
次の研究課題を検討する場合は、別taskで独立した`NEXT-RESEARCH-QUESTION REVIEW`を開始すること。

## 13. NRQ-001 — Next-Research-Question Review 完了（`NO_NONTRIVIAL_ESTIMAND_IDENTIFIED`）

2026-09-13、上記で予告された独立review `NRQ-001` を実施した（Phase Cタスクではない）。
ADR-0159のImpossibility-Dominance Dilemmaをrepresentation非依存の一般形へ拡張し
（nonlinear/interactive再定式化も同じdichotomyへ吸収）、6件の候補estimandを検査したが、
non-trivial・identifiable・oracle-free・relation-transfer両立・APC中核分離の実検証という
5条件を同時に満たす候補は0件だった。relation inventory不足（validation 1/2、sealed 1/2）は
ADR-0147・ADR-0150の独立2経路で再確認済みのまま未解消。

結論として、oracle-freeなsemantic/relation task inference研究系列（Phase B→Phase Cの本流）は
program levelでclosure確定（`PROGRAM_LINE_CLOSURE_CONFIRMED`）とした。これはAPC一般の不可能性
主張ではなく、Phase A/A1/A2のexplicit TaskSpec下core separation実証結果を覆すものでもない。
新規学習・候補選択・sealed評価・relation登録はすべて0。詳細は
[review文書](../../research/NEXT_RESEARCH_QUESTION_REVIEW_NRQ001.md)と
[ADR-0161](../../DECISIONS_PHASE_C.md#adr-0161-nrq-001-next-research-question-review-finds-no-non-trivial-identifiable-estimand-no_nontrivial_estimand_identified)を参照。
