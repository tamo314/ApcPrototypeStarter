# Phase B 再開計画 — 現在地と実行順の正本

版: 2026-09-12 / ADR-0127〜0133。

## 1. 目的と今回の実行権限

ユーザーの「現状を整理して計画を組みなおし、その後、修正したPhase Bを進行する」という
2026-09-12の指示に基づく。今後の進捗・依存関係・実行待ちはこの文書だけで管理する。
既存のtask ID、実験結果、受入閾値、sealedデータの境界は保持する。
旧文書の「proposed」「最初はREC-001のみ」等は作成当時の開始状態であり、現在地ではない。

この指示は、以下の順序で前提が成立した作業を進める許可として扱う。単なるtask完了ごとの
再承認は求めない。一方、失敗STOP GATEを飛ばす許可ではない。新しい研究介入は、
仮説・変更範囲・データ・予算・成功条件を本書へ**実行前に**固定してから行う。
性能未達に応じた無制限な実験追加、過去FAILの上書き、sealedを使った調整はしない。

## 2. 文書の役割

| 文書 | 今後の役割 |
|---|---|
| 本書 | 現在地、優先順位、依存関係、再開作業の契約、最新の完了/停止判断 |
| [Phase B親計画](PHASE_B_SEMANTIC_TASK_INFERENCE_OPEN_WORLD.md) | B1〜B6の研究目的・最終ゲートの原契約 |
| [Post-D2計画](PHASE_B_B2_POST_D2_REPAIR.md) | R3の技術仕様・G0〜G5の原契約 |
| [Recovery計画](PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md) | REC-001〜008の技術仕様・RG0〜RG6の原契約 |
| [監査報告](../../research/PHASE_B_BLOCKER_AUDIT_2026_09_12.md) | 2026-09-12時点の履歴・不具合の証拠。後続実行結果で書き換えない |
| [Decision index](../../DECISIONS.md) | 判断の索引。科学的解釈の変更は新ADRへ追記 |
| 過去REC-004A〜ACの契約・run | 完了済みの診断/介入の記録。自動再実行する待ち行列ではない |

ファイルの削除・移動や過去番号の振り直しは行わない。細分化されたタスクをさらに積み上げず、
既存タスクの訂正はrevisionとして扱い、以下の一つの未解決問題に結び付ける。

## 3. 確定した現在地

| 項目 | 作業の状態 | 科学的/復旧判定 | 再利用する成果 |
|---|---|---|---|
| B-C001〜003 | 完了 | B1 PASS（明示TaskSpec条件） | 未見familyでのlifecycle |
| B-C004〜005、D/R1/R2/G、D2 | 完了・診断系列は終了 | B2/re-gate FAIL | ranking/argument/adequacyの分離 |
| R3-001/003/005 | 完了 | G0/G2/G3 PASS | 再現可能な生成、指標、有限look verifier |
| R3-002 | 調査完了 | G1 relation不足 | exposure台帳・不足の証拠 |
| R3-006〜008 | 過去親上で完了 | 新しい親では再資格確認が必要 | scoped key、SELECT式修正、BIND coverage |
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
| REC-005〜008 | 未着手 | RG3に依存。ただし失敗時の引継ぎは可能 | 今後のcohort・runtime注入の原契約 |
| R3-011〜012 | 未着手 | G1/G4に依存 | 封印・B2_PROTOCOL_V2原契約 |
| B-C006〜014 | 未着手 | B2/G5、以降各gateに依存 | B3〜B6の原計画 |

**未解決問題は三つ:** (a) MIRRORを含む整合bundleの通常実行性能、
(b) 同一親での検索/引数/SHIFT/実K/C/N/R統合、(c) 独立relationの不足とholdoutへの学習露出。
I01〜I05は同一Core上の初期化であり、5個の独立modelではない。

## 4. 実行順と出口条件

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

## 8. 実行記録

- 2026-09-12: 本計画へ状態を集約。過去文書を仕様/証拠へ位置づけ直した（ADR-0127）。
- 2026-09-12: REC-004AC metric_v2を開始。過去の監査文書・変更途中のファイルを保持。
- 2026-09-12: metric_v2 `run_003` PASS。保存語彙を固定したruntimeでも、元の全manifest・EM・重みhashが一致（ADR-0128）。通常EM=0.7646484375、length10 J0=0.080078125、O1=1.0。RG3は未実行。
- 2026-09-12: `mirror_score_scale_precheck/run_001` 実行完了、研究判定FAIL_STOP（ADR-0129）。4条件固定、学習0、候補選択0。
- 2026-09-12: REC-004AE `run_004` 実行完了、`INSUFFICIENT_EVIDENCE_STOP`（ADR-0130）。length10の誤りはoracleで回復するrouting誤りだが、QKとposition biasを単一標的へ分離できなかった。修復学習・係数探索・RG3・sealedは0。
- 2026-09-12: REC-004AF qualified `run_005` 実行完了、`INSUFFICIENT_EVIDENCE_STOP`（ADR-0131）。4固定matched controlsによるQK endpoint置換は一部direct-error tokenを回復したが、routing/marginの非退化改善を示さず、同層のposition endpointは例間不変であった。repair target、recipe、学習、RG3、sealedは0。
- 2026-09-12: REC-004AG `run_001` 実行完了、`POSITION_ROUTING_TARGET_NOT_SUPPORTED`（ADR-0132）。長さ9からの非退化位置transportはEM=0へ崩壊し、位置bias単独標的は反証された。
- 2026-09-12: REC-004AH `run_001` 実行完了、`SCORE_DECOMPOSITION_IDENTIFIABILITY_STOP`（ADR-0133）。計算グラフと識別性行列の評価で全要素が条件(d)不成立。現行score分解における単一修復標的の探索を停止。
- 2026-09-12: 全2,514ケースの分割検証・ruff・mypy・文書/差分確認を完了。今回の整理・修正・有限precheckを閉じる。Phase B全体やRG3の完了ではない。

## 9. 最終検証と現在の停止点

| 検証 | 結果 | 証拠・制約 |
|---|---|---|
| pytest 全収集ケース | 分割実行で2,514件PASS | Windows先行977件＋再開1,536件＋WSL1件。全node IDの和集合と全収集IDが一致 |
| Windows単一プロセスの全件実行 | 異常終了、PASSではない | 長いXデータ検証中のPythonアクセス違反。原因未確定。既通過分を保存し、残りを再開 |
| ruff check . | PASS | 終了コード0 |
| mypy src/apc | PASS | 162 source files、終了コード0 |
| 文書・差分 | PASS | ローカルリンク225件の存在確認、git diff --check |

全ケースの検証範囲は満たしたが、単一プロセスの安定性を認定したとは扱わない。
WindowsはPython 3.12.13、WSLは3.12.14。Windows絶対パスの既存bundleはnative環境で読み、
WSLへ移したのはartifactパスに依存しない `test_rec004x_dataset_disjointness` 1件だけ。
残りの再開実行は全モジュールを収集したうえで実施し、操作登録数が増える条件も保持した。
チェックログと全ID台帳は `runs/phase_b_restart/verification_coverage_result.json`、
`verification_coverage_plan.json`、`verification_events.jsonl`、`final_*.log` に保存した。
研究の新学習0とは§5/6の研究実行を指し、検証用tiny fixtureの学習を含む全テストの更新数を指さない。

**現在の停止点はMIRROR通常実行の研究性能であり、REC-004AE/AF/AG/AHの診断・識別性評価系列は`SCORE_DECOMPOSITION_IDENTIFIABILITY_STOP`で閉じた。**
計測と親ロードの障害は修正済み。global score拡縮は失敗として終了し、AEはrouting誤りを局在化した。AFはQK endpoint置換の局所token感度を示したがrouting改善を満たさず、AGは位置transportの崩壊により位置bias単独標的を反証した。AHはこれらを統合した構造的識別性レビューを行い、現行の $S = S_{QK} + S_{\text{position\_bias}} + S_{\text{residual}}$ 分解では単一の修復標的を非退化・非破壊に分離できないことを確定させた。
追加係数・学習・全init展開・RG3・REC-005・sealed評価は実行しない。G1/G4は別の未解決条件として保持する。
