# Phase B 再開計画 — 現在地と実行順の正本

版: 2026-09-12 / ADR-0127〜0129。

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
| 再開: AC metric_v2 | 訂正・再分析・分割全件検証完了（§8） | 計測/再現PASS、性能FAILを保持 | 正しいkey指標、保存語彙の固定 |
| 再開: score-scale precheck | 完了 | 固定4条件すべてFAIL_STOP | 全体的な拡縮だけでは当該endpointを回復できなかった |
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

**状態: 訂正・再分析PASS、分割全件検証完了（§8）。新学習0、候補選択0、sealed0。**
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

**状態: 完了・FAIL_STOP（ADR-0129）。並行していた全件検証も完了（§8）。**
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

## 7. 実行記録

- 2026-09-12: 本計画へ状態を集約。過去文書を仕様/証拠へ位置づけ直した（ADR-0127）。
- 2026-09-12: REC-004AC metric_v2を開始。過去の監査文書・変更途中のファイルを保持。
- 2026-09-12: metric_v2 `run_003` PASS。保存語彙を固定したruntimeでも、元の全manifest・EM・重みhashが一致（ADR-0128）。通常EM=0.7646484375、length10 J0=0.080078125、O1=1.0。RG3は未実行。
- 2026-09-12: `mirror_score_scale_precheck/run_001` 実行完了、研究判定FAIL_STOP（ADR-0129）。4条件固定、学習0、候補選択0。
- 2026-09-12: 全2,514ケースの分割検証・ruff・mypy・文書/差分確認を完了。今回の整理・修正・有限precheckを閉じる。Phase B全体やRG3の完了ではない。

## 8. 最終検証と現在の停止点

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

**現在の停止点はMIRROR通常実行の研究性能。** 計測と親ロードの障害は修正済み。
今回のglobal score拡縮による修復は失敗として終了し、追加係数・学習・全init展開・RG3・REC-005は実行しない。
次に修復を再開する契約には、正しいkeyの順位を変えられる機構と、動作しているvalue経路を保持する対照を
明示する必要がある。有限pilotの成功から全init、固定I01採用、full-bundle RG3へ接続する条件も事前固定する。
これは新しい連番タスクの予約ではなく、次の実行判断に必要な最小条件である。G1/G4は別の未解決条件として保持する。
