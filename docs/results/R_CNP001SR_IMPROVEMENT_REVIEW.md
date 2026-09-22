# R-CNP-001S/R 結果確認と改善案

日付: 2026-09-22 / ADR-0205 / `ARTIFACT_REVIEW_COMPLETE_STOP_PRESERVED`

## 結論

**S/Rはいずれも0/5合格。順序改善だけでは不足し、固定lambda=1の保持制約は
旧条件の破壊を抑える一方、新条件の品質を全5親で悪化させた。**
次は、実行契約との不一致を修正して測定可能性を整え、保持制約が必要以上に
logitを固定しているか、new/replayの更新方向が衝突するかを分けて検討する。
係数の中間値、rank増加、学習延長の有効性は今回の結果からは分からない。

本レビューは保存JSONの算術再集計とコード読解のみ。追加model forward、データ生成、
optimizer update、候補採択、promotion、sealed accessは0。改善案は実行登録ではない。
S/Rの科学的FAIL、v1 G3 STOP、R-CNP-001 FAILを維持し、R-CNP-002へ自動継続しない。

根拠: [S結果](R_CNP001S_RESULT.md)、[R結果（誤記訂正）](R_CNP001R_RESULT.md)、
[再集計JSON](R_CNP001SR_IMPROVEMENT_AUDIT.json)、
[設計契約](../design-docs/CNP_ADAPTATION_SCHEDULE_AND_RETENTION.md)、
[実行計画](../exec-plans/active/CNP_ADAPTATION_SCHEDULE_AND_RETENTION.md)。

## 1. 判定の再確認

各new/old panelは120セル、各128集合。品質は全セルBA≥0.95かつ集合平均F1≥0.90、
保持は全旧セルで候補F1−親F1≥−0.01。表は5親の失敗セル数の平均であり、平均値のgateではない。

| 診断・方式 | new品質失敗 | old品質失敗 | old保持失敗 | candidate gate |
|---|---:|---:|---:|---:|
| S: A 順序固定 | 80.0 | 57.2 | 81.2 | 0/5 |
| S: B 同じ提示回数で分散 | 79.0 | 49.8 | 69.4 | 0/5 |
| S: C 提示回数も均等化 | 73.4 | 49.4 | 72.0 | 0/5 |
| R: C＋lambda=0 | 72.2 | 52.6 | 73.0 | 0/5 |
| R: C＋lambda=1 | 107.8 | 2.6 | 0.4 | **0/5** |

Rの結果文・計画・ADR-0204にあった「1/5 PASS」「4/5 FAIL」は記述の誤り。
保存`report.json`の全10候補は最初からFAILだった。lambda=1の内訳は以下。
報告文だけを訂正し、run、checkpoint、保存gate、過去ADRの本文は変更しない。
ADR-0204の誤記はADR-0205で明示的に訂正する。

| 親seed | new品質失敗 | old品質失敗 | old保持失敗 | gate |
|---|---:|---:|---:|---|
| 610200 | 108 | 2 | 0 | FAIL |
| 610201 | 108 | 3 | 1 | FAIL |
| 610202 | 104 | 2 | 1 | FAIL |
| 610203 | 109 | 3 | 0 | FAIL |
| 610204 | 110 | 3 | 0 | FAIL |

Python 3.12.13標準ライブラリで、25候補×3 panel×120セル×128集合の
**1,152,000保存集合行**（親・方式間の重複を含む）を検査した。
集合F1/EM、混同行列、全9,000セルsummary、失敗リスト、旧品質の遷移、candidate gateは
保存値と一致した。記録された基盤hashの一致、panel内の集合ID対応、A/Bのrecord別回数、
Cのquota、Rの両方式のscheduleファイル一致も確認した。
これはcheckpointの独立復元や、記録外の不変条件の実証ではない。

## 2. 効果と限界

### S: 順序改善は補助策

B−Aでは旧保持失敗が平均11.8セル減り、旧平均セルF1の親比低下は−0.021908から
−0.016051へ縮小した。new平均セルBA/F1が両方改善したのは3/5親であり、全親一様の改善ではない。
C−Bではnew品質失敗が平均5.6セル減る一方、旧保持失敗は2.6セル増えた。
従って「均等化すれば新旧とも改善する」とは言えない。

A/Bは同一record提示回数の対照なので、当該固定run内の順序・batch構成の差を観測できる。
独立world・複数schedule rootでの一般的な因果効果や、順序と提示回数の全交互作用は未証明。
以下の実行証跡不足もあるため、機序の確定とは扱わない。

### R: 保持は効いたが、可塑性との両立が未達

lambda=0→1でnew平均セルBAは0.923722→0.878786、F1は0.936883→0.880300。
両方とも5/5親で低下した。旧親合格→候補不合格のセルは平均50.8→0.0、
旧平均セルF1の親比変化は−0.018782→+0.000094となる。
一方、lambda=1のold品質失敗2.6セルはすべて親から継続した不合格であり、
親の不足と適応による追加破壊は区別できる。旧保持にはなお2親で各1セルの未達がある。

newの主な不足はBA。Rではlambda0/1とも品質失敗数とBA失敗数が一致する。
F1だけを目的に改善判定すると、この不足を隠す。

記録されたstep256のreplay-logit MSEはlambda=0で9.70–18.05、lambda=1で
0.00916–0.02433。同じR scheduleの終端batchで親出力への強い拘束と整合する。
ただしBCE/MSEの値の大小は勾配の大小ではなく、勾配衝突や係数過大の証明ではない。
固定panelの曲線がないため、未収束・過学習・容量不足も識別できない。

SとRはpanel/query/rootが異なる。旧親自体の失敗もS平均15.4セル、R平均3.0セルで異なるため、
SのCとRのlambda0の差を保持lossの効果として扱わない。比較は各診断内の対応するarm間に限る。

## 3. 実行契約と実装・保存証拠の差

`src/apc/cnp/repair_followup.py`とrun directoryの照合による。これらは科学的な性能FAILとは
別の検証不足であり、「実験全体の受入条件をすべて満たした」とは認定しない。

| 契約上の要求 | 現在の実装・記録 | 影響／次の実装で必要なこと |
|---|---|---|
| 共通train/replay全panelでstep0/16/64/256のloss | step16/64/256の当該batchの更新前lossだけ | 収束曲線に使えない。固定panel、new/replay別BCE、親正誤群を記録 |
| 親のnew/old初期評価 | 親oldのみ保存 | newへの親比改善量は復元不能。親newを事前記録 |
| 初期出力一致・必要な不変条件 | gateのinvariants入力は基盤before/after hash比較だけ | 初期parity・更新parameter数・分割・予算・復元を独立検証して合成 |
| 最終checkpointの別process復元 | runnerに呼出しがなく、run内にparity証跡なし | 契約指定のtrain/replay固定probeで検証し、shadowを再選択に使わない |
| 事前の全120セルsupport/正負例と正例率検査 | panel検査は学習後のgate。manifestは件数とPASSのみ | 学習前検査へ移す。モデル評価を使わず生成済み教師を検査 |
| 過去開封済みquery/recordを含む分離監査とmanifest | 4つの現行role内の入力/query重複を検査。履歴比較の呼出し・ID台帳なし | 既知履歴との比較、allowlist、hash付き台帳を保存。sealedは開かない |
| paired bootstrap CI | helperはあるがrunnerから未使用、CI保存なし | 宣言したF1/EM/BAの診断CIを実装。CIをgate救済に使わない |
| 詳細provenance、保持cache、方式別費用・容量 | config/親/cache hashとrun全体wall/CUDA allocated/RAMのみ。cache tensor未保存 | code/data/schedule lineage、cache実体又は再現仕様、CUDA reserved、方式別/共有費用、resident/active/temporaryを保存 |

記録されたS/Rの費用はそれぞれ372.672秒/1,970.313秒で、全体peakも登録上限未満。
ただしセル評価・保存を含む全工程での上限強制や、実行前の他process GPU空き検査は
runnerに実装されていない。run全体時間の差から保持lossの速度低下率を算出しない。

不足のある証跡を今回の文書だけでPASSへ補完しない。過去runは保存し、必要な追加検証は
範囲を定めた別作業とする。学習前に実施されなかった検査を事後に「実施済み」としない。

## 4. 改善の優先順位

1. **実行・計測を契約に合わせる。** 上表の不足を解消し、記録不足なら受入PASSを返さない。
   これは性能改善策と分離する。既存集合証跡の算術確認は今回完了した。
2. **保持制約の尺度と更新方向を測る。** 将来の限定診断では固定train/replay panel上で
   new BCE・replay BCE・keep MSE、各adapter勾配ノルムと内積、親正誤別のlogit変化を
   測る。現行のendpointは保存済みだが、中間checkpoint/step0の共通panel曲線はなく、
   過去の軌道全体は再構成できない。データ範囲・forward/backward予算を登録してから行う。
3. **第一候補は「logit値の完全な追従」から「旧判断を守る制約」への変更を検討する。**
   現行MSEは選択maskが同じでも確信度の変化を罰する。保持したいBA/F1と目的が一致するとは
   限らない。診断でこの過剰拘束を支持する場合、既存replay教師に対する符号付きmarginの
   下限制約を一案とする。親正解では必要なmarginまで守り、親誤例にはBCEによる修復を許す。
   親誤りを無条件に固定しない一方、replay上の制約は未見old queryの保持を保証しない。
   margin尺度・許容幅・目的式はtrain/development情報だけで事前固定する必要がある。
   この案の有効性は未検証であり、今回レシピとして採用していない。
4. **純粋な尺度問題なら、係数校正を別の有限開発として扱う。** lambda=0/1の二点から
   中間値に全gateを満たす解が存在するとは言えない。shadowで係数を探さず、開発予算と
   選択規則を事前固定する。更新方向の衝突が支配的なら単なる係数変更では不足し得る。
   まず一つの仮説・一つの変更に絞り、margin方式と係数変更を同時に投入しない。
5. **レシピ固定後に適格な修正版親で独立確認する。** 現在の親はold品質で全5親が不適格。
   初期学習から集合等重みlossを使う親のG1/G2/old品質を先に満たし、newを既に解く場合は
   `ADAPTATION_NOT_NEEDED`とする。FULL_REPLAY/LEARNED_METRIC対照、転移、両順序合成、
   全5seed、同等品質での費用比較を残す。親の再学習自体を忘却解消の証明にしない。

C scheduleは配分を固定する土台として残せるが、Sの成功レシピではない。
rank/層数/更新数の増加、replay比率の変更、oracle domainでのadapter切替は、
今回の証拠から最初の対策として導かれない。newとoldの両立が不可能という結論もまだ出せない。

## 5. 完了範囲と再現

入力は`runs/cnp_repair/r001s/rcnp001s_schedule3/report.json`と
`runs/cnp_repair/r001r/rcnp001r_retention1/report.json`。入力・schedule・読解対象codeの
SHA256、集合数、全seed別集計は再集計JSONに保存した。入力reportの処理前後hashは一致。
再集計scriptは`runs/cnp_repair/review_rcnp001sr_20260922/audit.py`、
stdoutは同directoryの`audit_stdout.log`。正式再集計は`.venv/Scripts/python.exe`の
Python 3.12.13で実行した（初回のキー/件数確認のみsystem Python 3.10.11）。

算術・保存gate一致、schedule検査、文書リンク・ADR番号・diff検査はPASS。
研究コード/config/checkpointの変更、追加実験はなし。文書・算術レビューのため
pytest全suite、ruff、mypyは未実行。研究実行の検証不足は§3に記録し、未実施をPASSとしない。
