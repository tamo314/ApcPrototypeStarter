# R-CNP-001M 親正解decision保持診断の結果

日付: 2026-09-22 / 実行commit: `e660633`

状態: **`M_DIAGNOSTIC_FAIL_STOP`**。候補の`PARENT_CORRECT_DECISION` armは、登録した
all-five candidate gateを0/5で満たせなかった。係数・marginの変更、候補選択、promotion、
sealed access、R-CNP-002は実行しない。

| arm | new品質 failure cells（5親×120） | old品質 failure cells | old保持 failure cells | candidate gate |
|---|---:|---:|---:|---:|
| `TASK_ONLY` | 400 | 336 | 430 | 0/5 PASS |
| `PARENT_CORRECT_DECISION` | 305 | 107 | 220 | 0/5 PASS |

候補は全5親でnew/old/retentionの各gateを依然FAILした。decision制約は失敗セル数を
newで95、oldで229、retentionで210減らしたが、これはgate PASSを意味しない。5親の候補の
new品質failure数は順に65、59、58、61、62、old品質は21、27、9、25、25、保持は36、57、23、33、71。

データ/手続きinvariantは全armでPASSした。fresh root 620310のsplit/support監査、初期adapter
parity、frozen base hash、10 candidate checkpointのfresh-process parity、update-eligible
parameter数1,024はすべてPASS。candidate selection=0、promotion=`NOT_AUTHORIZED`、sealed
access=0を維持した。

実行artifact: `runs/cnp_repair/r001m/rcnp001m_decision1/`。wall 709.830秒、peak CUDA
allocated 67,965,440 bytes、reserved 69,206,016 bytes、process RAM 2,608,521,216 bytesで、
登録した7,200秒、12GiB、32GiBの上限内だった。

R-CNP-001Dが示したMSE保持との更新方向衝突を避けるという機序は、このzero-margin
decision制約の記述的改善と整合する。しかし、全セルのnew品質を回復できなかったため、
本固定レシピは固定親診断でのviable候補として棄却する。追加の目的関数、margin、係数、
capacity、scheduleの探索はこの診断からは開始しない。
