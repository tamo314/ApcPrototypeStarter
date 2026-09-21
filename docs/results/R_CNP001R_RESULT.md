# R-CNP-001R 機能保持診断

日付: 2026-09-22 / 実行commit: `80215a1`

状態: **`R_DIAGNOSTIC_FAIL_STOP`**。

独立rootのnew/old shadowと、R-CNP-001Sで固定したC scheduleを用いて、固定CNP-003親seed
610200–610204に対する`LAMBDA_0`と`LAMBDA_1`を各256 update実行した。lambda=1はreplayの
正解BCEを残しつつ、親replay logitへの集合等重みMSEを加えた。全10候補を保存したが、
lambda=1のcandidate gateは4/5 seedでFAILのため、登録済み診断はFAILである。

| arm | new品質失敗cell/seed | old品質失敗cell/seed | old保持失敗cell/seed | candidate gate |
|---|---:|---:|---:|---:|
| LAMBDA_0 | 72.2 | 52.6 | 73.0 | 0/5 PASS |
| LAMBDA_1 | 107.8 | 2.6 | 0.4 | 1/5 PASS |

lambda=1は旧条件の品質と保持を大幅に改善した一方、new品質を悪化させ、全5親の条件を
満たさない。固定係数1を別の値へ変更したり、成功seedのみで採択したりしない。これは固定v1親を
使う限定診断であり、修正版H-CNP2の確認、candidate selection、promotion、R-CNP-002へ進む根拠に
ならない。

実行artifact: `runs/cnp_repair/r001r/rcnp001r_retention1/`。S reportを明示provenanceとして
保存し、各候補のparent logit cache hash、全cell/集合証跡、schedule、candidate checkpointを保存した。
wall timeは1,970.313秒、peak CUDA allocationは67,965,440 bytes、peak process RAMは
2,803,081,216 bytesで、2時間、12GiB、32GiBの登録上限内だった。candidate selection=0、
promotion=`NOT_AUTHORIZED`、sealed access=0を維持する。
