# R-CNP-001D 保持勾配整合診断

日付: 2026-09-22 / 実行commit: `d885408`、preflight修正: `25def7c`、`3ab0192`

状態: **`ALIGNMENT_CONFLICT_SUPPORTED`**。更新数、candidate write、selection、promotion、
sealed accessはいずれも0。

| parent seed | virtual task/keep cosine | virtual keep MSE |
|---|---:|---:|
| 610200 | -0.989426 | 0.069050 |
| 610201 | -0.930395 | 0.027856 |
| 610202 | -0.790345 | 0.064074 |
| 610203 | -0.832915 | 0.021328 |
| 610204 | -0.956665 | 0.093794 |

すべて事前登録した`cosine <= -0.10`を満たした。zero adapterで保持MSE勾配が0となるため、
task方向をメモリ内で一時適用した後のfull panel勾配を測り、adapterを復元した。全parentで
初期logit/mask parityとbase hash rollbackがPASSした。

これはMSE保持項が初回task方向と衝突する機序を支持する。decision-margin方式、係数、
new/old品質、保持gateの回復は未検証であり、candidate採択やR-CNP-002を認可しない。

成果物: `runs/cnp_repair/r001d/rcnp001d_alignment4/`。wall 34.391秒、peak CUDA allocated
68,075,520 bytes、reserved 69,206,016 bytes、process RAM 1,913,335,808 bytes。
前段の`alignment1`〜`alignment3`はいずれも完了runではなく、保存済みpreflight/GPU境界の
停止証跡である。model forward前停止は1/2、alignment3は最初の勾配計算後・optimizer update前。
