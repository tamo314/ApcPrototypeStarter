# R-CNP-001S 順序・提示回数診断

日付: 2026-09-22 / 実行commit: `2af73a5`

状態: **`S_DIAGNOSTIC_FAIL_STOP`**。

後続の[結果レビュー（ADR-0205）](R_CNP001SR_IMPROVEMENT_REVIEW.md)で保存gateとの
算術一致を確認した。共通panel学習曲線・別process復元等の証跡不足は別途記録し、
本runを実行契約の全検証完了とは扱わない。

固定CNP-003親seed 610200–610204について、`q1+q2+`の修正版LOCALを
`ORDERED`、`DISPERSED_MATCHED`、`DISPERSED_BALANCED`で各256 update実行した。
全15候補がnew品質、old品質、またはold保持の少なくとも一つを満たさず、事前指定候補C
`DISPERSED_BALANCED`も5/5 FAILだった。候補選択=0、promotion=`NOT_AUTHORIZED`、
sealed access=0を維持する。

| arm | new品質失敗cell/seed | old品質失敗cell/seed | old保持失敗cell/seed | candidate gate |
|---|---:|---:|---:|---:|
| ORDERED | 80.0 | 57.2 | 81.2 | 0/5 PASS |
| DISPERSED_MATCHED | 79.0 | 49.8 | 69.4 | 0/5 PASS |
| DISPERSED_BALANCED | 73.4 | 49.4 | 72.0 | 0/5 PASS |

分散・均等提示は失敗cell数を記述的には減らしたが、全cell gateを満たさないため候補の
採択根拠にはならない。固定v1親はparent-old品質の不足を含む開発用入力であり、この結果は
修正版H-CNP2の確認ではない。

実行artifact: `runs/cnp_repair/r001s/rcnp001s_schedule3/`。wall timeは372.672秒、
peak CUDA allocationは67,964,416 bytes、peak process RAMは2,535,723,008 bytesで、
登録済み2時間、12GiB、32GiB上限内だった。schedule JSONL、全cell/集合証跡、候補checkpoint、
data manifestを同directoryに保存した。

初回`rcnp001s_schedule1`と`rcnp001s_schedule2`は`reset_peak_memory_stats`のCUDA API
互換エラーで、データ生成・model forward・optimizer update前に停止した。空のnamespaceは
上書き・削除せず保存し、runnerを`2af73a5`で修正した後に新namespace `schedule3`を実行した。
