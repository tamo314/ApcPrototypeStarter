# R-CNP-001 固定親適応修正診断

日付: 2026-09-22 / 実行commit: `bdea060a582c16a4ce15bbd404fcd45331b05b86`

状態: **`FIXED_PARENT_DIAGNOSTIC_COMPLETE_GATES_FAIL`**。

## 結論

修正群 `REPAIRED_LOCAL` は同じ固定v1親に対する `LEGACY_LOCAL` より新domainの集約品質を
改善した。しかし、new/old shadowのdomain×length×threshold cell gateは両方式で全5 seed
不合格だった。従って、修正群を候補採択せず、R-CNP-002、転移、合成、block 2–4、G4へ進まない。

この試験はv1親を使う限定開発診断である。`REPAIRED_LOCAL`はadapter配置・集合重み損失・
replay規則をまとめて変え、replay入力も方式ごとに異なる。このため改善を個別の修正へ因果帰属
しない。また親自体はv1の要素一括lossで学習済みなので、H-CNP2を確認も反証もしていない。

## 固定実行と証拠

- run: `runs/cnp_repair/r001/rcnp001_fixed_parent1/`
- 親: `runs/cnp_v1/confirm/cnp003_confirm1/` の最終MLP、seed 610200–610204。
- block: `q1+q2+`。new train/new shadow/old shadowは各1,536集合、roleごとに分離。
- 方式: `LEGACY_LOCAL` と `REPAIRED_LOCAL`。各方式は256 update、new16＋replay16。
  計2,560 optimizer update。candidate selection=0、bundle promotion=`NOT_AUTHORIZED`、
  legacy sealed access=0。
- `data_manifest.json`はshared/各方式のcross-panel overlap 0。方式間replayの14件重複は、
  同じsource training recordを異なる方式が使うことによる許可済みの比較内overlapである。
- 各10候補の基盤hashは更新前後で一致。`metrics.jsonl`は10行、candidate checkpointは10個。
  run manifestのCNPコードhashは現作業treeと一致する。

## 結果

数値は5 seedの平均。new/old F1差は候補−固定親。品質条件は各cellでBA≥0.95、
F1≥0.90、保持条件は旧cellのF1低下≤0.01。

| 方式 | new BA | new F1 | 親からのnew F1差 | old F1 | 親からのold F1差 | new gate | old gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| LEGACY_LOCAL | 0.916345 | 0.905254 | +0.026504 | 0.947101 | −0.024897 | 0/5 | 0/5 |
| REPAIRED_LOCAL | 0.933231 | 0.925877 | +0.047127 | 0.943965 | −0.028033 | 0/5 | 0/5 |

修正群のnew F1は対照より平均+0.020623高く、最小集約F1の0.918274もF1基準0.90を上回る。
一方、最小集約BAは0.927348で0.95未満であり、個別cellの品質・保持条件は満たさない。
new quality失敗cellは方式別に平均84.6→74.6、new保持失敗cellは22.8→14.0へ減った。
一方、old quality失敗cellは63.2→58.0、old保持失敗cellは65.4→60.6で、保持は依然大きく
崩れている。最良の集約値や失敗cell数の減少でcell gateを救済しない。

| 指標 | LEGACY_LOCAL | REPAIRED_LOCAL |
|---|---:|---:|
| new BA最小値 | 0.908584 | 0.927348 |
| new F1最小値 | 0.897895 | 0.918274 |
| new quality失敗cell/seed | 84.6 | 74.6 |
| new保持失敗cell/seed | 22.8 | 14.0 |
| old quality失敗cell/seed | 63.2 | 58.0 |
| old保持失敗cell/seed | 65.4 | 60.6 |

## 資源と限界

wall timeは126.439秒、peak CUDA allocationは68,042,240 bytes、peak process RAMは
2,159,915,008 bytes。2時間、12GiB VRAM、32GiB RAMの上限内である。修正版replayの選定では
固定source training record 128,000件を再構成し、1,536件をquery×threshold cellごと8件に
層化した。source dataの学習や親の更新は0である。

run manifestはこの診断に必要な入力hash、config、code hash、seed audit、候補数、update数、
時間とメモリーを保存する。resident/active/temporary parameter bytesとfresh-load parityは
この候補未採択の限定診断では未測定であるため、費用・容量・bundle主張には使用しない。

実行前のWSL Python 3.12.14検証では`tests/test_cnp*.py`が35 PASS、`ruff check .`がPASS、
`mypy src/apc`が200 source filesでPASS。全repository pytestは既知の歴史artifact欠損のため
今回も実行していない。

2026-09-22追記（ADR-0201）: 上記の最小F1に関するfloor未達の説明を訂正した。
保存測定値・FAILは変更しない。[改善案レビュー](R_CNP001_IMPROVEMENT_REVIEW.md)で
親からの継承失敗、適応による追加劣化、提示順序・回数を分けて検討した。
