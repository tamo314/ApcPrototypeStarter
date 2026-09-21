# CNP Adaptation Repair v2 — 実装契約

日付: 2026-09-22 / ADR-0198

状態: **`R_CNP_001_COMPLETE_DIAGNOSTIC_GATES_FAIL`**。R-CNP-002は未認可。

## 目的と境界

CNP-004 v1の`G3_FAIL_STOP`を保持しつつ、ADR-0197で確認した実装と契約の相違を
別実装へ隔離して修正する。これはv1の再試行、候補の採択、block 2–4、転移、合成、G4、
bundle promotionを許可しない。

今回の実装は[repair.py](../../../src/apc/cnp/repair.py)にあり、v1の
`primitive.py`、`training.py`、`adaptation.py`を変更しない。したがって、v1実行commitと
保存artifactのhashは履歴証拠として残る。旧adapter候補を修正版で再測定して結果を置換しない。

## 修正済みの実装契約

| 論点 | v1の記録 | 修正版 |
|---|---|---|
| LOCAL adapter | 第2隠れ層後 | `input_proj → GELU → adapter → hidden_proj → GELU → readout` |
| 学習損失 | 有効要素を一括平均 | 非空集合ごとの有効要素BCE平均を取り、集合間で等重み平均 |
| replay | 初期1,536 source recordだけ | source query×threshold cellごとに同数、digest昇順で選定 |
| shadow | panel全体を集約 | domain×length×thresholdの全cellで品質・保持を判定 |

`CorrectedConditionalSelectPrimitive`は基盤MLPの重み形状と8,449 parameter数を変えず、
adapterの1,024 parameter数も維持する。ただしarchitecture signatureを変更するため、
v1 bundleとの暗黙の混在を拒否する。v1のadapter checkpointには新しい意味を与えない。

## 後続の研究実行に必要な二段階

### R-CNP-001: 既存親への限定診断

固定済みv1親をimmutable inputとして、修正版LOCALとv1実装LOCALを一つの新しい
**開発専用**shadow/transfer splitで比較する候補である。目的はadapter位置、set-weighted loss、
replay選定、cell gateのうち、親を固定して比較可能な要因を記述することだけである。
v1親自体は要素一括lossで学習済みなので、この段階はH-CNP2の確認にも、修正版全体の採否にも
ならない。両方式は同じnew/replay数、rank=8、256 updates、固定optimizerを使う。

順序効果は、この段階でも変更しない。順序比較をする場合は、query×threshold別の提示回数、
各batchの構成、seed、最大更新数を事前固定した独立対照とし、他の修正と同時に原因へ帰属しない。

#### R-CNP-001 固定実行契約

- **親:** `runs/cnp_v1/confirm/cnp003_confirm1/` のCNP-003最終MLP checkpoint、seed
  `610200`–`610204`だけをimmutable inputとして使う。新しいmodel seedは使わない。
- **blockと分割:** `q1+q2+`のみ。`repair_adapt_train`、`repair_new_shadow`、
  `repair_old_shadow`の3 roleを使い、各1,536集合（8 query×3 threshold×64集合）。
  v1 CNP-004のadapt/shadow/transfer、CNP-003/確認panelとはrole namespaceで分離する。
- **方式:** `LEGACY_LOCAL`（v1の後段adapter、要素平均loss、先頭1,536 source replay）と
  `REPAIRED_LOCAL`（第1隠れ層adapter、set-weighted loss、全128,000 source recordから
  query×threshold cellごと8件をdigest順に選ぶreplay）。各seedでadapter初期化は同じ乱数状態に
  戻す。方式間の入力、rank=8、AdamW lr=0.001、weight_decay=0、clip=1.0、256 updates、
  new16＋replay16は同じである。replay規則が方式固有なので、これは複合修正群の比較である。
- **測定:** 親と候補についてnew/old shadowのdomain×length×threshold全cellを記録する。
  各cellでBA≥0.95、F1≥0.90、old F1低下≤0.01を機械判定する。候補の基盤hash不変を要求する。
  shadowは256 step候補だけを一回測定し、checkpoint選択に使わない。
- **境界:** 5 seed×2方式×256 = **2,560 updates**、2時間、VRAM 12GiB、RAM 32GiB。
  転移・合成・block 2–4・METRIC・FULL・SCRATCH・candidate selection・bundle promotionは0。
  どの結果でもR-CNP-002を自動開始しない。失敗は候補を保存し、`FIXED_PARENT_DIAGNOSTIC`
  として記録する。H-CNP2/3/4のPASS/FAILには使用しない。

ユーザーの「次を進めてください」により、R-CNP-001の実装とこの一回の実行を認可する。
その実行結果は下記の実行記録へ追記し、R-CNP-002には別途の指示を必要とする。

#### R-CNP-001 実行記録

`runs/cnp_repair/r001/rcnp001_fixed_parent1/`をcommit `bdea060`で一回実行した。
新/旧cell gateはLEGACY_LOCAL、REPAIRED_LOCALとも5/5 FAIL。修正群は集約new F1を
0.905254から0.925877へ改善したが、old保持と多数のquality cellを満たさない。
この限定診断は`FIXED_PARENT_DIAGNOSTIC_COMPLETE_GATES_FAIL`で終了し、候補採択=0、
promotion=0、legacy sealed access=0を維持する。詳細は
[結果報告](../../results/R_CNP001_RESULT.md)。

ADR-0201の[改善案レビュー](../../results/R_CNP001_IMPROVEMENT_REVIEW.md)は保存値を
再集計し、評価契約・親適格性の明確化、提示順序/回数の対照、旧出力保持制約の順を提案する。
実験は追加実行していない。R-CNP-002の前に、上記の開発対照を検討する案であり、
本追記だけでその対照やR-CNP-002を認可しない。

ADR-0202で[順序・保持改善の詳細設計](../../design-docs/CNP_ADAPTATION_SCHEDULE_AND_RETENTION.md)と
[実装・診断計画](CNP_ADAPTATION_SCHEDULE_AND_RETENTION.md)を固定した。
評価セル128集合、A/B/Cのquotaと分散規則、lambda0/1の独立保持対照、予算・停止条件を定義。
状態は`DESIGN_COMPLETE_IMPLEMENTATION_PENDING`。本書のR-CNP-001実行記録は変更しない。

### R-CNP-002: 修正版親からの独立確認

set-weighted lossを初期学習にも適用するなら、source training、開発、5 seed確認、適応shadowを
すべて新しいrole namespaceと未使用query panelに分離して再構成する必要がある。
v1のCNP-003/CNP-V2-001確認panel、CNP-004 shadow、既存checkpointはその確認の代用にしない。
新worldを使うか、同一worldの未使用roleで分けるか、seed registry、データroot、計算予算、
causal判定、G1/G2/G3の停止条件は実行前に別ADRで固定する。

R-CNP-002で初めて、修正版H-CNP1の確認後に修正版H-CNP2/3を判定できる。品質条件は
各domain×length×threshold cellでBA≥0.95、平均集合F1≥0.90、各旧cellのF1低下≤0.01。
一つでも落ちたseedはそのstreamを停止し、成功seedだけを平均して救済しない。

## 実装テスト

`tests/test_cnp_adaptation_repair.py`は次を検証する。

- 長さ1と長さ16の集合が同じ重みでlossへ寄与すること。
- adapterが第1隠れ層後に適用されること。
- replayがcondition×thresholdで均等かつdigest順に選ばれること。
- shadowの一つのcellの失敗が全gateをFAILにすること。

R-CNP-001実行前の現時点では、新規データ生成、研究用model forward、optimizer update、
candidate selection、promotion、legacy sealed accessは0。R-CNP-001は上記の実行契約で認可済み。
R-CNP-002の実行には、固定契約を承認した後の明示的な実行指示が必要である。

検証: WSL Python 3.12.14で`tests/test_cnp*.py`は34 PASS、`ruff check .`はPASS、
`mypy src/apc`は199 source filesでPASS。全repository pytestは今回再実行していない。
既知の歴史artifact欠損により、v1時点でも全suiteはclean PASSではない。
