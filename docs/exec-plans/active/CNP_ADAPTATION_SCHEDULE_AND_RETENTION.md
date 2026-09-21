# CNP 適応改善 — 実装・診断計画

日付: 2026-09-22 / ADR-0202 / `IMPLEMENTATION_COMPLETE_EXECUTION_PENDING`

入口: [改善レビュー](../../results/R_CNP001_IMPROVEMENT_REVIEW.md)
→ [詳細設計](../../design-docs/CNP_ADAPTATION_SCHEDULE_AND_RETENTION.md)。
設計後の実装としてI-1〜I-3を完了した。`repair_protocol.py`、
`repair_schedule.py`、`repair_evaluation.py`、`repair_retention.py`、
`repair_followup.py`、2つの固定config、`repair-schedule`/`repair-retention` CLIを追加した。
新規研究データ生成・モデル評価・学習は未実行。v1 G3 FAIL、R-CNP-001 FAILを維持する。

## 1. 工程と完了条件

| 段階 | 内容 | 完了条件 |
|---|---|---|
| 設計（今回） | 評価契約、A/B/C、保持loss、分割・予算、停止条件 | 文書整合・算術確認・リンク・diff検査、ADR、local commit |
| I-1 | protocol/evaluation実装 | 完了。root明示生成、input/record/query分離監査、120cell/集合証跡、独立gateを追加 |
| I-2 | schedule実装 | 完了。A/B/C quota、整数比較の分散器、JSONL固定、input/target独立性を追加 |
| I-3 | runner・容量記録・親cache・保持loss実装 | 完了。親hash/cache検証、凍結hash、resource STOP、RのS結果参照を追加 |
| R-CNP-001S | 既存5親、A/B/Cの単一block診断 | 全結果とpaired差、CのPASSまたはFAIL_STOPを保存 |
| R-CNP-001R | 別split、固定Cでlambda0/1の比較 | 全結果と保持trade-off、lambda1のPASSまたはFAIL_STOPを保存 |
| R-CNP-002実行登録 | 固定レシピ・修正版親・新splitで正式確認する契約 | 新seed/データ/予算・G1/G2/G3・causal/controlのmanifest固定 |

設計完了から研究実行へ自動遷移しない。後日実装が指示された場合、I-1〜3と必要なテストは
その実装範囲に含め、通常の検証ごとに確認を挟まない。研究実行は名指しされた範囲で行う。
Sの失敗時は結果/ADRを残し、RやR-CNP-002を自動開始しない。
Rを別の診断として行う場合はSの失敗と独立診断の理由を実行記録に明記する。

## 2. 固定する実行上限

| 項目 | R-CNP-001S | R-CNP-001R |
|---|---:|---:|
| 固定親 | CNP-003 seed610200–610204 | 同じ5親 |
| block | q1+q2+のみ | q1+q2+のみ |
| 方式×親 | 3×5=15候補 | 2×5=10候補 |
| optimizer updates | 3,840 | 2,560 |
| unique new train / replay | 1,536 / 1,536 | 1,536 / 1,536 |
| shadow（new＋old） | 15,360＋15,360 | 別roleの15,360＋15,360 |
| wall上限（生成・監査・学習・評価・保存込み） | 2時間 | 2時間 |
| GPU / process RAM上限 | 12GiB / 32GiB | 12GiB / 32GiB |
| candidate selection / promotion / sealed access | 0 / 0 / 0 | 0 / 0 / 0 |

単一RTX5060Ti、float32、Python3.12、pyprojectの依存制約。別processのGPU利用量も確認し、
空き不足なら開始せず、他processを停止しない。上限到達は`RESOURCE_STOP`として保存し、
科学的FAILと区別する。上限の自動延長・中断候補の後日追加stepは行わない。
15/10候補は事前登録した同一診断の対照群。ある候補の性能FAILは同じ群の測定を打ち切る理由に
せず、全群の結果を記録してから科学的STOPを確定する。データ漏洩・凍結違反・予算超過は即時停止。

## 3. 開始前と測定手順

1. config/code/親hash、実行mode、空の新run namespace、seed予約、予算を検査。
2. new train/shadowと旧source replayを準備し、分離・件数・正負例・schedule監査を完了。
   不合格ならmodel forward・optimizer updateを開始しない。
3. 親のnew/oldを一度評価して固定。旧親不適格は記録し、固定親診断の対象として維持する。
4. 各親・各方式の初期出力一致、基盤hash、trainable数を検査して256更新。
   学習入力は固定scheduleからのみ取得。train全1,536＋replay全1,536の同じpanelで
   step0/16/64/256のlossを記録し、曲線を比較可能にする。これらをcheckpoint選択に使わない。
5. 各最終候補のnew/old shadowを一回評価。集合/セル指標、bootstrap、費用、gateを保存。
   step16/64ではshadowを評価しない。最終checkpointはすべて`selected=false`で保存。
6. 別processで最終checkpointを復元。固定したtrain/replay各32集合のlogitを
   atol=1e-6/rtol=1e-5、選択maskを完全一致で比較。shadowを再開封して採否を調整しない。
7. 全5親の結果と失敗セルを記録し、診断結果・ADR・local commitを完了。

## 4. 成果物と費用

run root予定: `runs/cnp_repair/schedule/<new_run_id>/` と
`runs/cnp_repair/retention/<new_run_id>/`。既存pathなら上書き拒否。

- `run_manifest.json`: code/config/親/data/schedule hash、全seed、環境、制約、終了状態。
- `data_manifest.json`: role/query/recordの監査、source lineage、各セルsupport、allowlist。
- `schedules/`: side別record ID列、quota、全batch構成、hash、提示偏りの要約。
- `metrics.jsonl` / `per_set_metrics.jsonl`: 全親/候補の集合・セル・domain集約を区別。
- `gate_reports.json`: 親適格性、新品質、旧品質、旧保持、凍結・分割・復元を別欄。
- `training_trace.jsonl`: 同一train panelのloss、BCE/保持lossの内訳、親正誤群の診断。
- `resource_report.json`: 方式別生成/監査/学習/評価/復元時間、CUDA allocated/reserved、
  RAM peak、CUDA同期方式、共有の前処理と方式固有の費用。
- `candidates/`, `parent_cache/`: 凍結hash、architecture signature、未採択state、cache lineage。
- `report.md`: 全5親・全方式、対照の差、PASS/FAIL/INCONCLUSIVE/NOT_EXECUTED、既知限界。

候補は9,473 parameter、そのうち更新1,024、基盤8,449。float32 weightは候補37,892 bytes、
親33,796 bytes、adapter4,096 bytes。親コピー、候補コピー、勾配、Adamの2 moment、step state、
replay、親logit cache、activationを別々に実測する。resident/active/temporaryを分離し、
更新parameter比を速度改善率と解釈しない。異なるセル数のR-CNP-001と総時間を直接比較しない。

## 5. 実装の検証と正式確認への出口

設計§6の回帰テストに加え、実装時には`python -m pytest -q`、`python -m ruff check .`、
`python -m mypy src/apc`を実行する。既知の歴史artifact欠損も実際の結果を記録し、全suiteを
PASSと呼ばない。標準G0の必要条件を満たさない状態から研究実行へ黙って進まない。
データ非依存の単体fixture・wiring検証と研究データでの実験は分けて記録する。

R-CNP-002は固定親診断と別の確認プログラム。今回の設計は、旧親の適格性不足を診断で
扱う例外を、正式確認へ持ち込まない。採用するレシピを先に固定し、新規親について
G1/G2とold品質が成立してから適応を検証する。転移・合成・G4の要件は元計画を維持する。

実装時検証: 合成fixtureによるCNP修正関連9 test、`ruff check .`、`mypy src/apc`は
Python 3.12でPASS。full pytestの結果は実行記録へ追記する。
研究実行は依然として0。`repair-schedule`は名指しでのみR-CNP-001Sを起動し、
`repair-retention`は名指しのsource runと完了済みR-CNP-001S reportを必要とする。

設計検証: Python3.12.13標準ライブラリの抽象整数IDによる算術確認で、A/Bのrecord別回数一致、
Bの入力列順に対する不変性、Cのquota（new170回×8/171回×16、replay21回×128/22回×64）、
unique各1,536/総提示各4,096/256 batchを確認した。panel数、更新上限、weight bytesの
算術、相対リンク、ADR番号の一意性、`git diff --check`もPASS。
これは実装済みモジュールのテストではない。設計のみのため全Python suite・ruff/mypyは未実行。
