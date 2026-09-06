# Agent Addendum — B2 Post-D2 Repair

## 適用

root `AGENTS.md`を入口とし、本系列のtask / experiment / design文書を現在の詳細仕様とする。古い修正文書の「点推定で早期ACCEPT」「未確定を強制二択」「平均support<64」は本系列の新primary仕様ではない。旧baselineを再現する場合だけ、`legacy`という明示名で保持する。

## 必須規則

1. 一度に一タスク。次へ自動継続しない。未実行・FAIL・UNRESOLVEDをPASSと混ぜない。
2. 局所学習はdevelopment、選択はvalidation。封印testでの候補比較・モデル選択・予算調整は禁止。
3. 既に見た失敗relationは「新しい未見relation」ではない。新seedは独立checkpointの証拠にもならない。
4. 元のsealed runは診断記録として保存。そこに含まれたsupport/queryをtrainingに再利用しない。既知の失敗構造は新規development例を生成する仮説に使えるが、露出台帳に記録する。
5. repaired modelとbaselineには同じ入力・同じ親checkpointを用いる。比較不能ならその理由を報告する。
6. 生成コードにあるoracle operation名はdata生成用であり、runtimeへの追加リークを許可しない。model-visible TaskSpecは現行explicit条件の範囲だけ。
7. safety stressの不十分candidateを削除・修復して安全性評価を易しくしない。修正版SHIFTと旧不十分SHIFTは別の評価役割。
8. query/referenceを候補提案・受理・早期停止へ使わない。停止後のoffline指標計算だけに使う。
9. 最終EMはabstention/未出力を除外して高く見せない。conditional EMとunconditional EMを併記する。
10. replayで正常keyを保持することと、正常keyを自由に書き換えることを区別する。変更許可一覧とstate hashを保存する。
11. `UNRESOLVED`なreference、空分母、未実行seedに0%、100%、PASSを代入しない。
12. 複数Pythonプロセスのうち他作業のprocessを終了しない。既存テストと重複する場合は実行未完として報告する。

## 完了報告

```text
Task ID / code commit:
Changed files:
Tests actually executed:
Experiment commands actually executed:
Run directory:
Parent and output checkpoint hashes:
Generator / dataset / split / protocol hashes:
Permitted parameters changed:
Frozen parameters audit:
Seeds and number of unique model checkpoints:
Metrics: numerator / denominator / per-seed / aggregate / interval:
Gate criteria: PASS / FAIL / UNRESOLVED / NOT_RUN:
Comparison status: paired / reconstructed / unavailable:
Assumptions and deviations:
ADR appended:
Blocked downstream tasks:
```

標準確認は `python -m pytest -q`、`python -m ruff check .`、`python -m mypy src/apc`。実リポジトリに新しいcanonical commandがある場合はそちらを使う。全体pytestが未完了なら完了と書かない。実験結果の定義と分母の検証もコードテストに含める。
