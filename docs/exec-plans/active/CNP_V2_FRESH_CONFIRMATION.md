# CNP v2 未使用panel確認契約

日付: 2026-09-21 / CNP-V2-001 / ADR-0193

## 目的と範囲

ADR-0192で訂正したG2計測器を使い、CNP-003で固定・保存された20個の最終checkpointを、
これまで生成・評価に使っていないquery namespaceで一回だけ測定する。これはCNP v1の
履歴結果を上書きする再試行ではなく、H-CNP1の独立query panelでの再確認である。

実行対象は `configs/cnp/v2_confirmation.json`、固定source config
`configs/cnp/v1.json`、および `runs/cnp_v1/confirm/cnp003_confirm1/` の保存checkpointに限る。
新規学習・optimizer step・checkpoint選択・閾値変更・モデル変更・candidate selection・
bundle promotion・legacy sealed accessはすべて0である。CNP-004はこの範囲に含まれない。

## 固定データ境界

- source: 64 query（`source_train`）
- 既開封v1確認: 32 query（`confirm_eval`）
- v2確認: 32 query（`confirm_v2_eval`、`confirm_v2_00`〜`confirm_v2_31`）

各v2 queryについて、既知長 `{1,2,4,8,16}` と補間/外挿長 `{3,12,32}`、5閾値、
1セル128集合を固定する。source・v1・v2の全record digestに交差があれば実行前にFAILする。
v1 source manifestも再生成して完全一致を要求する。v2の結果を読んで追加panelを生成しない。

## 判定と上限

既知長の全長さ×閾値セルで、5個の固定MLP checkpointすべてについて、v1と同じ修正済みG2を
適用する: balanced accuracy ≥0.95、平均集合F1 ≥0.90、見本別F1 p10 ≥0.80、各介入別の
reference-effectful集合でCorrect ≥0.95かつCorrect−介入 ≥0.50、CUDA別プロセスfresh-load
完全一致。各controlのeffectful集合数は32以上を要する。

比較方式（LEARNED_METRIC、UNCONDITIONED、RAW_DISTANCE_FIT）も同じ固定checkpointで記録する。
MLPにはv2の二回SELECT/COUNT/SUM_FIRST副次panelも記録するが、G2を救済しない。
wall-clock上限は4時間、VRAM 12 GiB、process RAM 32 GiB、追加学習stepは0。資源上限は
`RESOURCE_STOP`、G2未達は`CNP_V2_G2_FAIL_OR_INCONCLUSIVE`として保存し、下流へ進まない。

5 seedすべてがPASSした場合、H-CNP1は同一world内の二つのquery-disjoint確認panelで支持される。
これはCNP-004の実行許可ではない。CNP-004には別途の適応実行指示が必要である。
