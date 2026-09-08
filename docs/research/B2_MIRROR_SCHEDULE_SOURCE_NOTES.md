# Source Notes — REC-004B

**作成日：2026-09-08。** 添付資料の事実、今回の解釈、今回新設する実験条件を区別する。

## S1：REC-004A / ADR-0096

原添付：`貼り付けたマークダウン（1）(20260908-113719).md`。本パックには[原文snapshot](evidence/REC004A_ADR0096_SUPPLIED.md)をbyteコピーした。

参照箇所：Status、Design、Evidence 1–5、Consequences。3操作のvalidation合格、MIRROR_HALVESの未達、T_max=1000の延長、child未公開、RG3再判定未実行、共有cache非変更は実行担当者の報告である。現repoのcheckpoint・コードを本パック作成時に独立検査したものではない。

「長さ依存」は観測として引き継ぐ。「capacity limit」「training budgetではない」という帰属は報告側の解釈として残し、本タスクでは未確定扱いにする。原文を書き換えたものではない。

## S2：REC-004A指示書

[配布時snapshot](evidence/REC004A_TASK_SUPPLIED.md)。正式な追加条件は今回の`CODEX_TASKS_PHASE_B_B2_MIRROR_SCHEDULE_REPAIR.md`が定義する。

変更する箇所は、MIRROR_HALVESに対するLR schedule比較、終端6000-step選択、旧3操作の4000-step固定、今回の独立query。非SHIFT floor、freeze、履歴保存、scope認証、最終query後の選び直し禁止は継承する。

## S3：Model Bundle Recovery受入条件

[配布時snapshot](evidence/RECOVERY_ACCEPTANCE_SUPPLIED.md)は、会話内で既に作成・添付された`APC_Phase_B_B2_Model_Bundle_Recovery.zip`中の原ファイルをそのまま取り出したもの。

特に§2.2と§6を継承する。非SHIFT15操作の各query EM>=0.95、各1024例、SHIFTの限定例外、機能floorとreference認証の区別、fresh-process確認を変更しない。

各snapshot内の相対参照は元repo用のまま保存している。snapshot内の全参照先を本パックへ再収録したという意味ではない。実装時は現repoのsource of truthを確認する。

## W1：PyTorch CosineAnnealingLR公式仕様（API確認のみ）

2026-09-08閲覧：

`https://docs.pytorch.org/docs/2.12/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html`

参照範囲：閉形式、単調に進む内部counter、optimizerの後のscheduler.step、get_last_lrの意味。本タスクのLR表はその式へ指定値を代入した検査用計算であり、APCの実測値ではない。T_max=6000の比較が性能を改善することは未実証。

この参照はrepoのPyTorch version変更を許可しない。実行時はインストール済みversionとrepo制約を維持し、CPU traceで確認する。

## W2：PyTorch Saving and Loading Models（保存契約の補助）

2026-09-08閲覧：

`https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html`

参照範囲：学習再開用checkpointにはモデルだけでなくoptimizer状態等を保存するという説明。本タスクはさらに、paired実験のためscheduler／RNG／data位置も保存する独自の再現性契約を置く。API追加・framework更新の根拠にはしない。

## 今回新設した設計

A/B各6000 updates、共通初期state・入力、T_maxのみ変更、500-step観測、B優先・A fallbackの終端選択、固定3候補の4000-step採用は、今回ユーザーの依頼に対して作成した新しい実験仕様である。S1の結論や実績として引用しない。

出力物は指示文書のみ。コード修正、GPU学習、モデル機能検証は実施していない。
