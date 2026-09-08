# Source Notes — B-C005REC-004C

## 1. Sourceの優先と対応

| ID | 元資料 | 本パック内 |
|---|---|---|
| S1 | 添付 `貼り付けたマークダウン（1）(20260908-130813).md`、ADR-0097 | [REC004B_ADR0097_SUPPLIED.md](evidence/REC004B_ADR0097_SUPPLIED.md) |
| S2 | 添付 `貼り付けたマークダウン（1）(20260908-113719).md`、ADR-0096 | [REC004A_ADR0096_SUPPLIED.md](evidence/REC004A_ADR0096_SUPPLIED.md) |
| S3 | 前回配布 `CODEX_TASKS_PHASE_B_B2_MIRROR_SCHEDULE_REPAIR.md` | [REC004B_TASK_SUPPLIED.md](evidence/REC004B_TASK_SUPPLIED.md) |

source snapshotは実際の添付bytesを変更せずコピーしています。hashはpack manifestに記録します。エスケープされたMarkdown等も原文のままです。資料に書かれた実repo内の重み／APIが現在も存在するかは、実装担当者が確認してください。

## 2. Sourceが直接報告するもの

S1 Evidence 1は、A/Bの終端EM=0.4717/0.4336、train-fit=0.5029/0.4268、候補未達を報告します。Designは共通初期stateと各stepの入力一致、Evidence 4は非単調な長さ別失敗、Evidence 5は固定3候補のsource replay一致、Evidence 7はchild／独立query／RG3未実施を報告します。

S2はMIRROR_HALVES=0.6875と以前の長さ依存を報告します。「容量限界」という文はsourceの因果解釈であり、新たに実証済みの前提として採用しません。

## 3. Sourceを超える推論の扱い

初期化感度は有力な候補ですが、旧runとの比較には他の訓練乱数・runner差を確認する必要があります。同じ初期化A/Bで長さ別の成績が変わったことは観測であり、特定の位置機構が原因という証明ではありません。

train-fitとvalidationの近さは、測定した乖離の大きさを説明します。最適化の完了、全leakageの排除、容量限界を単独で証明しません。attentionや出力値の位置対応は、因果介入と区別します。

保護対象数の文章上の不一致は、physical ID一覧で確認すべき点として残し、実際にfreeze違反があったとは断定していません。

## 4. 今回新設するもの

本タスクID、診断専用の終了境界、5初期化の固定集合と30000-update総上限、共通training RNGの分離、位置識別／counterfactual／padding診断、length-balancedの件数、候補を採用しない規則は今回の設計です。

5初期化でA/B両scheduleを再試験する計画ではありません。A一条件を固定し、初期化軸だけを変えます。5本の最高値を採用せず、0.95到達は記述用です。

既存validationを再利用するため、結果はdevelopment上の診断です。新しいRG3/B2の独立queryやsealedは生成・評価しません。旧PASS／FAILを変更せず、REC-005を解除しません。

## 5. 未確認事項

MIRROR_HALVESの実際の位置変換規則、attention mask、合法語彙、optimizerの全引数、source checkpointの現在の実在、完全resume状態の充足、数値許容差は実装時に確認します。本文はそれらを推測で埋めません。

外部研究・APIの新規調査は本パックの根拠として使用していません。LR契約はS3と実repoの既存traceを継承し、インストール済み環境で確認する指示です。原文S3中の外部参照は旧文書の一部であり、本タスクで新たに検証した情報ではありません。
