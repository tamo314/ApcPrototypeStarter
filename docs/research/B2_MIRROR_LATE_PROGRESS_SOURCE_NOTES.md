# Source Notes — MIRROR_HALVES Late-Stage Audit & Conditional Extension

## 1. 根拠資料

- **S1**：[REC-004G／ADR-0102](evidence/REC004G_ADR0102_SUPPLIED.md)。ユーザーが結果として添付した報告のbyte-preserving snapshot。全5本の6000→12000の結果、長さ10の正解数、完全resume・source replay・新規30000 updates・候補非採用の報告を参照する。
- **S2**：[REC-004D指示書](evidence/REC004D_TASK_SUPPLIED.md)。既存P構造、5初期化、全5本の採用条件、固定I01、固定3候補、保護対象と独立queryの境界を参照する。これは以前作成した計画のsnapshotであり、実装済みの証拠ではない。
- **S3**：[Model Bundle Recovery指示書](evidence/MODEL_BUNDLE_RECOVERY_TASKS_SUPPLIED.md)。REC-005の既存ID、復旧pilot／cohort／研究Gateの区別、loaderとキャッシュの境界を参照する。これも計画snapshot。

原文には過去の解釈・報告表現を含む。現在の設計と異なる箇所があっても改変しない。証拠の保存は、原文中の全ての因果解釈を再確認したという意味ではない。

## 2. このパックからは確認していないもの

実repo、最新AGENTS、REC-004Gの実タスク文書 `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_BUDGET_EXTENSION_12000.md`、checkpoint、optimizer状態、全てのJSONと曲線は、このパックの作成時には独立検査・実行していない。S1が述べる実在pathとrecipeを、実装担当agentが現repoで照合する。

G指示書の本文は今回の配布sourceにない。存在を主張してsnapshotを創作せず、現repoから読む。本文の取得ができなくても、source full state・run metadata・実trainerから全レシピと権限の照合ができるかを報告する。必要な情報を確定できない場合は停止する。

旧E/Fのoracle診断は背景に限り、本タスクで再実行・訓練利用しない。外部文献調査・ライブラリversion更新もこの指示の作成では行っていない。

## 3. 原文の主張と新しい計算を分ける

S1 Evidence 4／Consequencesは「改善が長さ10に集中」と説明する。Hでは、その文言を原文から除去せず、実整数値から改善した正解数を長さ別に分解する。

例として、I03はS1で長さ10が10→26／206と報告されており、その増加は16例である。全体EMの丸め表示0.5713→0.8037と1024例から再構成すると全体増加は238例に対応するが、これは報告の確定整数列を読んだ結果ではない。**Hの集計では必ずraw JSONまたは同一予測で確定する。**

「どの長さが改善に寄与したか」と「現在の誤りがどこに残るか」は別指標。長さ6～9の合算成績を、各長さの個別成績と混同しない。

## 4. 今回新しく設計したもの

- 正式ID `B-C005REC-004H`。一件の中で監査と、条件付きの有限延長を扱う。
- 8000／10000／12000の同位相3点を主判定にし、位置別曲線とその他の位相は解釈用に残す。
- `EM_PROGRESS`／`SOFT_PROGRESS`の比較式、loss幅1e-6、未達全initの成立を要する運用条件。
- 条件成立時のみ全5本へ共通の12000→18000、最大30000新規updates。
- 条件不成立・未確定なら、新規学習0で報告して停止。
- 18000で全5本が到達しても候補選択・child・RG3・REC-005は別の明示指示へ残す。

これらは測定済みの事実でも、過去から定まっていた閾値でもない。学習進行条件は、追加計算を承認する限定的なresource-decision ruleであり、収束の検定でも効果予測でもない。loss幅は数値比較用であり、統計的有意水準ではない。

既存0.95の性能floor、P構造、Core、他操作のfreeze、既存validationが適応的利用済みであること、全5本とI01固定の将来採用方針、旧FAIL保存は継承する。

## 5. Source manifest

原本のSHA-256とbyte数は `source_manifest.json` に記録している。パックはプロジェクトのモデル・コード・runの変更を行わず、指示文書とsource copyのみを含む。
