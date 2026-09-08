# Source Notes — B-C005REC-004D

## 1. Sourceの優先と対応

| ID | タスク文書内の呼称 | 実際の対応先（Stage Aで確認） |
|---|---|---|
| S1 | `research/evidence/REC004C_ADR0098_SUPPLIED.md`（ADR-0098実行報告） | **該当ファイルは本repoに存在しない。** 実体は `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md` のADR-0098本文と `runs/phase_b_b2_model_bundle_recovery/rec004c/run_001/` の実artifact一式（`report.md`、`position_confusion.json`、`position_error_summary.json`、`initialization_summary.json`ほか）。 |
| S2 | `research/evidence/REC004C_TASK_SUPPLIED.md`（REC-004C指示書snapshot） | **該当ファイルも存在しない。** 実体は `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_SCHEDULE_REPAIR.md`（REC-004Bの指示書。REC-004C自体は既存のREC-004B/REC-004Aインフラを継承する形で実装されており、独立の指示書snapshotは`docs/research/evidence/`に置かれていない）。 |
| S3 | `REC004B_ADR0097_SUPPLIED.md` / `REC004B_TASK_SUPPLIED.md` | 存在する（`docs/research/evidence/REC004B_ADR0097_SUPPLIED.md`、`docs/research/evidence/REC004B_TASK_SUPPLIED.md`）。 |
| S4 | `RECOVERY_ACCEPTANCE_SUPPLIED.md` / `MODEL_BUNDLE_CONTRACT_SUPPLIED.md` | 前者は存在する（`docs/research/evidence/RECOVERY_ACCEPTANCE_SUPPLIED.md`）。後者は存在しないが、実体である `docs/design-docs/B2_MODEL_BUNDLE_RECOVERY_CONTRACT.md` が同名別pathで存在し、内容はこちらを正とする。 |

タスク文書自身が §0 で明記している通り、「本パック作成時には実repoのコード・checkpoint・JSONを独立実行していない」。Stage Aはこの前提を検証する段階であり、上記の通りS1・S2は実在しないことを確認した。以降の§2/§3はS1/S2が指す**実際のREC-004C成果物**（ADR-0098本文、`report.md`、各JSON）を直接読んだ結果に基づく。

## 2. Sourceが直接報告するもの

ADR-0098（REC-004C）はI01〜I05の終端`existing_validation` sequence EMを0.2041/0.3135/0.4346/0.4814/0.5029（mean=0.3873, sample SD=0.1260, n=5, 0.95到達0件）と報告する。`initialization_summary.json`の`per_init.<ID>.final_validation_em`と厳密一致することをStage Aで再確認した。

`MirrorHalvesOp.apply`（`src/apc/environments/operations.py:551-555`）は`mid = n // 2; reversed(seq[:mid]) + reversed(seq[mid:])`であり、位置写像`pi_n(i) = mid-1-i (i<mid) else n+mid-1-i`はREC-004C自身の`operation_contract_audit.json`で長さ2〜12・各64試行、oracle interpreterとの完全一致として検証済み。

`position_confusion.json`／`position_error_summary.json`の数値集計（moved 6948件・fixed 1240件、各init・各checkpointのcorrect数）は、Stage Aで`pi_n`から独立に再計算し直しても一致する（`metric_erratum.json`参照）。

## 3. Sourceを超える推論の扱い

`report.md:68-69`と`report.md:153`、およびADR-0098 Evidence 4の「fixed位置は奇数長のみ、かつ中央no-op位置」という文章表現は、REC-004C自身が保存した`pi_n`の値から機械的に再計算すると誤りである（偶数長6・10がそれぞれ2個ずつfixed位置を持ち、8は0個、7・9は中央ではない1個ずつを持つ；4/5の長さがfixed位置を持つ）。この誤りは**文章のみ**であり、`position_confusion.json`等のJSON数値自体は誤っていない。タスク文書§1.3が示唆した通りの帰結であり、本タスクではこれを`REPORT_ONLY_ERRATUM`として`fixed_position_audit.json`に記録し、REC-004C自身のJSON成果物は書き換えない（新ADR-0099に訂正を追記するのみ）。

「容量限界」「attention span不足」「位置スコアが必須」は、いずれもREC-004A/REC-004Cの報告内で因果的に実証された前提ではなく、本タスクが検証する仮説として扱う。position biasを追加すれば改善するという結論は、Stage D/Eの実測結果が出るまで採用しない。

## 4. 今回新設するもの

汎用座標（出力位置i・入力位置j・実長n・固定基準長L_ref）のみを入力とする4→32→1の2層MLP位置bias（192 parameters、出力層0初期化）、U/P 5ペア×6000-updateの新規paired比較、bias-zero ablation、全5本のP existing_validation EM>=0.95という採用条件、P/I01@6000を条件成立時に固定して採用する規則（最高値のinitを選ばない）、新operator型`cross_position_length_bias_v1`の厳格な(de)serialization契約は、いずれも本タスクで新設した設計である。既存資料が示唆していた内容ではない。

## 5. 未確認事項

Stage Aで実際に確認済み（未確認ではなくなった）事項：I01〜I05の`initial_states/*.pt`の実在とhash一致、固定3候補（CYCLE_FOUR/ROTATE_TRIPLETS/SWAP_ENDS）のstep=4000 checkpointの実在とEM再現、`CrossPositionPrimitive.forward`内での実content長・実位置へのアクセス可能性、`nn.MultiheadAttention`への追加biasの注入経路。

実装時にのみ判明した事項：ゼロbias時の勾配一致はCUDA上では完全一致（0.0）だが、CPU上ではbackward計算経路の違い（bool `key_padding_mask` vs float `attn_mask`）により約2e-8の浮動小数点差が生じる。これはStage Cのprotocol lock前に許容差として明示的に固定した（`zero_bias_parity.json`参照）。BIND等、偶数長のみ有効な保護対象operationについては、汎用の長さ範囲サンプラーではなく`is_valid_for_length`を尊重した生成が必要であることも実装時に判明した。
