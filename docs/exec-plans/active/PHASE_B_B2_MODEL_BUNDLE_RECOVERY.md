> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](../../results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# Execution Plan — Phase B / B2 Model Bundle Recovery

> **現在地・実行順の正本:** [Phase B 再開計画](PHASE_B_RESTART.md)（2026-09-12、ADR-0127）。
> 本文は既存の仕様・作成当時の状態を保持する。現在の進捗と今回の継続指示は再開計画を参照。

**B-C005REC-001～008 / proposed v1.0 / 2026-09-07**

> **状態確認（2026-09-12、ADR-0126）:** REC-001〜003の基盤整備後、REC-004のRG3は
> 非SHIFT4操作のfloor未達でFAIL。REC-004Aではうち3操作のvalidation条件を満たしたが、
> MIRROR_HALVESの挿入タスクはREC-004ACまで進んでも採用・child bundle・RG3再検証に至っていない。
> 最新ACの通常J0 EMは全長混合0.764648、長さ10で0.080078、O1 EMは長さ10で1.0。
> 今回の監査でACの正解key位置の計測バグを確認した。原因指標の訂正を先に提案するが、
> EMベースのFAILは維持する。REC-005以降・研究G4/G1・封印評価のblockは未解除。
> [監査と再開方策](../../research/PHASE_B_BLOCKER_AUDIT_2026_09_12.md)は調査・提案であり、
> 新しい修正実装・研究実験を許可しない。以下の既存契約と閾値は変更しない。

## 1. 開始条件と目的

R3-010は`DEVELOPMENT_INTEGRATION_FAIL`。レポートでは、新shared encoderと以前のbankの混在、再構築時に16個中10 primitiveを学習しない経路、COUNT↔BINDの別のrouting劣化が報告された。[S1: §3.3–3.5]

本分岐は、**読み込むモデルを固定し、必要なら完全に再生成し、その同じモデル上で比較できる状態**を回復する。新しいloss・scorer architecture・verifierの統計ルールの探索は行わない。

## 2. マイルストーン

| Task | 内容 | 境界 |
|---|---|---|
| REC-001 | 保存・依存台帳・復元／再構築の分類 | RG0：調査完了。学習なし |
| REC-002 | ModelBundle／fail-closed loader／認証の契約 | RG1：CPU contract PASS |
| REC-003 | 既存レシピによる全16個の明示build経路と実験契約 | RG2：build plan固定。長時間学習なし |
| REC-004 | 事前指定1 seedで復元／clean build／fresh-load検証 | RG3：パイプラインとbaselineの確認 |
| REC-005 | 独立5モデルへ展開し親bundleを固定 | RG4：`RECOVERY_COHORT_READY` |
| REC-006 | 既存runtimeへbundle注入。既定R3修正を別工程で再適用／照合 | RG5：runtime・lineage比較の成立 |
| REC-007 | 評価専用のC0～C5と実K/C/N/R回帰 | RG6と研究G4を別判定 |
| REC-008 | 復旧完了範囲・未達・再開条件の引き継ぎ | STOP。封印や次タスクなし |

正式IDは各行の先頭に`B-C005`を付ける。各taskの実施には個別のユーザー指示が必要。

## 3. 復旧経路

```text
既存artifactを保存・調査
  ├─ 整合した許可範囲の一式が復元可能
  │    └─ 新namespaceへ読み取り専用で取り込み、現在の固定fixtureで機能確認
  └─ 欠落／不整合／復元不能
       └─ 既存レシピを呼ぶ明示buildを実装し、不足dependencyだけ構築
               ↓
全16個のcoverageとbaseline資格を確認
               ↓
5個の親bundleを固定
               ↓
修正準備と評価を分け、実K/C/N/Rへ注入
```

「古いtimestamp」「同じseed」「state_dictがloadできた」は互換性の証拠にならない。旧Coreへ戻すだけで十分とも、8 canonicalだけ学習すれば16個完成とも仮定しない。

既存一式を取り込めた場合、REC-004で全heavy buildを重複実行する必要はない。ただしREC-003のclean-build経路は全16個について存在しなければならない。clean-build未実測の部分は`CLEAN_BUILD_IMPLEMENTED_NOT_FULLY_EXERCISED`と明記し、完全再生成を実証したとは言わない。

## 4. 完了と未達を分ける

- artifact整合・完全build経路・fresh-load・注入が成功しても、SHIFT mean EM>=0.99、G4、G1のrelation要件は別。
- 既知SHIFT不足は明示した`COHERENT_LIMITED`の復旧baselineでのみ許容する。未学習SHIFTへのfallback、未修正15操作の崩壊の許容ではない。
- 新親で既定修正が失敗しても、復旧基盤を直せた成果は残る。新しい機構修正を自動追加せず、研究blockとして渡す。
- unknown来歴を新manifestに包んでも既往露出は消えない。未見relationの主張は行わない。

## 5. Block台帳

| 項目 | 本分岐開始時 | この分岐で可能なこと |
|---|---|---|
| R3-002 / G1 | `PROTOCOL_INSUFFICIENT_RELATIONS` | 制約を保持。対象限定復旧は可能だがtransfer成功にはしない |
| R3-006 | 過去PASSは当時の親に限定、現親ではFAIL | 同一親・既定方式で再評価 |
| R3-009 | 3/5 commit、Gate FAIL | 完全baselineで既定candidateを照合。別学習探索は不可 |
| R3-010 / G4 | FAIL、K/C/N/R代替のみ | 真の注入経路で新runのG4を測る |
| R3-011／012 | 未実行 | このパックでは実行しない |
| B-C006／Task Inference | blocked | このパックでは解除しない |

RECの技術的なPASSが研究Gateのblockを自動的に消さないことをdependency manifestにも実装する。

## 6. 文書の権限

現在のユーザー依頼は復旧タスクの設計である。エージェントが実装する際は選択taskだけを実行する。本分岐は復旧・限定回帰を明示的に許可するが、過去のGate閾値を変更せず、旧FAILをPASSへ更新しない。

詳細は[タスク](../../CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md)、[受入条件](../../EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md)、[source notes](../../research/B2_MODEL_BUNDLE_RECOVERY_SOURCE_NOTES.md)。
