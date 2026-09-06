# APC Phase B / B2 — Post-D2 Repair Pack

**作成日：2026-09-06｜状態：実装前の指示書｜対象：第二診断完了後**

## 目的と最初の実行単位

本パックは **Phase Bを継続するB2修正サブフェーズ**である。Phase Cへの移行ではない。新タスクIDは `B-C005R3-001`～`B-C005R3-012` とする。既存のB-C005R1/R2や、その実験条件名R0/R1/R2とは別のタスク系列である。

最初は **B-C005R3-001のみ**を実行する。文書導入は同タスクの一部とする。以後も一タスクずつ、ユーザーから明示された範囲だけを実装する。B-C006とTask Inferenceは、最後の新GateがPASSするまでブロックする。

今回の順序：

```text
再現性修正 → relation分割 → 指標・統計契約 → 同一入力の対照確認
  → verifier → COUNT↔BIND → SELECT encoding → BIND scorer → SHIFT primitive
  → 統合・旧機能回帰 → 封印 → 新B2 Gate
```

ADR-0081の研究上の第一優先であるOption Cを維持し、その前提としてADR-0080で独立に指摘された再現性バグを先に修正する。複数の機構を一つの学習実験で同時に変更しない。

## 導入

このフォルダの `docs/` を既存リポジトリの `docs/` に追加する。既存ファイルと競合する場合は中断して差分を確認し、無条件に上書きしない。既存ソースコード・checkpoint・runsは本パックに含まれない。

R3-001でroot `AGENTS.md` の現在タスク参照を本系列へ切り替え、既存Phase Bのexecution planとtask queueに修正分岐とB-C006のblocked状態を追記する。過去文書を削除せず、root AGENTS.mdを別ファイルで丸ごと置換しない。

`docs/DECISIONS_PHASE_B.md` は既存ファイルに追記する。次のADR番号は実リポジトリの `docs/DECISIONS.md` から取得し、本パックでは予約しない。

## 読む順序

1. [実行計画](docs/exec-plans/active/PHASE_B_B2_POST_D2_REPAIR.md)
2. [AIコーディング向けタスク文書](docs/CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md)
3. [実験計画・Gate基準](docs/EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md)
4. [エージェント追加規則](docs/AGENTS_PHASE_B_B2_POST_D2_REPAIR_ADDENDUM.md)
5. [再現性・relation分割設計](docs/design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md)
6. [Functional Adequacy v2設計](docs/design-docs/B2_FUNCTIONAL_ADEQUACY_V2.md)

[最初の依頼文](AGENT_START_PROMPT.md)は、そのままエージェントへの開始指示として使える。

## 根拠と新規提案の境界

[出典ノート](docs/research/B2_POST_D2_SOURCE_NOTES.md)に、添付ADR由来の事実、今回提案した方針、参照した一次資料を分離した。添付されたADR-0080/0081は[原文コピー](docs/research/evidence/PHASE_B_D2_ADR0080_0081_SUPPLIED.md)として同梱する。

ソースコードやJSON成果物をこのパック作成時に独立検証したとは主張しない。タスクで指定する新しい型名・CLI・configは**実装予定の契約**であり、既存APIがある場合は重複実装を避けて適合させる。

Gate v2の数値基準、support予算、relationスプリット方式、UNCERTAINの運用は今回の**新規設計**であり、過去の確定事項ではない。旧B-C005/B-C005GのFAILを変更しない。文書を配置しただけでは実験PASSにならない。
