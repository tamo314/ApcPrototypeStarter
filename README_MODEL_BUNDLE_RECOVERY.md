# APC — Phase B / B2 Model Bundle Recovery

**作成日：2026-09-07 / 版：1.0 / 状態：実装前の指示パック**  
**タスク：B-C005REC-001～008 / 最初の実行対象：B-C005REC-001のみ**

## このパックの目的

R3-010が報告したCore／PrimitiveBank不整合、不完全な再構築経路、暗黙の再較正・共有キャッシュ依存を解消する。まず既存の整合した一式を復元できるか調べ、復元できない部分だけを新しい独立build経路で構築する。

これはPhase Bを継続する**復旧分岐**であり、Phase Aの全再実験、新しいmodel architecture、sealed評価への移行を指示するものではない。R3-011／012、B-C006、Task Inferenceは引き続きblocked。

**復旧成功と研究Gate合格は同義ではない。** 五つの独立モデルで整合した親構成を作れたこと、局所修正が効いたこと、G4が通ったこと、relation transferが実証できたことは別々に報告する。

## 読む順序

1. [実行計画](docs/exec-plans/active/PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md)
2. [AIコーディング向けタスク](docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md)
3. [復旧の受入条件](docs/EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md)
4. [ModelBundle設計](docs/design-docs/B2_MODEL_BUNDLE_RECOVERY_CONTRACT.md)
5. [エージェント規則](docs/AGENTS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_ADDENDUM.md)
6. [資料・変更範囲](docs/research/B2_MODEL_BUNDLE_RECOVERY_SOURCE_NOTES.md)

## 導入方法

新規文書を既存の同名ファイルへ無確認で上書きせず、差分を確認してリポジトリの`docs/`へ配置する。R3の実装ファイルはこのZIPには含まれない。

REC-001でroot `AGENTS.md`に「現在の実行分岐：B-C005REC」を追記し、上記文書を案内する。R3の実行計画・タスクキューにはG4後の復旧分岐へのリンクとblock状態を追記する。旧タスク本文・測定値・ADRは消さない。`README.md`のstatusのみ更新し、旧setupや依存versionを勝手に変えない。

新ADRは既存の`docs/DECISIONS_PHASE_B.md`へ追記し、`docs/DECISIONS.md`へ索引を追加する。資料はADR-0091までを報告するが、次番号は実repoで確認し、先に決め打ちしない。

[開始用プロンプト](AGENT_START_PROMPT.md)はREC-001だけを依頼する。

## 受入結果の読み方

| 結果 | 意味 | 意味しないこと |
|---|---|---|
| `RECOVERY_COHORT_READY` | 整合した5モデルを固定し、必要な復旧baseline検査が済んだ | SHIFT修正完了、G4、未見relation成功 |
| `RECOVERY_INFRASTRUCTURE_COMPLETE_RESEARCH_BLOCKED` | 復旧・注入・評価経路は成立したが、研究上の失敗が残る | sealedへ進めること |
| `G4_RECHECK_PASS` | 同一bundle系譜による新しいdevelopment統合評価が旧G4を満たした | 過去G4 FAILの撤回、G1やG5の自動PASS |
| `RECOVERY_BLOCKED` | 不整合／未学習／比較不能等が残る | 完全再学習を無条件に開始してよいこと |

## 資料の扱い

添付レポートと前回のR3指示書・実験計画のsnapshotを`docs/research/evidence/`に格納している。source notesは報告事項と今回の新規設計を区別する。新しいAPI名・manifest・復旧用数値は**提案する契約**であり、現コードの実在や実測成功を意味しない。

本パックはMarkdown指示文書である。モデルの復元、再学習、コード変更、GPU評価は実施していない。
