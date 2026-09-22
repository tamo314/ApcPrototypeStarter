# ManiSkillによる身体性APC研究

2026-09-22開始の独立した実験トラックです。従来のREADME・Phase A–C・NRQ・CNPの
結論と成果物はそのまま残し、本トラックの進行条件にはしません。

**実装 → 環境内で試行 → データを見る → 一点を再設計 → 再試行**を優先します。
フェーズごとの厳格なゲートは設けません。大きな機能を実装した時点で、
その機能全体を対象とする少数の統合テストだけを書きます。

- [開始手順・実装済み機能](experiments/maniskill/README.md)
- [AIコーディングの指示](experiments/maniskill/AGENTS.md)
- [研究計画・最初の実装候補](experiments/maniskill/docs/RESEARCH_PLAN.md)
- [アーキテクチャ・データ形式](experiments/maniskill/docs/ARCHITECTURE.md)
- [実験メモ・引き継ぎ](experiments/maniskill/docs/ITERATION_LOG.md)

```bash
# リポジトリ直下から。既存APCの仮想環境を変更しない。
bash experiments/maniskill/scripts/setup.sh
bash experiments/maniskill/scripts/run_iteration.sh
```

今回用意したものは、研究計画と環境試行・記録の基盤です。
プリミティブ学習・ルーティング・圧縮は、まだ実装していません。
GPU・ManiSkill実環境での検証状況はワークスペースの実験メモに記録します。
