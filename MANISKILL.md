# ManiSkillによる身体性APC研究

2026-09-22開始の独立した実験トラックです。従来のREADME・Phase A–C・NRQ・CNPの
結論と成果物はそのまま残し、本トラックの進行条件にはしません。

**実装 → 環境内で試行 → データを見る → 一点を再設計 → 再試行**を優先します。
フェーズごとの厳格なゲートは設けません。大きな機能を実装した時点で、
その機能全体を対象とする少数の統合テストだけを書きます。

- [開始手順・実装済み機能](experiments/maniskill/README.md)
- [AIコーディングの指示](experiments/maniskill/AGENTS.md)
- [APC実現性の学習計画（2026-09-23改訂）](experiments/maniskill/docs/RESEARCH_PLAN.md)
- [アーキテクチャ・データ形式](experiments/maniskill/docs/ARCHITECTURE.md)
- [実験メモ・引き継ぎ](experiments/maniskill/docs/ITERATION_LOG.md)

```bash
# リポジトリ直下から。既存APCの仮想環境を変更しない。
bash experiments/maniskill/scripts/setup.sh
bash experiments/maniskill/scripts/run_iteration.sh
```

現在はCPU環境での試行・記録、台車の模倣学習と小型方策への蒸留、
手動2目標連鎖、操作BCとDAgger再ラベルまで実装・実測しています。
学習済み操作は探索3例で把持・持上げまで改善しましたが、目標到達・保持は未達です。
自動スキル選択、銀行への能力追加、temporary解放を伴うAPC循環は未実装です。

改訂計画では、操作学習の診断と既存移動方策の銀行化から始め、
追加学習→圧縮→解放→再利用・過去能力保持を小さく検証します。
GPU・動画を含む実際の検証範囲はワークスペースの実験メモを参照してください。
