# ManiSkillによる身体性APC研究

2026-09-22開始の独立した実験トラックです。従来のREADME・Phase A–C・NRQ・CNPの
結論と成果物はそのまま残し、本トラックの進行条件にはしません。

**実装 → 環境内で試行 → データを見る → 一点を再設計 → 再試行**を優先します。
フェーズごとの厳格なゲートは設けません。大きな機能を実装した時点で、
その機能全体を対象とする少数の統合テストだけを書きます。

- [開始手順・実装済み機能](experiments/maniskill/README.md)
- [AIコーディングの指示](experiments/maniskill/AGENTS.md)
- [小さな身体操作と毎step判断：最新の設計・実装計画](experiments/maniskill/docs/PRIMITIVE_DECISION_PLAN.md)
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
学習済み操作は探索3例中2例で目標へ到達しましたが、保持と未使用条件への汎化は未達です。
自動スキル選択、銀行への能力追加、temporary解放を伴うAPC循環は未実装です。

最新設計では、小さな手先移動・指開閉・保持の10候補と毎stepの判断器から始め、
回転・台車を含む20候補へ拡張します。大きな移動・把持・運搬は組合せで生じる行動として扱います。
この方式は設計段階です。固定小操作の選択学習と、新しい小制御器の追加・圧縮・解放・再利用を
区別して検証します。
GPU・動画を含む実際の検証範囲はワークスペースの実験メモを参照してください。
