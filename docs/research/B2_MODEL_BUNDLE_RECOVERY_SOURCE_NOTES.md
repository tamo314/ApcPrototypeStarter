# Source Notes — Model Bundle Recovery

## 1. 使用した資料

本パックはこの会話で提供された資料に基づく。実repoのsource tree、全checkpoint、生のrun JSON、ADR全文を独立に取得・検査したものではない。現状の事実は**レポート上の報告**として引用し、REC-001で実体を確認する。

| ID | 同梱snapshot | 位置づけ |
|---|---|---|
| S1 | [R3_SERIES_STATUS_REPORT_SUPPLIED.md](evidence/R3_SERIES_STATUS_REPORT_SUPPLIED.md) | 2026-09-07付のユーザー提供状況レポート。実装・実行結果の主要根拠 |
| S2 | [R3_TASKS_SUPPLIED.md](evidence/R3_TASKS_SUPPLIED.md) | 前回のR3-001～012タスク文書。元の作業境界 |
| S3 | [R3_EXPERIMENT_PLAN_SUPPLIED.md](evidence/R3_EXPERIMENT_PLAN_SUPPLIED.md) | 前回ドキュメントパックの実験計画。旧Gateの数値定義 |
| S4 | [R3_FUNCTIONAL_ADEQUACY_V2_SUPPLIED.md](evidence/R3_FUNCTIONAL_ADEQUACY_V2_SUPPLIED.md) | 前回パックのAdequacy v2設計。新たな統計設計ではなく引継ぎ対象 |

S2～S4は前回の提案・指示仕様であり、実装済みかどうかはS1と実repoを照合する。snapshotは元byteを保存しているため、その中の相対リンクや文献番号は旧パック内の参照であり、本パックでは完全には解決しない。snapshot自体を新しいactive planとして扱わない。

## 2. Sourceが支持する内容

| 報告された内容 | Source位置 | 復旧指示への反映 |
|---|---|---|
| 入力生成はversion付きSHA256導出へ修正済み | S1 §2 R3-001 | 新たにbuiltin hash方式へ戻さない |
| G1はrelation不足、既存recipeはMINING_HOLDOUT_ONLY | S1 §2 R3-002 | 復旧をtransfer成功へ言い換えない |
| development Core未学習でraw EMが無意味だった | S1 §2 R3-004／005 | 旧Coreへ戻せば正常とは仮定しない |
| SELECTはBCE学習に対してsoftmax読み出しをしていた | S1 §2 R3-007 | sigmoid formula変更をexecution signatureで識別 |
| BIND rare-value失敗へ層化coverageが有効だった | S1 §2 R3-008 | 既定recipeの再検証。新scorer探索ではない |
| SHIFTは3/5 commitでGate FAIL | S1 §2 R3-009 | failed seedを省略せず、trained親へのrollbackを要求 |
| 新Coreとbank不整合、cache missでも10個未学習 | S1 §3.3 | 全16個のcoverageとdependency-aware loaderを新設 |
| COUNT↔BIND再実行がFAILへ変化しvariantも変化 | S1 §3.4 | evaluation内の再較正・選択を禁止 |
| 実K/C/N/R runnerへ修正部品を注入できなかった | S1 §3.1 | 最小adapterと実stream回帰を要求 |
| C0も壊れており相対回帰PASSは正常性を示さない | S1 §3.5 | 絶対性能を復旧用に追加 |
| failed G4後は別taskを設計する | S2 R3-010 | 本復旧分岐を新IDで定義 |
| 完成modelを封印しGate runnerはtraining API禁止 | S2 R3-011、S3 G5 | 復旧中はsealedを測らず、build/evalを分離 |

S1が「根本原因」と報告したキャッシュ不整合と、COUNT↔BINDに関する「新Coreが分離しにくくした可能性」は同じ確実性ではない。後者をこのパックで確定原因に昇格させない。

## 3. 今回新しく設計したもの

以下は資料中の既存実装・測定結果ではなく、今回のユーザー依頼に対する**復旧仕様の提案**。

- `B-C005REC-001～008`とRG0～RG6。
- immutable ModelBundle、scope付きqualification、fail-closed loader。
- dependencyを含むcache key、restore／build／evaluateの分離。
- `COHERENT_LIMITED`などの復旧状態と、SHIFT限定の既知baseline欠陥の扱い。
- 復旧用のdirect query 1,024例、非SHIFT15操作の各model×operation EM>=0.95。
- pilot seed10→独立5model cohortという実装順。
- 固定レシピ再適用と実K/C/N/R注入を分けた再評価手順。
- 技術的復旧完了と研究blockを併記する最終ラベル。

これらの提案値はREC-003で結果を見る前に固定する。R3-009のmean EM>=0.99、verifier tau=0.95、G4/G5等の旧基準を下げるためのものではない。学習step数・モデル容量・optimizerを新たな想像値で埋めてはいない。

## 4. 今回確定できないこと

- 整合した旧development一式が実際に復元できるか。
- 良い重みの完全な学習来歴と元pipelineの実在場所。
- 全16個の既存recipeの引数・step・容量・予算。
- 新Coreに対するCOUNT↔BIND低下の正確な原因。
- 現scorerのL4適用がruntime label漏れを含むか、単なるevaluation controlか。
- 復旧後、既定のSHIFT recipeが全5modelで目標へ届くか。
- 新しいG4が通るか、relation分割をどの追加研究で成立させるか。

不明点はREC各taskの監査・評価結果として埋める。復旧成功を先に仮定した結論を書かない。

## 5. 歴史と権限

旧R3-009、R3-010、B-C005、B-C005GのFAILは不変。新bundleでの結果は旧runの再分類ではなく、新runの測定になる。

現在のユーザー依頼は復旧指示文書の作成。本パックの作成だけではrepoを書き換えていない。実装時にも指定taskだけを実行し、REC-008後はSTOPする。

本パックは外部研究の追加調査ではなく、既存報告を受けた復旧設計である。S4に残る旧文献参照はsnapshotの一部で、今回改めて検証した文献という意味ではない。
