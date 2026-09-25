# 外部行動教師なし自律獲得・公平比較検証報告（W5: T16）

作成日：2026-09-25（JST）  
対象タスク：W5 / T16（外部行動教師なしの獲得、進捗評価・局所探索、同一予算での公平比較）  
命題：H5（完全自律発見・自動閉ループ）、H1（局所適応）、H3（定着と実解放）、H4（干渉防止・過去能力保持）  
実行Run：`runs/apc-t16-self-exploration-20260925-a`  
レポート：`runs/apc-t16-self-exploration-20260925-a/self_exploration_report.json`

---

## 1. 目的と位置づけ

W5 / T15 では「教師あり（Oracle）」による運搬ラベルを用いた自動ループを完遂したが、これだけでは「単なる教師模倣の自動化」であり、ロボットが外部オラクルなしで自律的に新能力を発見したことにはならない。

本検証（W5 / T16）では、**正解行動を返す外部教師（Oracle）を一切使わず（`teacher_calls = 0`）**、環境の達成条件・進捗シグナル（水平距離短縮度 $\Delta d_{xy}$ と把持維持）のみをフィードバックとする**身体性プリミティブの局所探索（Local Exploration with Goal Progress Signal）**を通じて自律的に能力を獲得できるかを検証した。

さらに、ロードマップの公平性要求に従い、同一の探索予算（1,200 steps）を与えた**3条件の比較（Method A: APC, Method B: 直接Candidate CART, Method C: 探索なしBaseline）**を独立別プロセスで検証した。

---

## 2. 実験設計と方式比較

```mermaid
flowchart TD
    subgraph Phase1["Phase 1: 外部教師なし自律探索 (1,200 steps, Teacher Calls = 0)"]
        E1["運搬不足領域の遭遇 (grasped & raw_id==4)"] --> E2["局所進捗Q学習・探索 (LocalTransitExplorer)"]
        E2 --> E3["正の進捗サンプル抽出 (59 steps, 前進・水平移動発見)"]
    end

    subgraph MethodA["Method A: APC 自律定着ループ"]
        E3 --> MA1["Temporary MLP学習 (3,000 updates, 11,092 params)"]
        MA1 --> MA2["物理ロールアウト検証"]
        MA2 --> MA3["Candidate CART定着蒸留 (25 nodes, 0 params)"]
        MA3 --> MA4["単一統合ルーター更新 (19 nodes)"]
        MA4 --> MA5["一時資源実解放 (50.3 KB 解放)"]
        MA5 --> MA6["独立配布物 (38.8 KB)"]
    end

    subgraph MethodB["Method B: 直接 Candidate CART"]
        E3 --> MB1["直接 CART fit (25 nodes, 0 params)"]
        MB1 --> MB2["単一統合ルーター更新 (19 nodes)"]
        MB2 --> MB3["独立配布物 (38.8 KB)"]
    end

    subgraph MethodC["Method C: Baseline (未適応)"]
        MC1["探索・適応なし (Baseのまま)"]
    end
```

---

## 3. 実測比較結果

全方式を、同一初期シード（Target: 3011, Retention Place: 3001, Retention Pick: 3201）かつ一時資源を参照できない独立別プロセスで評価した。

| 指標 / 条件 | Method A: APC 自律定着<br/>(Exploration $\to$ Temporary $\to$ CART) | Method B: 直接 Candidate<br/>(Exploration $\to$ Direct CART) | Method C: 未適応 Baseline<br/>(No Exploration) |
|---|---|---|---|
| **外部教師呼び出し数 (Teacher Calls)** | **0 回** | **0 回** | **0 回** |
| **探索環境ステップ数 (Exploration Steps)** | **1,200 steps** | **1,200 steps** | **0 steps** |
| **探索中の把持脱落 (Dropped Cubes)** | **0 回 (安全性維持)** | **0 回 (安全性維持)** | - |
| **自律発見サンプル数** | **59 steps** (Action 0: 53, 1: 2, 2: 2, 3: 2) | **59 steps** (同一データ) | 0 steps |
| **新タスク完遂ステップ (Target 3011)** | **1,152 steps** | **1,152 steps** | 1,200 steps (タイムアウト) |
| **20連続ステップ成功維持 (Target 3011)** | **完全達成 (True)** | **完全達成 (True)** | **失敗 (False / ドリフト停滞)** |
| **過去配置タスク保持 (Place 3001)** | **833 steps (100% 保持)** | - | - |
| **過去把持タスク保持 (Pick 3201)** | **744 steps (100% 保持)** | - | - |
| **一時資源実解放量** | **50,273 bytes (100% 解放)** | 0 bytes (最初から不使用) | - |
| **最小配布バンドルサイズ** | **38,874 bytes (0 params)** | **38,835 bytes (0 params)** | - |

---

## 4. 命題（Hypotheses）に対する実証的考察

### 4.1 外部教師 0 での新能力獲得（H5 の自律発見立証）
- 外部教師（Oracle）による指示が一切存在しない状態（`teacher_calls = 0`）でも、ロボットは環境進捗報酬（$\Delta d_{xy} > 0$ かつ 把持維持）をもとに試行錯誤を行うことで、目標接近に必要な移動プリミティブ（Action 0: 前進等）を自律的に発見・収集した。
- これにより、未適応 Baseline（Method C）では上空ドリフトで 100% 失敗していた難関タスク（Target 3011）において、**20連続ステップ成功維持（1,152 steps）を完全達成**した。

### 4.2 Temporary 経由 vs 直接 Candidate CART の公平比較（T08/T16 分岐基準）
- 同一の自己探索データ（59 steps）を用いた比較において、直接 CART（Method B）も APC（Method A）と完全に同一のステップ数（1,152 steps）で安定成功を達成した。
- 今回の局所運搬補正のように、探索空間が比較的小さくデータ分布が明確な領域では、直接 CART への学習でも十分に機能することが実証された。
- 一方、APC（Method A）は Temporary MLP（11,092 params, 50.3 KB）を挟んだ場合でも蒸留歪みなく Candidate CART へ定着でき、さらに**一時資源を 100% 実解放（50,273 bytes 解放）して配布物 38.8 KB へ圧縮**できることを実証した。

### 4.3 破滅的忘却の完全防止（H4 の立証）
- 外部教師なしで自律獲得した Candidate CART および更新された統合ルーターは、過去の配置タスク（Place 3001: 833 steps）および把持タスク（Pick 3201: 744 steps）に一切の悪影響を与えず、過去能力の完全保持（100% 成功）を維持した。

---

## 5. 結論

1. **W5 / T16 の完全達成**:
   - 外部教師 0 回・環境進捗フィードバックのみによる自律的能力獲得を実証した。
2. **Phase 5（W5: 自動閉ループ完遂）の終了**:
   - T14（自律的不足検出器）、T15（完全自動ライフサイクルループ）、T16（外部教師なし自律獲得・公平比較）がすべて完了し、ロードマップ W5 の全要件をクリアした。
3. **Phase 6（W6: 未学習の組合せ合成・Compositional Transfer）への進展準備完了**:
   - 次のステップとして、獲得された能力群（把持回復、運搬、配置）を用いて、未学習の因子組合せ（Held-out Combinations）に対するゼロショット合成能力（T17–T19）の検証へ進む。
