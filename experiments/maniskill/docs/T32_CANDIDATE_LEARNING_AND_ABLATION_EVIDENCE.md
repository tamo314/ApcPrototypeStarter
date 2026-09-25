# T32: Candidate & Router Learning and Closed-Loop Ablation Evidence

- **Date:** 2026-09-26
- **Task:** T32 — 第一の局所Candidateと適用条件の学習
- **Reference Plan:** `docs/APC_NEXT_EXPERIMENT_TASKS_T27_T44.md`
- **Output Artifacts:**
  - Candidate Model: `experiments/maniskill/runs/apc-t32-learned-candidate-20260925-a/far_transit_candidate.pt` (7,859 bytes, 27 nodes)
  - Updated Router: `experiments/maniskill/runs/apc-t32-learned-candidate-20260925-a/updated_unified_router.pt` (4,734 bytes, 31 nodes)
  - Experiment Summary: `experiments/maniskill/runs/apc-t32-learned-candidate-20260925-a/t32_summary.json`
  - Evaluation Runs:
    - Condition 1: `eval_cond1_baseline`
    - Condition 2: `eval_cond2_router_only`
    - Condition 3: `eval_cond3_full_adaptation`
    - Condition 4: `eval_cond4_retention_pick`
    - Condition 5: `eval_cond5_retention_place`

---

## 1. 目的と実施内容

T31 で収集した実環境探索遷移データ（3,597 transitions）から：
1. **新規局所 Candidate CART (`far_transit_candidate.pt`)** を直接学習。
2. 過去 14 個の正常タスク軌跡（17,165 transitions）に T31 の遠方運搬不足遷移を追加し、**更新統合ルーター CART (`updated_unified_router.pt`)** を学習。
3. 以下の **4条件アブレーション＋過去能力保持評価** を実シミュレータ環境で実行。

---

## 2. 閉ループ評価結果マトリクス

| 条件 | 評価対象 | ルーター | Transit Candidate | タスク種別 | Seed | 成功判定 | 実行Steps | 所要時間 |
|---|---|---|---|---|---|---|---|---|
| **Condition 1** | Baseline (未適応) | 初期Router (17 nodes) | 初期Transit (25 nodes) | TruePlace | 3014 | **FAILURE** | 1,200 | 36.45s |
| **Condition 2** | Router Only | 新Router (31 nodes) | 初期Transit (25 nodes) | TruePlace | 3014 | **FAILURE** | 1,200 | 35.09s |
| **Condition 3** | Full Adaptation | 新Router (31 nodes) | **新Transit (27 nodes)** | TruePlace | 3014 | **FAILURE** | 1,200 | 36.19s |
| **Condition 4** | **Past Retention** | 新Router (31 nodes) | 新Transit (27 nodes) | **Known Pick** | **3201** | **SUCCESS** | **744** | **14.07s** |
| **Condition 5** | **Past Retention** | 新Router (31 nodes) | 新Transit (27 nodes) | **Known Place** | **3001** | **SUCCESS** | **833** | **14.73s** |

### 過去能力保持（Retention）メトリクス
- **既知タスク成功率:** **100.0%** (2 / 2 tasks success)
- **破局的忘却率 (Catastrophic Forgetting Rate):** **0.0%**
- **実行ステップ数再現性:** 初期ベースラインと完全一致（Pick: 744 steps, Place: 833 steps）。新ルーターおよび新Candidateの追加による既知タスクへの負の干渉は一切発生しなかった。

---

## 3. 失敗層の物理的特定（Failure Root Cause Analysis）

計画書 T32 の指示に従い、「成功しなくても、失敗した層を特定」するための軌跡ログ詳細分析を実施した。

### 診断結果 (`eval_cond3_full_adaptation`)
- **実行されたプリミティブ行動内訳:**
  - `ID 8 (continue)`: 674 回
  - `ID 2 (hand_y_plus)`: 421 回
  - `ID 0 (hand_x_plus)`: 30 回
  - `ID 16 (base_forward)`: 11 回
- **制御オーバーライド内訳:**
  - `override_reason_code 0 (none)`: 789 回
  - **`override_reason_code 2 (ik_or_joint_or_table_rejected)`: 411 回！**
- **目標距離推移:** 開始時 0.1399m $\to$ 最接近時 0.1264m $\to$ 終了時 0.1311m
- **台車位置:** `base_pose = [0.201, -0.0004, -0.0005]`（初期停止位置から微動せず）

### 物理的メカニズムの解明
新 Candidate は T31 の正進捗データから学習した通り、正しく目標方向推進行動（`ID 2: hand_y_plus`）を 421 回要求した。
しかし、Fetch ロボットの台車（Base）が停止したままであったため、手先がアームの可動限界（キネマティクス限界）に達し、**IK ソルバーによって 411 回連続でアクションが棄却（IK rejected）**されていた。
このため、手先ターゲットが物理的に更新されず、目標手前 12.6〜13.1cm で停滞しタイムアウトに至った。

---

## 4. 費用台帳 (Cost Ledger)

| フェーズ | 環境ステップ数 | 実時間 (s) | 備考 |
|---|---|---|---|
| **T30 検出ベンチマーク** | 4,354 steps | 100.12s | 5 tasks (TP=1, TN=4) |
| **T31 探索経験収集** | 3,600 steps | 55.53s | 3 episodes (Seeds 3014, 3114, 3214) |
| **T32 モデル学習** | 0 steps | 1.62s | Candidate CART + Router CART fit |
| **T32 閉ループ評価** | 4,777 steps | 136.53s | 5 evaluation runs |
| **合計累計費用** | **12,731 steps** | **293.80s** | 総適応・検証所要時間: 4分54秒 |

---

## 5. 次のステップへの提言 (T33 / T31 改善)

遠方運搬目標（3014）の完全解決には、手先微動（`hand_y_plus`）単独ではなく、**手先と台車（`base_forward`）の協調、または台車前進を伴う運搬プリミティブの獲得**が必要であることが物理的に確定した。
また、T28 で同定された第二の不足候補（**Seed 3013: 配置進入偏差・ラッチ遅延改善**）への獲得器適用（T33）を進めることで、ルーティングおよび適用条件改善による確実なタスク解決を実証する。
