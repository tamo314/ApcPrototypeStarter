# T31: New Experience Collection Evidence

- **Date:** 2026-09-25
- **Task:** T31 — 新規環境経験を、行動と結果が対応する遷移として集める
- **Reference Plan:** `docs/APC_NEXT_EXPERIMENT_TASKS_T27_T44.md`
- **Output Artifacts:**
  - Transitions Dataset: `experiments/maniskill/runs/apc-t31-experience-collection-20260925-a/transitions.jsonl`
  - Collection Summary: `experiments/maniskill/runs/apc-t31-experience-collection-20260925-a/t31_collection_summary.json`
  - Source Script: `experiments/maniskill/scripts/run_t31_experience_collection.py`

---

## 1. 目的と実施内容

T30 で動的に発行された不足イベント `DEFICIT-EVT-S3014-ST781`（遠方運搬目標に対する水平前進停滞）を起点とし、
一切の旧 run や外部教師、事前 Candidate コピーを排除した上で、実シミュレーション環境での探索試行により新規遷移データを収集した。

### 収集プロトコル
- **トリガーイベント:** `DEFICIT-EVT-S3014-ST781`
- **探索セレクター:** `AdaptiveExplorationSelector`
  - 物体把持・リフト完了後、目標遠方領域（$dist_{xy} > 0.035\mathrm{m}$）において探索モードに遷移。
  - 20 ID プリミティブ空間から、目標誘導プリミティブ（主軸ベクトル移動）、ベース方策提案行動、および摂動行動を試行。
  - 遷移タプル $(s_t, a_{\mathrm{prop}}, a_{\mathrm{exec}}, s_{t+1}, \mathrm{goal}, \mathrm{progress}, \mathrm{safety\_event})$ を全ステップ漏れなく記録。

---

## 2. 収集結果とデータ内訳

| 項目 | 実測値 |
|---|---|
| **試行エピソード数** | 3 episodes (Seeds: 3014, 3114, 3214) |
| **総環境ステップ数** | 3,600 steps |
| **実測定所要時間** | 55.53 wall seconds |
| **記録遷移総数** | 3,597 transitions |
| **正の進捗（Positive Progress, $\Delta dist > 0.5\mathrm{mm}$）** | **239 transitions** |
| **停滞遷移（Stagnant, $|\Delta dist| \le 0.5\mathrm{mm}$）** | 1,425 transitions |
| **後退遷移（Negative Progress, $\Delta dist < -0.5\mathrm{mm}$）** | 106 transitions |
| **不意の落球（Accidental Drop）** | 0 transitions |
| **未把持・接近フェーズ遷移** | 1,827 transitions |

---

## 3. データ特徴と次のタスク（T32）への接続

収集された遷移データには、正進捗の行動だけでなく、停滞・後退行動も完全に網羅されている。
正進捗遷移（239件）は、遠方運搬目標に対する手先移動（ID 2 `hand_y_plus` 等）が把持を崩さず目標距離を短縮する挙動を明確に示している。

この `transitions.jsonl` を直接の学習源として、T32 において新しい局所 Candidate CART（`far_transit_candidate.pt`）および統合ルーターの更新へと進む。
