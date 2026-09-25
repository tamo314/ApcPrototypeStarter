# T30: Dynamic Deficit Detection Benchmark Evidence

- **Date:** 2026-09-25
- **Task:** T30 — 不足検出イベントから学習を開始する
- **Reference Plan:** `docs/APC_NEXT_EXPERIMENT_TASKS_T27_T44.md`
- **Output Artifacts:**
  - Summary JSON: `experiments/maniskill/runs/apc-t30-dynamic-deficit-20260925-a/t30_deficit_benchmark_summary.json`
  - Event Log: `experiments/maniskill/runs/apc-t30-dynamic-deficit-20260925-a/deficit_events.jsonl`

---

## 1. 目的と実施内容

従来のベンチマーク（W7）における「ブロック番号 `b_idx == 3` に依存したハードコード分岐」および「未把持や停滞の誤判定」を排除し、
実行中の物理的振る舞い（観測履歴、進行速度、目標距離、把持状態）から**自律的に不足を検出・イベント発行する閉ループ監視**を実証した。

### 検出器の改修点 (`src/apc_maniskill/deficit_detector.py`)
1. **不意の落球と正常解放の峻別**: 把持後に目標外領域で物体が脱落した場合を `accidental_grasp_loss` として即時検知。
2. **目標近傍停滞の検知**: 目標近傍（3.5 cm 以内）であっても整定・解放に至らず一定ステップ滞留した場合を `near_goal_placement_stall`（3013型）として検知。
3. **長距離運搬停滞の検知**: 把持状態を維持したまま水平移動が一定ステップ停滞（ストリーク120ステップ以上）した場合を `horizontal_transit_stall`（3014型）としてイベント発行。

---

## 2. 実行結果

5つの実タスク連続実行列（Pick既知 $\to$ Place既知 $\to$ TruePlace未知(3014) $\to$ Pick再遭遇 $\to$ Place再遭遇）において、検出器をオンライン監視した。

| Seq | Task Type | Seed | Description | Success | Steps | Time (s) | Deficit Triggered | Classification |
|---|---|---|---|---|---|---|---|---|
| 1 | pick | 3201 | Known Pick Baseline | True | 744 | 17.14 | False | **TN (True Negative)** |
| 2 | place | 3001 | Known Place Baseline | True | 833 | 14.73 | False | **TN (True Negative)** |
| 3 | true_place | 3014 | Unseen Far-Transit Deficit | **False** | 1200 | 36.90 | **True** (Step 781) | **TP (True Positive)** |
| 4 | pick | 3201 | Re-encounter Known Pick | True | 744 | 15.64 | False | **TN (True Negative)** |
| 5 | place | 3001 | Re-encounter Known Place | True | 833 | 14.68 | False | **TN (True Negative)** |

### 検出性能メトリクス
- **Total Tasks:** 5 (4,354 steps, 100.12 wall seconds)
- **True Positives (TP):** 1
- **True Negatives (TN):** 4
- **False Positives (FP):** 0
- **False Negatives (FN):** 0
- **Precision:** 100.0%
- **Recall:** 100.0%
- **False Positive Rate:** 0.0%

---

## 3. 発行された不足イベントの詳細

```json
{
  "event_id": "DEFICIT-EVT-S3014-ST781",
  "trigger_step": 781,
  "status": "genuine_deficit",
  "reason": "Carrying cube but failing to make horizontal progress toward goal (dist=0.136m)",
  "streak_steps": 120,
  "cube_position": [0.0684, 0.0760, 0.0308],
  "goal_position": [0.0648, 0.2124, 0.0200],
  "dist_xy_to_goal": 0.1364,
  "grasped": true
}
```

このイベント ID (`DEFICIT-EVT-S3014-ST781`) を入力キーとして、T31 の新規環境経験収集および T32 の Candidate 学習へと直接接続する。
ブロック番号や事前正解ラベルへの依存は完全に排除された。
