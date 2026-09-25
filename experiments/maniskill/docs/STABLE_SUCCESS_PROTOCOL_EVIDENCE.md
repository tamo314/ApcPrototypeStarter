# 安定成功プロトコルの再集計・新プロトコル検証報告（T00）

作成日：2026-09-25（JST）  
対象タスク：W0 / T00（安定成功の再集計と最小再実行）  
対象課題：`APC-FetchTruePlaceFar-v1`

---

## 1. 課題の背景と動機

`APC_RESEARCH_SUMMARY.md`（第12節）およびアブレーション実験ログにおいて、以下の現象が確認されていた：
- `seed 3012` の限定介入 MLP（`mlp_guarded`）は「最終成否：成功、Hold 20維持：失敗（755 steps）」
- `seed 3002` の更新 Candidate CART は「最終成否：成功、Hold 20維持：失敗（861 steps）」
- `seed 3011` の手設計ガードは「最終成否：成功、Hold 20維持：失敗（889 steps）」

これらは「最終成否が成功であるにもかかわらず、なぜ連続20 step維持が失敗となっているのか？」という疑問を提起していた。

---

## 2. 既存 step ログ（`steps.jsonl`）の厳密な追跡と原因特定

スクリプト `experiments/maniskill/scripts/analyze_stability_metrics.py` により、初回成功（`first_success_step`）以降の全ステップの幾何・力学状態を 1 step 刻みで分析した。

| 評価対象 Run | 初回成功 | 成功中断 step | 中断の物理的原因 | 中断後の連続成功 | 終了 step | 旧判定結果 |
|---|---|---|---|---|---|---|
| **seed 3012** (`mlp_guarded`) | 734 | 735 (1 stepのみ) | 指先の一時的接触（`is_grasped=True`, `is_released=False`） | **19 steps** (736〜754) | 754 (post=20) | **失敗** (`max_consec=19`) |
| **seed 3002** (`cand_cart`) | 840 | 847 (1 stepのみ) | 指先の一時的接触（`is_grasped=True`, `is_released=False`） | **13 steps** (848〜860) | 860 (post=20) | **失敗** (`max_consec=13`) |
| **seed 3011** (`hand_guard`) | 868 | 872〜874 (3 steps) | キューブの微動・接触判定（`obj_stat=False`） | **14 steps** (875〜888) | 888 (post=20) | **失敗** (`max_consec=14`) |

### 【結論】：観測窓（打ち切り）の仕様による誤判定
現行ラッパー `HoldAfterSuccess` は「初回成功から 20 steps 経過した時点で無条件にエピソードを終了（terminated=True）させる」仕様であった。  
そのため、解放直後に 1〜3 steps の過渡的な指先接触や微動が生じると、**その後どれほど完全に手放して静止を維持していても、残りステップ数（19, 13, 14）で強制終了されてしまい、「連続20 steps成功」を満たす前に Wrapper 自身が機会を奪っていた**ことが判明した。

---

## 3. 新プロトコル `HoldContinuousSuccess` の設計と実装

下位互換性を保ちつつ、真の安定成功を厳密に計測するため、新ラッパー `HoldContinuousSuccess` を実装した（`src/apc_maniskill/protocols.py`）。

- **仕様**:
  - `info["success"]` の連続成功ステップ数（`current_consecutive_success`）をカウント。
  - 連続成功が目標ステップ数（`target_consecutive=20`）に到達した瞬間に `consecutive_success_achieved=True`, `hold_complete=True`, `terminated=True` として終了。
  - 過渡的な微動（接触等）が生じた場合は streak が 0 にリセットされるが、環境の最大ステップ予算（1200 steps）の範囲内で姿勢が整定し、連続20 steps を満たせば成功と判定される。

---

## 4. 最小再実行（Minimal Re-runs）による実証結果

新プロトコル（`--consecutive-success-steps 20`）を用いて、該当 3 条件の最小再実行を実施した。

| 条件 (Config) | Seed | 初回成功 step | 達成 step | 連続成功 streak | 最終成否 | 安定達成 | 総ステップ数 |
|---|---|---|---|---|---|---|---|
| **mlp_guarded** (限定MLP) | 3012 | 735 | 756 | **20 steps 達成** | **成功** | **達成 (True)** | 756 (旧 755) |
| **cand_cart** (Candidate CART) | 3002 | 841 | 868 | **20 steps 達成** | **成功** | **達成 (True)** | 868 (旧 861) |
| **hand_guard** (手設計ガード) | 3011 | 869 | 895 | **20 steps 達成** | **成功** | **達成 (True)** | 895 (旧 889) |

新プロトコルにより、seed 3012 の限定 MLP、seed 3002 の Candidate CART、および seed 3011 の手設計ガードの全条件において、解放直後の過渡現象後に完全に整定し、**自律的に連続20 steps のタスク達成を維持できること**が明確に実証された。

これにより、W0 / T00（安定成功の再集計と最小再実行）は完全に達成され、次のタスク（W1 運搬の一般化、W2 上流能力の追加）に向けた厳密な評価基準が確立された。
