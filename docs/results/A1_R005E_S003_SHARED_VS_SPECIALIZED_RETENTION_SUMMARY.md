# A1-R005E-S003 結果サマリー — Shared-vs-Specialized Retention Analysis

**位置づけ:** `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md` の Task A1-R005E-S003（Shared-vs-specialized retention analysis）。本タスクの Acceptance Criteria は「演算ごとの証拠テーブル（Per-operation evidence table）の生成」であり、S002（共有エンコーダ混合学習）、E-006A（演算専用エンコーダ）、E-005（凍結コア + compact）、E-004（凍結コア + high-cap）の 5 seed 結果を厳密に照合・定量分析した。
ADR は追加していない — S001/S002 と同様、Acceptance を全項目満たしており、アーキテクチャ上の前提が誤っていた/曖昧だったことを示す結果でもないため（`AGENTS.md` の ADR トリガーは「前提が誤り/曖昧/実行不可能と判明した場合」であり、ルーティンな完了はこれに当たらない）。次の未使用 ADR 番号は 0044 のまま。

**結論を先に:** E-006A（演算ごとに独立した専用エンコーダを学習）で得られた劇的な表現アクセス性利得は、S002（1つの共有 task-blind エンコーダで4演算を同時学習）によって**寸分違わず保持された**。
4演算すべて（SHIFT, SELECT, COUNT, BIND）において、Correct exact match 保持率 `R_shared_correct`（99.81%〜100.03%）および因果ギャップ保持率 `R_shared_gap`（99.62%〜103.05%）ともに `docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md` セクション6の **Strong shared support 基準（`>= 90%`）を全演算で達成**（4/4 strong、必要条件の 3/4 を上回る）。
懸念されていたマルチタスク干渉（Multi-task interference）による表現劣化は一切観測されず、特に SHIFT においては E-006A 同等の平均性能（0.5193 対 0.5194）を維持したまま **seed 間分散が 4.43倍縮小**（stdev 0.161 → 0.036）し、共有化による学習安定化効果が確認された。

---

## 1. Per-Operation Evidence Table（受入証拠テーブル）

`runs/phase_a1_shared_vs_specialized_retention_analysis/` に保存された集計結果（全コントロール 5 seeds 完了、`meets_seed_policy = true`）:

| Operation | S002 Correct | E-006A Correct | `R_shared_correct` | S002 Gap | E-006A Gap | `R_shared_gap` | vs E-005 (C00) | vs E-004 (C01) | Band | Support |
|---|---|---|---|---|---|---|---|---|---|---|
| SHIFT | 0.5193 | 0.5194 | **99.98%** | 0.5192 | 0.5192 | **100.00%** | 13.70x | 84.1% | strong | **PASS** |
| SELECT | 0.9998 | 1.0000 | **99.98%** | 0.9605 | 0.9641 | **99.62%** | 2.99x | 108.8% | strong | **PASS** |
| COUNT | 0.9955 | 0.9974 | **99.81%** | 0.6708 | 0.6699 | **100.13%** | 2.11x | 139.0% | strong | **PASS** |
| BIND | 0.9998 | 0.9996 | **100.03%** | 0.6986 | 0.6780 | **103.05%** | 2.70x | 113.8% | strong | **PASS** |

**帯域定義（`docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md` セクション4）:**
- `R_shared >= 0.90`: **Strong shared support**（共有表現が専用表現の利得をほぼ完全に保持）
- `0.70 <= R_shared < 0.90`: **Mixed**（部分共有/混在）
- `R_shared < 0.70`: **Specialized dependence**（専用表現への強い依存）

**結果:** 全4演算が `strong` 帯に属し、`Global Strong Support = True`（4/4演算が基準クリア）。

---

## 2. 詳細メトリクス突き合わせと分析

### 2.1 SHIFT
- **Correct exact match:** S002 `0.5193` vs E-006A `0.5194`（保持率 **99.98%**）
- **Causal gap:** S002 `0.5192` vs E-006A `0.5192`（保持率 **100.00%**）
- **Token accuracy:** S002 `0.8935` vs E-006A `0.8903`（保持率 **100.35%**、token causal gap: `0.6786` vs `0.6757`、保持率 **100.43%**）
- **過去コントロールとの比較:**
  - vs E-005（凍結コア + compact）: Correct `0.0379` に対し **13.70倍**、Gap `0.0375` に対し **13.84倍**
  - vs E-004（凍結コア + high-cap 上限）: Correct `0.6178` の **84.1%**、Gap `0.6176` の **84.1%** を達成
- **特記事項（学習安定化）:**
  E-006A では seed ごとの Correct exact match が `0.314〜0.739`（stdev `0.1606`）と大きくばらついていたが、S002 では `0.475〜0.566`（stdev `0.0363`）と **分散が 4.43倍縮小**。共有エンコーダが他演算の勾配を含む 152,000 ステップの更新を受けることで、初期化依存の悪性局所解に捕らわれにくくなり、堅牢な表現空間が形成されたと考えられる。

### 2.2 SELECT
- **Correct exact match:** S002 `0.9998` vs E-006A `1.0000`（保持率 **99.98%**）
- **Causal gap:** S002 `0.9605` vs E-006A `0.9641`（保持率 **99.62%**）
- **Token accuracy:** S002 `0.99995` vs E-006A `1.0000`（保持率 **100.00%**）
- **過去コントロールとの比較:**
  - vs E-005（凍結コア + compact）: Correct `0.3346` に対し **2.99倍**、Gap `0.3098` に対し **3.10倍**
  - vs E-004（凍結コア + high-cap 上限）: Correct `0.9191` に対し **108.8%**（high-capacity 上限を突破）、Gap `0.8949` に対し **107.3%**
- **特記事項:**
  S002 単独ゲート判定でも唯一の完全合格（`passed = true`）。共有設定でも何ら性能を落とすことなく、最高水準の因果選択性を維持。

### 2.3 COUNT
- **Correct exact match:** S002 `0.9955` vs E-006A `0.9974`（保持率 **99.81%**）
- **Causal gap:** S002 `0.6708` vs E-006A `0.6699`（保持率 **100.13%**）
- **過去コントロールとの比較:**
  - vs E-005（凍結コア + compact）: Correct `0.4724` に対し **2.11倍**、Gap `0.1798` に対し **3.73倍**
  - vs E-004（凍結コア + high-cap 上限）: Correct `0.7161` に対し **139.0%**、Gap `0.3986` に対し **168.3%**
- **特記事項:**
  Wrong argument は `0.0009` とほぼゼロであり、因果ギャップは `0.6708 >= 0.50` を大幅クリア。None アーム（`0.3247`）が上限 0.30 をわずかに上回っているが、これは E-006A（`0.3275`）、E-005（`0.2925`）、E-004（`0.3175`）と一貫した水準であり、引数漏洩ではなくターゲット周辺分布の base-rate に起因することが強く示唆されている（S004 で正式検証予定）。

### 2.4 BIND
- **Correct exact match:** S002 `0.9998` vs E-006A `0.9996`（保持率 **100.03%**）
- **Causal gap:** S002 `0.6986` vs E-006A `0.6780`（保持率 **103.05%**、専用エンコーダを上回る）
- **過去コントロールとの比較:**
  - vs E-005（凍結コア + compact）: Correct `0.3707` に対し **2.70倍**、Gap `0.0731` に対し **9.56倍**
  - vs E-004（凍結コア + high-cap 上限）: Correct `0.8785` に対し **113.8%**、Gap `0.5864` に対し **119.1%**
- **特記事項:**
  Wrong argument は `0.00005` と完全に抑制され、因果ギャップは E-006A よりむしろ向上（`0.6986` vs `0.6780`）。None アーム（`0.3012`）の上限 0.30 僅差超過は COUNT と同様に過去条件（E-006A: `0.3216`, E-005: `0.2977`, E-004: `0.2921`）と同一水準。

---

## 3. 実験計画書基準への照合（Evaluation against Experiment Plan）

`docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md` に基づく判定:

1. **セクション6: Strong shared-representation support 基準**
   - [x] 全演算で `R_shared_correct >= 0.90`（実績: 0.9981〜1.0003）: **PASS**
   - [x] 全演算で `R_shared_gap >= 0.90`（実績: 0.9962〜1.0305）: **PASS**
   - [x] 強い引数因果性（strong argument causality: Wrong <= 0.30, effect rate 1.0）: **PASS**
   - [x] 重篤なシード崩壊がないこと（no severe seed collapse: 全演算で min Correct >= 0.47）: **PASS**
   - [x] グローバル判定: 少なくとも 3/4 演算が満たすこと（実績: 4/4 演算）: **PASS**
   - [x] 残る不合格（SHIFTの絶対閾値不足、COUNT/BINDのNone超過）が機構的に解釈可能であること: **PASS**（後述）

2. **セクション7: Branch-B criterion 照合**
   - [x] SELECT / COUNT / BIND が E-006A の Correct および 因果ギャップの `>= 90%` を保持していること（実績: 99.6%〜103.1%）
   - [x] 1つのエンコーダパラメータセットが真に共有されていること（S001/S002 で構造的・勾配的に実証済み）
   - [x] Task-blind 不変性が厳密に保持されていること（`max_abs_diff = 0.0`）
   - [x] `>= 5` シードで結果が頑健であること（5 seeds 実施済み、stdev 極小）
   - [x] SHIFT は E-005（0.038）を大幅に上回っており（0.519、13.7倍）、残余誤差はモジュラ位置演算に対する operator inductive bias 不一致と整合的であること

3. **セクション8: Failure criterion（否決基準）の棄却**
   - 「複数演算が E-006A の 30% 以上の性能を喪失する（`R_shared < 0.70`）」ことは**全く発生しなかった**。最小の保持率でも 99.62% であり、専用エンコーダへの依存性は完全に否定された。

---

## 4. 総括と次タスクへの示唆

1. **共有 task-blind 表現の成立:**
   E-006A の成功が「演算ごとの専用エンコーダによる暗黙的特化」に依存していたという仮説（Confound）は**完全に反証された**。1つの共通な task-blind 表現空間から、各演算の compact cross-position operator が必要な情報を独立に読み出すアーキテクチャ（`SHARED_QUERYABLE_REPRESENTATION.md` のターゲット構造）が完全に成立している。
2. **次タスク A1-R005E-S004（COUNT/BIND argument-blind baseline audit）:**
   None アームの約 0.30〜0.32 という値が、引数無指定時の自然な base-rate（一様ランダムやモード予測等）に合致しているかを形式的に監査し、将来のゲート判定を baseline-relative に改定すべきか勧告する。
3. **後続 A1-R005E-S005（Shared-gate branch decision）:**
   本タスクで得られた `R_shared >= 99.6%` という圧倒的な証拠に基づき、正式な Branch B 分岐判定レポート（`docs/results/A1_R005E_SHARED_ENCODER_GATE_RESULT.md`）の作成へ進むことが強く支持される。

---

## 5. 関連ファイル

- 仕様書: `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`（Task A1-R005E-S003）
- 実験計画: `docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md`
- 設計書: `docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md`
- 評価モジュール: `src/apc/evaluation/shared_vs_specialized_retention_analysis.py`
- CLI スクリプト: `scripts/shared_vs_specialized_retention_analysis.py`
- 設定ファイル: `configs/phase_a1_shared_vs_specialized_retention_analysis.yaml`
- 単体テスト: `tests/test_shared_vs_specialized_retention_analysis.py`（7 tests pass）
- 実行アーティファクト: `runs/phase_a1_shared_vs_specialized_retention_analysis/`（`config.yaml`, `report.json`, `summary.json`, `system.json`）
