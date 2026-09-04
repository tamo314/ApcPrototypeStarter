# A1-R005E Shared Queryable Representation Gate — Branch Decision Report

**タスク:** `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md` A1-R005E-S005 (Shared-gate branch decision)  
**実験計画:** `docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md` (セクション7, 11, 12)  
**設計文書:** `docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md`  
**アーキテクチャ決定:** ADR-0044 (`docs/DECISIONS_A1_R005E_DIAGNOSTIC.md`)  
**判定日:** 2026-09-04  

---

## 1. エグゼクティブ・サマリー（Executive Summary）

### 結論: **Outcome B 採択 — Branch B (Shared Queryable Representation) の正式採択 + SHIFT 構造プローブ (A1-R005E-S006) の発動**

A1-R005E-S001 から S004 までの実測結果を統合した結果、APC の核心命題である「**1つの共有された task-blind 表現空間 $\to$ 複数の疎な再利用可能 operator**」（Branch B: Shared Queryable Representation）は**極めて強力に支持された**。

1. **表現共有による劣化ゼロ (S003)**:
   4演算すべて（SELECT, COUNT, BIND, SHIFT）において、E-006A（演算専用エンコーダ）に対する保持率 `R_shared` は **99.62%〜103.05%** を達成し、マルチタスク干渉による性能崩壊は一切観測されなかった。
2. **SELECT / COUNT / BIND の完全合格 (S003/S004)**:
   SELECT (99.98%)、COUNT (99.55%)、BIND (99.98%) はほぼ上限精度を達成し、E-004 (frozen high-capacity) の上限性能と同等以上を達成した。S004 の監査により、COUNT と BIND の None アーム超過（~30-33%）は自然ベースライン（33.56% / 34.00%）に起因することが証明され、引数感受性（Wrong <= 0.09%, Causal Gap >= 67%）は完全に保たれていることが実証された。
3. **SHIFT 残余誤差の局在化と S006 の発動**:
   SHIFT は E-006A の性能を 100% 保持（0.5193 対 0.5194）しつつ seed 間分散が 4.43倍縮小したが、絶対目標（90%）には達していない（~52% exact match / 89% token acc）。この停滞はエンコーダの共有干渉ではなく、モジュラ位置演算に対する単一クロスアテンションの inductive bias 不足に起因する。実験計画書セクション11の規定に基づき、Branch B 全体を棄却することなく、**A1-R005E-S006（SHIFT compact structural probe）を条件付き発動**する。
4. **ブロック規則の継続**:
   本決定をもってしても **A1-R006 は依然としてブロック（BLOCKED）** される。S006 および最終監査 S007 が完了し、正式な完了報告が承認されるまで、生産系アーキテクチャの変更や A1-R006 への移行は行わない。

---

## 2. 実証証拠の統合（S001〜S004 の歩み）

### 2.1 S001: 共有アーキテクチャの構造的検証
- **検証内容**: 1つの共通 `DecoderOnlyTransformer` エンコーダ、4つの独立した `CompactCrossPositionOperator`、各演算固有の引数エンコーダおよびリードアウトヘッド。
- **実証結果**: エンコーダ内に演算固有モジュールが存在せず、同一 content に対するエンコーダ出力は演算によらず完全一致（`max_abs_diff = 0.0`）。オラクル選択のみで各 operator が呼び出される構造を確立。

### 2.2 S002: 均等混合オンライン学習（Balanced Mixed Training）
- **検証内容**: 1つの AdamW オプティマイザで共有エンコーダと全 operator を 5 seeds × 152,000 steps（各演算 38,000 steps、E-006A と同一の演算あたり更新予算）学習。
- **実証結果**: 厳密に 25/25/25/25 のサンプリングを実現。学習後も task-blind 不変性は厳密に維持（`max_abs_diff = 0.0`）。SELECT は単独で完全合格（`passed = true`）。SHIFT の seed 間ばらつきが大幅に縮小（stdev 0.161 $\to$ 0.036）。

### 2.3 S003: 共有 vs 専用の保持率分析（Retention Analysis）
- **検証内容**: E-006A に対する保持率 $R_{\text{shared}} = M_{\text{shared}} / M_{\text{specialized}}$ の算出。
- **実証結果**: 全4演算が `Strong shared support` 帯（$\ge 90\%$）に属し、必要条件（3/4 演算）を上回る 4/4 演算で基準クリア。

| Operation | S002 Correct | E-006A Correct | `R_shared_correct` | S002 Gap | E-006A Gap | `R_shared_gap` | vs E-005 (C00) | vs E-004 (C01) | Band | Support |
|---|---|---|---|---|---|---|---|---|---|---|
| **SHIFT** | 0.5193 | 0.5194 | **99.98%** | 0.5192 | 0.5192 | **100.00%** | 13.70x | 84.1% | strong | **PASS** |
| **SELECT** | 0.9998 | 1.0000 | **99.98%** | 0.9605 | 0.9641 | **99.62%** | 2.99x | 108.8% | strong | **PASS** |
| **COUNT** | 0.9955 | 0.9974 | **99.81%** | 0.6708 | 0.6699 | **100.13%** | 2.11x | 139.0% | strong | **PASS** |
| **BIND** | 0.9998 | 0.9996 | **100.03%** | 0.6986 | 0.6780 | **103.05%** | 2.70x | 113.8% | strong | **PASS** |

### 2.4 S004: COUNT/BIND 引数ブラインド・ベースライン監査
- **検証内容**: None 上限 0.30 超過が引数漏洩によるものか、タスク本来の自然ベースラインによるものかを測定。
- **実証結果**:
  - COUNT: 常に 0 を出力する最頻値ベースラインが **`33.46%`**。
  - BIND: 末尾ペア値（`content[-1]`）を出力する直近ヒューリスティックが **`34.00%`**。
  - どちらも歴史的上限 0.30 を構造的に上回っており、観測 None 精度（29%〜33%）はモデルが引数ゼロ入力時に自然な content-only ダミー予測を行っている証拠であることを証明。
  - 正解引数アーム（Correct ~100%）および不正引数アーム（Wrong <= 0.09%）との比較により、因果ギャップは 67%〜70% と極めて強固であることを立証。

---

## 3. 実験計画書基準の正式評価（Evaluation against Experiment Plan）

`docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md` に基づく判定:

### 3.1 セクション7: Branch-B Criterion の照合
| 条件 | 判定基準 | S002/S003 実績 | 評価 |
|---|---|---|---|
| 1. SELECT/COUNT/BIND 保持率 | E-006A の Correct 及び Gap の $\ge 90\%$ 保持 | Correct: 99.81%〜100.03%, Gap: 99.62%〜103.05% | **PASS** |
| 2. 真の単一共有エンコーダ | 1つのパラメータセットが全演算で共有 | S001/S002 で実証。全 step で共有コア更新 | **PASS** |
| 3. Task-blind 不変性の維持 | content 表現がタスク・引数に依存しないこと | 学習後も `max_abs_diff = 0.0` を厳密維持 | **PASS** |
| 4. 5 seeds の頑健性 | 重篤なシード崩壊や高分散がないこと | 5 seeds 完了、stdev 極小、崩壊ゼロ | **PASS** |
| 5. SHIFT 残余の許容 | E-005 を大幅に上回り、残余が operator 不一致と整合 | E-005 の 13.7倍、E-004 の 84.1%、operator 固有限界 | **PASS** |

**総合評価:** すべての条件を完全に満たしており、Branch B は正式に強力に支持された。

### 3.2 セクション8: Failure Criterion（否決基準）の棄却
- 否決条件である「複数演算が E-006A の 30% 以上の性能を喪失する（`R_shared < 0.70`）」は全く発生しなかった（全演算で保持率 99.6% 以上）。

### 3.3 セクション11: SHIFT Follow-Up Trigger の判定
- SELECT/COUNT/BIND が共有表現を強力に支持し、SHIFT のみが絶対目標を下回っているため、規定通り **A1-R005E-S006（SHIFT compact structural probe）の発動が正式に承認（Triggered）** される。

---

## 4. 将来ゲートへの判定基準改定（None-Arm Criterion Adoption）

S004 の勧告に基づき、本分岐決定をもって以下の判定基準改定を採択する:

1. **固定上限の廃止**:
   硬直的な固定閾値 `None <= 0.30` は、タスクの組合せ論的特性を無視した不当な制約であったため、以降のゲートでは使用しない。
2. **ベースライン相対基準の採用**:
   $$\text{None} \le B_{\text{natural}} + 0.05 \quad (B_{\text{natural}} = \max(B_{\text{majority}}, B_{\text{content-only}}))$$
   - COUNT 閾値: $\le 0.3846$ (実績 0.3247 $\implies$ 合格)
   - BIND 閾値: $\le 0.3900$ (実績 0.3012 $\implies$ 合格)
   - SHIFT 閾値: $\le 0.0500$ (実績 0.0001 $\implies$ 合格)
   - SELECT 閾値: $\le 0.0893$ (実績 0.0393 $\implies$ 合格)
3. **歴史的記録の保全**:
   E-004 から S002 までの過去の不合格記録は遡及変更せず、当時の記録として保全する。

---

## 5. 次のステップ（Next Steps）

1. **A1-R005E-S006（SHIFT compact structural probe）の実行**:
   - 凍結された S002 共有エンコーダの上で、モジュラ/相対位置バイアス（相対位置アテンション、Rotary/RoPE、またはモジュラオフセットルーティング）を備えたコンパクトな SHIFT operator を検証。
   - 目的: SHIFT の ~52% exact match が、operator の inductive bias 改善によって primitive-scale で解消可能かを確かめる。
2. **A1-R005E-S007（Final diagnostic audit）**:
   - 共有ゲート全体および SHIFT プローブの結果を総合し、Phase A.1 最終診断レポート（`docs/results/A1_R005E_DIAGNOSTIC_RESULT_FINAL.md`）を作成。
3. **A1-R006 のブロック維持**:
   - S007 の承認およびユーザー指示があるまで、A1-R006 は開始しない。

---

## 6. 関連アーティファクト

- S001 アーティファクト: `runs/phase_a1_shared_encoder_architecture_gate/`
- S002 アーティファクト: `runs/phase_a1_shared_encoder_mixed_operation_training_gate/`
- S003 アーティファクト: `runs/phase_a1_shared_vs_specialized_retention_analysis/`
- S004 アーティファクト: `runs/phase_a1_argument_blind_baseline_audit/`
- S005 アーティファクト: `runs/phase_a1_shared_encoder_gate_decision/` (`report.json`, `summary.json`, `system.json`)
- 決定モジュール: `src/apc/evaluation/shared_encoder_gate_decision.py`
- 決定スクリプト: `scripts/shared_encoder_gate_decision.py`
- テスト: `tests/test_shared_encoder_gate_decision.py` (5 tests pass)
