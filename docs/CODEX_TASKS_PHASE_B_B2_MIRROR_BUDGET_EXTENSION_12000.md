> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# AI Coding Task — MIRROR_HALVES P Cumulative Budget Extension (6000 → 12000)

**正式ID：B-C005REC-004G**
**版：1.0 / 2026-09-09 / 状態：ユーザーの明示指示により実装・実行を許可。**
**位置：REC-004Fの結果（`MIXED_ACROSS_INITS`, `next_repair_contract.md`超の追加診断ではない）を受け、
ユーザーが直接チャットで指示した限定実験。REC-004D/Cの`OPTIMIZATION_PROGRESS_OBSERVED`
（終盤の学習曲線が6000stepで未収束）という所見に基づく「Pの未収束仮説」を検証する。**

## 0. このタスクで行うこと・行わないこと

REC-004DのP（`P_LENGTH_POSITION_BIAS`, I01〜I05, 全5本）を、各々の保存済み
**完全training state（step=6000）** から再開し、**累計12000stepまで**（追加6000 updates）
同じレシピで学習を継続する。変更するのは累計学習予算だけであり、Core・他primitive・
アーキテクチャ・feature・optimizer設定・schedule規則・訓練データ分布は一切変更しない。

比較対象は各Pの**自分自身**のstep=6000状態であり（paired self-comparison）、Uを5本
再学習し直すことは行わない。ここで検証するのは位置biasの有無（REC-004Dで既に確認済み）
ではなく、「既に有効だったPが6000stepでは未収束だっただけではないか」という仮説である。

本タスクは**限定実験**である。全5本がstep=12000で既存validation floor（0.95）へ到達しても、
本タスク単独では候補採用・child bundle組立・RG3再判定を行わない。これらは別の明示指示を
要する。理由：これまでの全REC-00N系タスクと同じ「一件ずつ明示承認」原則を維持するため、
学習予算の延長という一つの変更と、候補採用という別の意思決定を混在させない。

```text
REC-004D：全5ペアで改善、Pは0/5合格（最良I05=0.9023）、RG3 NOT_EXECUTED（保存）
    ↓
REC-004E：位置score残差を診断。length_10=SCORE_COMPONENT_INTERACTION_LEAD等、
          OPTIMIZATION_PROGRESS_OBSERVED（6000stepで未収束）。次修正契約は
          oracle-attention probeのみ提案（PROPOSED_NOT_AUTHORIZED）
    ↓
REC-004F：oracle-attention probeを実行（ユーザー承認）。結果MIXED_ACROSS_INITS
          （3/5はattention自体がbottleneck、I05はattention下流に残差）
    ↓
REC-004G（本タスク、ユーザーが直接チャットで指示。正式なB-C005REC-005
          「Five-Model Coherent Cohort」の予約番号は消費しない）
 A. 保存済みstep=6000 training state（optimizer/scheduler/RNG込み）を読込み、
    REC-004D自身の記録値に対しsource replayで一致を確認
 B. 同じrecipe・同じper-step data関数を、step=6001から12000まで継続
    （新規optimizer updates=6000×5=30000、schedule/RNGはリセットしない）
 C. 各Pの step=6000 vs step=12000 のpaired self-comparison、既存floor再判定
 D. 結果を報告してSTOP。候補採用・child bundle・RG3再判定はこの一件では行わない
    ↓
STOP。B-C005REC-005（Five-Model Coherent Cohort）以降、R3-011、B-C006、
Task Inferenceは引き続きblocked。
```

### 0.1 このタスクで行わないこと（明示禁止）

- oracle attention（REC-004Fの`_oracle_attention`）を訓練へ使うこと。通常の実forwardのみ。
- I01〜I05以外の新しい初期化、または一部initの除外・並べ替え・「良さそうなものだけ」の選択。
- 新しいhead・新しいloss項・アーキテクチャ変更・feature変更の追加。
- 12000を超えるstepへの延長（上限に達したら必ず停止する）。
- Uアーム5本の再学習（不要、比較対象は各P自身の6000-step状態）。
- 全5本の平均化、最良initの選択的採用、途中checkpointの恣意的採用。
- 過去run（REC-004D `run_001`）のいかなるファイルの上書き・削除・変更。
- floor到達の有無にかかわらず、候補採用・child bundle組立・RG3再判定・独立query生成。

---

## 1. 根拠・継承・技術契約

### 1.1 継承する既存事実（実ファイルから確認済み）

- REC-004Dは`runs/phase_b_b2_model_bundle_recovery/rec004d/run_001/{I01..I05}/P_LENGTH_POSITION_BIAS/training_states/step6000.pt`に、各Pの**完全な再開可能state**を保存済み：
  `primitive_state_dict`、`optimizer_state_dict`（AdamW）、`scheduler_state_dict`
  （`CosineAnnealingLR(T_max=1000, eta_min=1e-5)`、`last_epoch=6000`を含む）、
  `cpu_rng_state`、`cuda_rng_state`、`cumulative_updates=6000`、`init_id`、`arm`。
- 同じrun直下の`learning_curve.jsonl`に、各(init_id, arm, step)ごとの
  `checkpoint_state_hash`（`primitive_state_dict`のcanonical hash）と
  `existing_validation.correct_exact_match`が記録済み。本タスクの起点整合性検査に使う。
- per-step訓練データは`ibc._generate_step_training_examples(seed, step, operation, ...)`
  という**(seed, step)だけの純粋関数**（内部で独立な`random.Random`を使用し、
  ambient torch/randomのglobal stateに依存しない）。step=6001以降を同じseed・同じ関数で
  呼ぶだけで、6000以前の反復に戻ることなく「レシピの次のstep」が一意に決まる。
- schedule規則：`CosineAnnealingLR(T_max=1000, eta_min=1e-5)`は6000stepの間に
  3周期（各1000stepで谷、次の1000stepで山に戻る）を「機械的に延長」する形で運用されてきた
  （REC-004B用語）。`scheduler.load_state_dict(...)`で`last_epoch=6000`から再開すれば、
  この式は特別な分岐なしにそのまま7周期目以降へ続く。
- 現在の実行環境はCUDA利用可（`NVIDIA GeForce RTX 5060 Ti`）で、REC-004D実行時の
  `system.json`記載deviceと同一。

### 1.2 変更する一箇所・変更しないもの

| 項目 | 内容 |
|---|---|
| 対象 | REC-004DのP/I01〜I05、全5本（Uは対象外） |
| 開始点 | 各Pの保存済み6000-step **完全training state**（重み＋optimizer＋scheduler＋RNG） |
| 変更するもの | 許可する累計学習予算だけ（6000 → 12000） |
| 追加予算 | 各初期化につき6000 updates、累計12000まで（新規optimizer updates=30000合計） |
| 学習するparameter | `CrossPositionLengthBiasPrimitive`の全parameter（本体cross-attention＋既存192-parameter position bias net）。REC-004Dの`primitive.parameters()`と同一集合 |
| 固定するもの | Core、他15 primitive、REC-004Aの固定3候補、`CrossPositionLengthBiasPrimitive`のアーキテクチャ、4入力feature式、optimizer種別・LR・weight decay・grad clip、`CosineAnnealingLR`のT_max/eta_min、`checkpoint_interval=500`、per-step訓練データ生成関数、既存validation split・floor（0.95） |
| 禁止 | oracle attentionの訓練利用、初期化の選び直し、head／loss追加、12000超の延長 |
| 比較対象 | 各P自身のstep=6000状態（self-comparison）。Uの再学習はしない |

12000という上限は**今回の新しい提案値**であり、既存タスクで事前承認済みの予算ではない
（ユーザー指示原文のとおり、ここに明記する）。

### 1.3 候補採用ルール（変更しない）

REC-004Dの採用規則（P全5本の終端existing-validation EM >= 0.95で候補確定、
最良init選択は禁止）を、この延長実験の成績で緩和しない。全5本がstep=12000で
floorへ到達しても、本タスクは`candidate_status`を機械的に報告するのみで、
`selected_init`・`selected_intervention`・`child_bundle`は`null`、
`rg3_recheck`は`NOT_EXECUTED`のまま維持する。

---

## 2. 不変条件・許可範囲

読み取り専用で扱う対象：REC-004D `run_001`配下の全ファイル（`training_states/step6000.pt`を
含む）、parent bundle全16 slot、Core、REC-004Aの固定3候補、REC-004Cの初期state、
router／keys／ArgumentScorer／controller／verifier、全共有cache。これらへの書き込み・
上書き・削除は一切行わない。新規成果物は全て`runs/phase_b_b2_model_bundle_recovery/rec004g/<run_id>/`
配下にのみ書く。

学習を許可する対象は、REC-004Dのstep=6000状態から複製した**新しいprimitiveインスタンス**
だけである。学習中、Core・eval_bank内の他primitiveへの書き込みは行わない
（REC-004Dと同じ`eval_bank.replace_primitive`/restoreパターンを使う）。

---

## 3. 段階A — 起点の整合性確認（新規optimizer updatesの前）

各initについて：

1. `training_states/step6000.pt`を読み込み、`cumulative_updates==6000`、
   `init_id`・`arm`が一致することを確認する。欠落・不一致は`SOURCE_ARTIFACT_UNAVAILABLE`。
2. 読み込んだ`primitive_state_dict`のcanonical hashを、`learning_curve.jsonl`の
   該当行（`init_id`, `arm="P_LENGTH_POSITION_BIAS"`, `step=6000`）に記録された
   `checkpoint_state_hash`と比較する。不一致は`SOURCE_REPLAY_MISMATCH`でSTOP
   （このinitについて新規updateを開始しない）。
3. この状態で（新規update前に）既存validation splitを評価し、`learning_curve.jsonl`
   記載の`existing_validation.correct_exact_match`と1e-9未満の差で一致することを
   確認する（`SOURCE_REPLAY_MISMATCH`ゲート）。

全5本でこの整合性が確認できて初めてStage Bへ進む。1本でも欠落・不一致があれば、
その旨を明示して報告し、影響を受けたinitの学習を開始しない
（他の整合するinitがあっても、勝手に「5本のうち動くものだけ」を候補化しない）。

---

## 4. 段階B — 継続学習（新規optimizer updates）

各initについて、`optimizer.load_state_dict(...)`・`scheduler.load_state_dict(...)`・
`torch.set_rng_state(...)`／`torch.cuda.set_rng_state(...)`で完全復元した後、
step=6001から12000まで、REC-004Dの`run_one_arm`と同一のループ本体
（同じ`_generate_step_training_examples`呼び出し、同じforward／loss／
`clip_grad_norm_`／`optimizer.step()`／`scheduler.step()`）を継続する。

`checkpoint_interval=500`ごとに、REC-004Dと同じ形式で
`checkpoints/step{N}.pt`（primitive state のみ）と
`training_states/step{N}.pt`（optimizer/scheduler/RNG込みの完全state）を保存し、
既存validation・train-fit・length別breakdownを評価する。step=12000到達時のみ、
REC-004Dと同じ`_position_level_breakdown`・`run_bias_ablation`を実行する。

損失がnon-finiteになった場合はその時点で打ち切り、`diverged_at`を記録する
（REC-004Dと同じ扱い）。

---

## 5. 段階C — Paired self-comparison と floor再判定

各initについて、step=6000とstep=12000の状態を同じ既存validation例（1024例）で
評価し、EM・token accuracy・loss・both/only混同表・length別内訳のpaired deltaを
計算する。集計は5本を個別に報告し、平均や最良選択で代表させない。

`candidate_status_at_step_12000`：REC-004Dと同じ0.95 floorを、全5本の
step=12000 existing-validation EMへ機械的に適用して報告する
（`FIVE_INIT_VALIDATION_FLOOR_PASS_AT_STEP_12000` /
`VALIDATION_TARGET_NOT_MET_AT_STEP_12000`）。**この値がPASSであっても、
`selected_init`・`selected_intervention`・`child_bundle`は`null`のまま、
`rg3_recheck`は`NOT_EXECUTED`のまま維持する。**

---

## 6. 完了条件と状態

```text
implementation_status: COMPLETE / PARTIAL / BLOCKED
source_replay_status: VERIFIED / SOURCE_REPLAY_MISMATCH / SOURCE_ARTIFACT_UNAVAILABLE
extension_training_status: COMPLETE / PARTIAL (per-init diverged_at) 
candidate_status_at_step_12000: FIVE_INIT_VALIDATION_FLOOR_PASS_AT_STEP_12000 /
                                 VALIDATION_TARGET_NOT_MET_AT_STEP_12000
new_optimizer_updates: <= 30000 (6000 x 5 inits; less if any init diverges early)
selected_init: null
selected_intervention: null
child_bundle: null
rg3_recheck: NOT_EXECUTED
```

---

## 7. 実装先・テスト・成果物

### 7.1 配置

```text
docs/CODEX_TASKS_PHASE_B_B2_MIRROR_BUDGET_EXTENSION_12000.md   # 本書
src/apc/evaluation/mirror_budget_extension.py                  # 新規
scripts/run_phase_b_b2_model_bundle_recovery.py                # B-C005REC-004G明示dispatch
configs/phase_b_b2_model_bundle_recovery_rec004g.yaml
tests/test_mirror_budget_extension.py
runs/phase_b_b2_model_bundle_recovery/rec004g/<run_id>/
```

```bash
python scripts/run_phase_b_b2_model_bundle_recovery.py --task B-C005REC-004G --config configs/phase_b_b2_model_bundle_recovery_rec004g.yaml
```

### 7.2 必須CPU/小fixtureテスト

1. optimizer/scheduler/RNGの完全復元により、`scheduler.get_last_lr()`が
   REC-004Dの`step6000.pt`記録値と一致すること。
2. 起点hash不一致・欠落ファイルが`SOURCE_REPLAY_MISMATCH`/`SOURCE_ARTIFACT_UNAVAILABLE`
   を正しく発生させ、学習を開始しないこと。
3. per-step訓練データ生成が(seed, step)のみに依存し、step=6001以降が
   6000以前のいずれの値とも重複しないこと。
4. Core・他14 primitive・REC-004Aの固定3候補が学習前後で不変であること
   （freeze audit）。
5. REC-004D `run_001`配下のいかなるファイルも変更されないこと（byte-identical監査）。
6. oracle attentionへの依存が存在しないことのsource-scanテスト（REC-004Fの
   `_oracle_attention`をimportしない／呼ばないこと）。
7. optimizerの新規生成なしに`load_state_dict`だけで再開できること（dynamic guardで
   新規updateがゼロ回未満／12000超に及ばないことを検査）。
8. floor到達時でも`selected_init`/`child_bundle`が`null`、`rg3_recheck`が
   `NOT_EXECUTED`のまま報告されること。

その後、repo標準検証：

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

### 7.3 保存物

```text
config.yaml / system.json / summary.json / report.md
source_replay.json (per-init整合性確認)
{init_id}/checkpoints/step{N}.pt / {init_id}/training_states/step{N}.pt
learning_curve.jsonl / lr_trace.jsonl
paired_extension_comparison.json / candidate_decision.json
per_length_position_metrics.json / bias_ablation.json
freeze_audit.json / side_effect_audit.json / cost_accounting.json
```

### 7.4 完了報告

```text
Task: B-C005REC-004G
Source replay (5 inits): hash一致 / EM一致:
新規optimizer updates合計 / 各initのdiverged_at:
Paired self-comparison (step6000 -> step12000), 5本個別:
candidate_status_at_step_12000:
Core / 他primitive / REC-004D run_001の不変性:
tests / repo標準検証 / new ADR:
selected_init=null / child_bundle=null / RG3 NOT_EXECUTED:
```

**STOP：延長学習の結果を報告して、この一件を終了する。floorへ到達した場合でも、
候補採用・child bundle組立・RG3再判定・B-C005REC-005（Five-Model Coherent
Cohort）以降へは自動移行しない。**
