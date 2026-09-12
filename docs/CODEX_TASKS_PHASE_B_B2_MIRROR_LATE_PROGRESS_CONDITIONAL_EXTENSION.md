> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# AI Coding Task — MIRROR_HALVES Late-Stage Length-10 Audit & Conditional Extension

**正式ID：B-C005REC-004H**  
**版：1.0 / 2026-09-09 / 状態：新規指示。プロジェクトでの実装・実験は未実施。**  
**位置：REC-004G／ADR-0102の12000-step未達後、REC-005の前。一件の条件付きタスク。**

## 0. 目的・権限・終了境界

保存済みの8000～12000-step checkpointを用い、**I01～I03、とくにI03の長さ10で、終盤にも学習進行があるか**を確認する。I04／I05は成功側の対照として同じ指標を測る。

事前に定義する継続条件を、未達の全初期化が満たした場合だけ、**P/I01～I05をすべて12000から累計18000 updatesまで同じレシピで継続**する。条件を満たさなければ新規学習0で終了し、残差の根拠と次の一案までを報告する。

本タスクは、全体EMの上昇だけを理由に追加学習しない。原因を容量・位置score・下流のどれかへ無理に確定しない。**条件付き延長の判定規則は今回新設する運用上の基準であり、統計的な収束判定や改善保証ではない。**

```text
REC-004：RG3 FAIL                                     ← 保存
REC-004A～F：各測定・診断結果、RG3未実施                ← 保存
REC-004G：12000-stepで2/5到達、候補なし、RG3未実施      ← 保存
    ↓
REC-004H
 A. source replay・実データ・指標定義の確認
 B. 保存済み8000～12000の長さ別／位置別の終盤監査
 C. 未達初期化ごとの継続条件を機械的に判定
    ├─ 条件成立：全5本を同じ累計18000まで継続 → 終端評価
    └─ 不成立／不明：追加学習しない → 根拠を整理
 D. 結果・未確定事項・次に必要な一件を報告
    ↓
STOP。候補選択・child作成・RG3再判定・REC-005は実行しない。
```

ユーザーが本タスクを指定した場合、前提成立の範囲でA～Dを一件として実行できる。段階ごとの再承認は不要。ただし条件外の修正や予算追加は許可されない。REC-005の既存タスク番号を消費・改番しない。004Hが現repoで別用途に使われていれば `TASK_ID_CONFLICT` として停止する。

**18000-stepで全5本が95%以上になっても、この一件では候補を採用しない。** 全5本条件とI01固定採用という将来の規則を維持しつつ、実際の採用・独立queryは別の明示指示に残す。

---

## 1. 根拠と新設条件

### 1.1 読む資料

- **S1：REC-004G／ADR-0102の実行報告。** [原文snapshot](research/evidence/REC004G_ADR0102_SUPPLIED.md)。今回の開始点。
- **S2：REC-004Dの指示書。** 現repoの `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_LENGTH_POSITION_BIAS.md`。[配布時snapshot](research/evidence/REC004D_TASK_SUPPLIED.md)。構造・5初期化・固定3候補・採用境界を参照する。
- **S3：復旧指示書。** 現repoの `docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`。[配布時snapshot](research/evidence/MODEL_BUNDLE_RECOVERY_TASKS_SUPPLIED.md)。REC-005とRG3の位置づけを参照する。
- **現repoで必ず照合：** `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_BUDGET_EXTENSION_12000.md`、REC-004Gのconfig／JSON／training states、root `AGENTS.md`、現行復旧plan／受入条件／ModelBundle契約。
- [Source notes](research/B2_MIRROR_LATE_PROGRESS_SOURCE_NOTES.md)：原文・計算・新設条件の区別と、未提供資料の範囲。

S1の数値は実行担当者の報告。この指示書作成時に、実repoのコード・重み・学習曲線を独立検査したものではない。S1に記載されたpath・APIは実装時の確認対象。REC-004G指示書の本文は配布資料に含まれておらず、内容を推測して補わない。

### 1.2 引き継ぐ結果

| Init | 全体EM @6000 | 全体EM @12000 | 長さ10 correct @6000→12000 | 全体95% @12000 |
|---|---:|---:|---|---|
| I01 | 0.6768 | 0.8662 | 6→88 / 206 | 未達 |
| I02 | 0.7070 | 0.9336 | 40→157 / 206 | 未達 |
| I03 | 0.5713 | 0.8037 | 10→26 / 206 | 未達 |
| I04 | 0.7373 | 0.9688 | 95→203 / 206 | 到達 |
| I05 | 0.9023 | 0.9766 | 153→201 / 206 | 到達 |

全体EMはS1の丸め表示、長さ10はS1の整数値。source replayと計算には実JSONの分子／分母とhashを使い、丸め値を正確な定数へ作り変えない。

S1は改善を「長さ10へ集中」と記載する。この記述を原文から削除しない。一方、本タスクでは **改善した正解数の長さ別内訳** と **終端に残る誤りの長さ別内訳** を別に計算する。例えばI03の長さ10の増加は16例であり、全体の増加と同じ量ではない。原文への補足が必要なら、新しい集計と出典を添えて追記する。

### 1.3 今回の新設条件と継承事項

| 項目 | 条件 |
|---|---|
| 監査の主対象 | 全P/I01～I05。I01～I03の長さ10を重点、I04／I05を対照 |
| 保存済み主観測点 | 8000、8500、9000、9500、10000、10500、11000、11500、12000 |
| 継続判定に使う点 | 実LR位相の一致を検証した8000／10000／12000 |
| 固定評価 | 既存 `rec004a_budget_validation` 1024例、同じsample順・hash |
| 継続条件 | §4の正解数・token誤り・lossに基づく規則。今回新設 |
| 継続する場合 | **全5本**、各12000→18000。最大30000新規optimizer updates |
| 新しい観測点 | 12500、13000、…、18000。主要到達判定は18000終端のみ |
| 変更可能 | 累計予算のみ。既定Pの本体＋既存192 bias parametersを継続学習 |
| 変更不可 | Core、構造、feature、LRレシピ、optimizer設定、loss、batch、dtype、データ分布 |
| 採用・RG3 | 結果にかかわらず未実施。独立queryも生成・評価しない |

18000、継続条件、同一終端への全5本継続は本タスクが新たに許可する範囲。旧12000上限を書き換えない。継続条件は追加計算を使う判断であり、旧RG3の性能floorを置き換えない。

---

## 2. 不変条件と変更範囲

### 2.1 監査区間

**新規optimizer updates=0。** 保存済みモデルをeval modeでforwardするだけ。optimizerは構築・更新しない。weight／buffer／modeの書換えや、oracle attention・bias倍率介入・学習可能probeは行わない。

評価に必要なモデル構築は、完全stateのstrict loadを前提に許可する。欠落重みのrandom fallback、`get_or_build`による暗黙学習、別Core／最新cacheへのfallbackは不可。

### 2.2 継続区間が許可された場合

各試行の隔離された `CrossPositionLengthBiasPrimitive` の既存parametersだけを更新する。新しいbranchやparameterは追加しない。optimizerのparameter ID集合と、元stateの名前・順序・shapeの対応を照合する。

不変対象は次のとおり。

- parent bundleの全16slot原本、Core／task側／decoder／vocabulary／schema。
- MIRROR_HALVES以外の15操作と、router／keys／ArgumentScorer／controller／verifier。
- REC-004Aの固定3候補、REC-004Cの初期states、REC-004D～Gの全source artifact。
- 全共有cache、構造signature、bias feature式、`length_ref=32`、既存192-parameter構成。

以前の「assembly時に4slotを変え、残り12を保護」という規則と混同しない。**今回assemblyは0。parent全16slotと固定3候補はread-only、学習するのは隔離Pだけ。**

Coreは `h_content=f(content)` を維持する。oracleは通常の訓練target生成と評価指標のみに使い、REC-004Fのoracle attention、正解位置lookup、hard mask、補助position lossを訓練へ入れない。

### 2.3 明示的に禁止する修正

長さ10へのoversampling、curriculum、hard-example mining、新length特徴、bias倍率／temperature変更、head別bias、容量増加、別LR、scheduler reset、optimizer reset、別初期化、Uの再学習、I04／I05の早期固定、成功するまでの延長は行わない。

本番generator／mask／indexing／loaderの契約違反が見つかれば、その場で修正して学習を続けない。最小再現例を保存し停止する。新しい監査コード自身の集計誤りは、source予測を変えず修正・再集計できるが、差分を記録する。

---

## 3. 段階A/B — Sourceを確認し、終盤の残差を読む

### A1. 正確なsourceを指定する

S1が報告した確認対象：

```text
runs/phase_b_b2_model_bundle_recovery/rec004g/run_001/
  config.yaml / system.json / source_replay.json
  learning_curve.jsonl / lr_trace.jsonl
  paired_extension_comparison.json / candidate_decision.json
  per_length_position_metrics.json / freeze_audit.json
  side_effect_audit.json / cost_accounting.json / summary.json
  I01..I05/P_LENGTH_POSITION_BIAS/
    checkpoints/step6500..12000.pt
    training_states/step6500..12000.pt

src/apc/evaluation/mirror_budget_extension.py
scripts/run_phase_b_b2_model_bundle_recovery.py
configs/phase_b_b2_model_bundle_recovery_rec004g.yaml
```

`step6500..12000`は500刻みの範囲表記であり、literal filenameではない。実index／metadataから完全pathを確定する。parent Core、旧validation、REC-004Dの6000-step状態へはS1とrun manifestの正確な参照を辿る。

P/I01～I05の12000-step full stateは必須。中間weightsが欠けていても同じ時点の完全stateから読込可能ならそれを使う。必要な8000／10000／12000の証拠を復元できない場合、継続判定は `EVIDENCE_INSUFFICIENT`。再学習で穴埋めしない。

source／recipe／parent不一致は `SOURCE_OR_RECIPE_BLOCKED`。Coreを新規構築して解決しない。

### A2. 12000-stepのsource replay

全5本について、canonical weight hash、architecture／Core依存、実validation hashを確認し、通常forwardでsource JSONの全体EM・長さ別正解数を再現する。許容差は元の再生契約から取得し、測定後に広げない。

初期化ごとのfull stateは、weights、AdamW状態、scheduler状態、CPU／CUDA RNG、累計step、必要なscaler／data位置または決定的generatorの仕様を検査する。optimizerのmomentum等を構築時の空stateで代用しない。

監査区間ではfull stateの読取検査のみ。実optimizerの生成・復元は§5の学習preflightに限定する。通常forwardの再生を、全GPU training trajectoryのbit-exact再現証明と混同しない。

### A3. 追加の曲線評価より先に監査条件を保存する

source replayが成立した時点で、観測checkpoint、data hash、metric定義、§4の固定predicateとloss幅、許容差を `late_progress_protocol.json` へ保存する。**B1の追加forwardに入る前にlockする。** 本文の節順を理由に、進行結果を見た後でpredicateを登録してはいけない。

### B1. 固定入力上で、分母を揃えた表を作る

全5本×主観測9点を計測する。sourceに予測・分子／分母が保存されていれば再利用し、不足するものは同じcheckpoint・同じ入力のforwardで補う。補足計測と元ログは区別する。

履歴6000-stepとの内訳比較はREC-004D/Gの保存値を優先する。必要なら6000を1点追加再生できる。**主評価forwardの上限は50 checkpoint-set×1024例＝51200 prediction examples**。source replayは可能な限り同じ結果を再利用し、重複実行は費用に明記する。新しい評価分布は作らない。

各init×step×長さについて保存するもの：

```text
n_examples / sequence_correct / sequence_EM
valid_token_count / token_error_count / token_accuracy
unreduced_token_CE_sum / mean_valid_token_CE
per_output_position: correct, errors, valid_count
first_error_position / errors_per_sequence
lr_used / lr_after_scheduler / scheduler_counter
```

lossは既存token lossの合法maskと同じ対象に基づく。集約lossの単位は「有効tokenあたり」と明示し、異なる長さのbatch平均を同じlossとして混ぜない。元ログのreductionが異なるなら新列として計算し、旧値を改竄しない。

今回の固定validationでは長さ10の分母はS1上206、全体は1024。実datasetと不一致なら新しい分母で黙って進めず、hash／schema／sourceを確認する。

### B2. 「どこが改善したか」と「どこに誤りが残るか」を分離

各initで6000→12000、および8000→10000→12000について、以下を計算する。

- 全体の正解数増加、各長さの正解数増加。
- 各長さの改善寄与 `delta_correct_length / delta_correct_all`。全体増加が0以下ならnull＋理由。負値や1超は他長の回帰であり、0～1へclampしない。
- 終端の誤り集中 `errors_length / errors_all`。全体誤り0ならnull＋分母0。
- 同一sampleのwrong→correct、correct→wrong、both_wrong、both_correct。
- 長さ6～9の合算と各長さの個別値を両方表示。「合算97%」を「各長さ97%」へ言い換えない。

整数値は実JSONまたは予測から得る。ADRの丸めEMから再構成した正解数をsourceの確定値として書かない。S1の「改善は長さ10に集中」という記述と相違する場合、原文を保持し、新しい表による適用範囲の補足を別ADRへ追記する。

### B3. 長さ10の位置誤りを、成功側と同じ表で比較

I03を中心に、I01／I02／I04／I05の同じ長さ×位置表も作る。I03の誤り例だけで評価集合を作り直さない。

評価専用の `pi_n(i)` は実sourceとの一致を確認し、出力←入力の向きで使用する。長さ10の固定位置は0始まりで2と7。fixed／moved／見かけ上のchanged位置を区別する。重複tokenの参照元は集合、推定不能はUNKNOWNとして残し、attentionの使用位置を出力tokenだけから断定しない。

8000／10000／12000で同じ出力位置・同じ誤答例に停滞するか、誤りが別位置へ移るかを記録する。I04／I05との違いは記述的比較であり、根本原因の証明とはしない。

新たなoracle-attention probe、bias-zero／倍率介入、observer再設計は行わない。今回は通常出力・既存loss・位置誤りだけで判断する。

### B4. LR位相を照合する

8000／10000／12000は、既定T_max=1000の同位相比較の候補。**step番号だけで同位相と決めず、実lr_used／lr_after_scheduler／counterで確認**する。

補助的に8500→10500、9000→11000、9500→11500も並べる。phaseごとの山谷があっても、それだけでscheduler不良とはしない。評価点を都合のよい位相へ置換しない。

位相・データ・計測単位が不一致のまま、§4の数値規則を実行してはいけない。必要証拠不足なら追加学習は行わない。

---

## 4. 段階C — 条件付き延長を許可する機械的規則

### C1. 判定規則の性質

以下は**今回新設する計算予算の許可条件**。旧資料が定めていた性能Gateでも、学習が今後成功するという保証でもない。条件不成立は「収束が証明された」「容量限界」ではなく、**この一件では追加計算を許可するだけの終盤進行が確認できなかった**と記載する。

本規則とloss用許容幅は、Hの追加forward結果を見る前に `late_progress_protocol.json` へ固定する。元sourceの監査・既に提示された12000終端値の参照は可能。結果を見てpredicate、対象init、比較点を変更しない。

### C2. 必須の入力

各initについて、長さ10・同じ206例の8000／10000／12000時点の値を用いる。

- `C_t`：sequence正解数。整数。
- `E_t`：有効tokenの誤り数。整数、分母は同じ2060。
- `L_t`：有効tokenあたりCE。finiteな実数、同じ計算契約。
- `loss_drop(a,b)`：`L_b < L_a - 1e-6`。`1e-6`は今回の数値比較用固定幅であり、有意水準ではない。

全体EMは継続対象の判定にのみ用いる。`below_floor`は `100 * sequence_correct_all < 95 * n_examples_all` と整数で計算する。S1上の未達集合はI01／I02／I03であり、これを再生で照合する。I04／I05も監査から外さない。

### C3. 初期化ごとの進行フラグ

```text
EM_PROGRESS =
    (C_12000 > C_10000) AND (C_12000 > C_8000)
    AND (
        (E_12000 < E_10000)
        OR loss_drop(10000, 12000)
    )

SOFT_PROGRESS =
    (C_12000 >= C_10000) AND (C_12000 >= C_8000)
    AND (E_12000 < E_10000) AND (E_12000 < E_8000)
    AND loss_drop(10000, 12000) AND loss_drop(8000, 12000)

INIT_PROGRESS_CONFIRMED = EM_PROGRESS OR SOFT_PROGRESS
```

`SOFT_PROGRESS`は、系列EMが離散値で横ばいでも、誤token数とlossが改善する場合を扱う。過去6000→12000の大きな全体増加だけでは、どちらのフラグも立たない。

必要値欠落／非finite／分母不一致はfalseへ押し込まず、`EVIDENCE_INSUFFICIENT`または該当契約エラーとする。

### C4. 全体のbranch

```text
if source / data / phase / freeze contract fails:
    decision = BLOCKED
elif required evidence is missing:
    decision = EVIDENCE_INSUFFICIENT
elif no init is below the overall floor:
    decision = ALREADY_AT_FLOOR_AUDIT_ONLY
elif every below-floor init has INIT_PROGRESS_CONFIRMED:
    decision = EXTEND_ALL_FIVE_TO_18000
else:
    decision = STOP_NO_EXTENSION
```

今回の重要点は、**I03が進行条件を満たさなければ、I01／I02だけを延長しない**ことである。弱い一試行を置き去りにして全5本安定性の問いを変えない。I04／I05の成績を未達3本の代わりに使わない。

`STOP_NO_EXTENSION`の場合、predicateのどこが不成立だったかと位置誤りを示す。結果が混在なら `MIXED_OR_NO_CONFIRMED_LATE_PROGRESS` とし、plateau／capacity／downstream故障を強制分類しない。ここで学習を終了したことにして実は別修正を試すことは禁止。

---

## 5. EXTEND_ALL_FIVE_TO_18000の場合のみ行うこと

### 5.1 追加学習前にresume契約を検証

REC-004Gの実装とfull stateを使う。対象は**REC-004Gの12000-step**であり、REC-004Dの6000-stepや成績のよい別checkpointへ戻らない。

optimizer／schedulerは現repoで検証済みの構築・load順序を再利用し、初期化時にLRを書き換える副作用がないことをCPUの連続実行／分割再開試験で検査する。S1のload順に関する一般化を、未検証の別schedulerへ拡張しない。

次を実値で確認してから学習する。

```text
cumulative_optimizer_updates = 12000
source = G/<same init>/P_LENGTH_POSITION_BIAS/training_states/step12000.pt
next training sample-step = 12001
scheduler = sourceと同じCosineAnnealingLR、T_max=1000
optimizer / parameter ordering / state tensor shapes = sourceと一致
Core / architecture / bias feature / length_ref = sourceと一致
```

共通recipeはAdamW、既定max LR=0.0008、eta_min=0.00001、T_max=1000。残りの引数、batch、dtype等は実config／stateから継承する。ライブラリや精度の変更で解決しない。model seed10はCoreの識別子、I01～I05はそのCore上の初期化であり、別Coreへ変換しない。

full stateが足りなければ `RESUME_STATE_INCOMPLETE`。重みだけからoptimizerを再作成した近似再開やfresh再学習は禁止する。

### 5.2 新しい有限予算

各initを同じ12000-step状態から、**追加6000 updates、累計18000**まで実行する。成功しているI04／I05も同じ終端へ進める。

- 5本合計で最大30000新規updates。各sourceが持つ既往12000を累積費用へ残す。
- 成績が良くても悪くても、通常は全5本を18000まで進める。
- 途中95%到達で止めない。12000の成功checkpointを最終値へ混ぜない。
- 元の純粋step generatorの12001～18000を用い、1000／6000／12000へstepを巻き戻さない。
- 同じstepでは全5本に同じbatch・target・sample順を与え、digestで確認する。
- オンライン生成なら追加更新と追加例への露出が同時に増えると記録する。更新回数だけの純粋な効果と呼ばない。

未来の18000超のbudget、別LR、追加init、length10だけの選択的訓練は、この一件では許可しない。

### 5.3 計測・保存

各updateの使用LR／scheduler更新後LR、loss、sample digest、optimizer更新数・skipを記録する。12500／13000／…／18000で、§3の全長・位置別指標、weights、完全training stateを保存する。

同じ既存validation1024例を使う。`train_fit`はGが使用した実露出済み集合と出自が確認できる場合だけ継承し、取得不能なら `TRAIN_FIT_UNAVAILABLE`。新しい同分布例をtraining accuracyと呼ばない。

観測用evalがtraining RNG／mode／bufferを変えないようにする。装置中断からは完全state・同じprotocolの厳密resumeだけ許可し、上限をリセットしない。数値発散は該当trialの欠測として保存し、隔離が保たれる場合だけ残りを継続する。発散trialの交換は禁止。source／stream／freeze違反では全依存学習を停止する。

### 5.4 データ役割と露出

既存validationは既に適応的に分析したdevelopment資料であり、独立holdoutではない。新しいRG3 query・reference・sealed dataは生成・評価しない。

今回追加する訓練streamと既存validationの重複を、sample IDに加え `(task, arguments, content)` digestで、モデル出力を見ず監査する。重複があれば、訓練streamやvalidationをその場で都合よく変更せず、汚染範囲と比較上の制約を記録し、追加学習を停止する。履歴Core等の不明な露出はUNKNOWNのまま残す。

継続区間を始める前に、`extension_protocol.json`へsource full-state hash、recipe、step range、data-generator version、入力列digest、全5本、観測点、終端判定、費用上限を固定する。学習前のdata生成・hash検査は許可するが、独立queryを使ってはいけない。

---

## 6. 段階D — 結果をまとめ、次の一件へ引き継ぐ

### 6.1 延長した場合

全5本について、6000／12000／18000の全体EMと各長さの正解数、12000→18000のpaired増減、長さ10の残差、他長の回帰を示す。主要到達判定は**18000終端だけ**とする。

- 全体EM>=0.95の本数／予定5本、完了本数、diverged／missingを別々に記録する。
- mean／sample SD／min／max、全例正誤と長さ別の分母を表示する。
- 長さ10の失敗を全体EMだけで隠さない。一方、各長さ95%という新しい採用Gateは追加しない。
- 12000で合格していたI04／I05が18000で下がった場合もそのまま失敗として記録する。古い成功値へ戻さない。
- Pだけの継続比較なので、Uに対する同予算優位性、他Coreへの一般化、自律的architecture discoveryは主張しない。

`ALL_FIVE_AT_18000_FLOOR`となっても、これは候補採用前のdevelopment到達確認に限る。**I01を含め、どのinitも本タスクでは選択しない。** 全16操作の独立query／fresh-process確認を次に実施すべき条件が揃ったと報告できるだけである。

### 6.2 延長しなかった場合

`late_progress_decision.json`に初期化ごとのC／E／Lの実値、各predicate、missing、位置誤りを残す。`new_optimizer_updates=0`を監査する。

次の提案は最大一件。たとえば「I03の長さ10の特定位置で、同位相でも誤りが残る」という観測があれば、その位置対応へ限定した次の確認・修正契約を提案できる。ただし、本タスク内で新loss・feature・mask・oracle probeを実装しない。証拠不足なら `EVIDENCE_INSUFFICIENT` とし、原因や修正案を創作しない。

### 6.3 状態を分離して出す

```text
implementation_status: COMPLETE / PARTIAL / BLOCKED
source_replay_status: VERIFIED / BLOCKED
late_audit_status: COMPLETE / PARTIAL / BLOCKED
continuation_decision:
  EXTEND_ALL_FIVE_TO_18000 / STOP_NO_EXTENSION /
  EVIDENCE_INSUFFICIENT / BLOCKED / ALREADY_AT_FLOOR_AUDIT_ONLY
extension_status:
  COMPLETE / COMPLETE_WITH_NUMERICAL_FAILURES / PARTIAL /
  NOT_AUTHORIZED_BY_PROGRESS_RULE / NOT_EXECUTED
new_optimizer_updates: <実数、0～30000>
terminal_floor_status:
  ALL_FIVE_AT_18000_FLOOR / TARGET_NOT_MET_AT_18000 /
  INCOMPLETE / NOT_EVALUATED
selected_init: null
selected_intervention: null
child_bundle: null
rg3_recheck: NOT_EXECUTED
rec005_eligible: false
rec005_executed: false
```

読み取り診断の完了、追加学習の実施、18000終端の到達、RG3再判定を同じPASSにまとめない。旧Gの12000-step未達・旧Dの0/5・旧RG3 FAILは保存する。

---

## 7. 配置・テスト・成果物

### 7.1 既存実装を最小限再利用する

```text
docs/CODEX_TASKS_PHASE_B_B2_MIRROR_LATE_PROGRESS_CONDITIONAL_EXTENSION.md
src/apc/evaluation/mirror_budget_extension.py              # 報告上存在、resumeロジック再利用
src/apc/evaluation/mirror_late_progress_conditional_extension.py # 必要なら新規
scripts/run_phase_b_b2_model_bundle_recovery.py            # REC-004Hの明示dispatchのみ
configs/phase_b_b2_model_bundle_recovery_rec004h.yaml
tests/test_mirror_late_progress_conditional_extension.py
runs/phase_b_b2_model_bundle_recovery/rec004h/<run_id>/
```

新規名は提案。既存abstractionが同じ契約を満たせば重複作成しない。Gの6000→12000動作や原defaultを変更せず、H専用entry point／configで範囲を指定する。

実装後の推奨CLI形：

```bash
python scripts/run_phase_b_b2_model_bundle_recovery.py --task B-C005REC-004H --config configs/phase_b_b2_model_bundle_recovery_rec004h.yaml
```

実CLIが異なればrepoに従い、完了報告へ実commandを記載する。`--all`、latest探索、次task自動dispatchを追加しない。新run namespaceを使い、過去run_001を上書きしない。新ADR番号は実索引で確認し、0103等を推測で予約しない。

### 7.2 必須テスト

1. source hash／step／Core不一致を訓練前に拒否し、丸めEMから誤った厳密値を作らない。
2. 改善寄与と残差集中を区別し、分母0・負delta・1超・missingを正しく扱う。
3. per-length／token lossの分母・mask、固定位置n=10:{2,7}、重複tokenの曖昧さを検査する。
4. `EM_PROGRESS`、`SOFT_PROGRESS`、全体EMだけ改善、初期8kだけ改善、最終lossだけ改善、欠測の各fixtureを検査する。
5. I01／I02だけ進行しI03が未達なら、全5本の新規学習が0であることをdynamic spyで確認する。
6. 同位相不成立、必要値欠落、recipe不一致が継続条件を通過しないこと。
7. full stateの連続実行と分割再開をCPU tiny fixtureで検査し、weight-only resumeを拒否する。
8. sample-stepが12001から始まり、全5本が同じstreamを使い、schedulerをresetしないこと。
9. 監査／eval内のoptimizer・builder呼出しを動的に拒否し、条件成立後の明示training区間だけ許可すること。
10. optimizerの対象は隔離MIRROR本体＋既存biasのみ。Core／親全16slot／固定3候補の不変性。
11. 合格済みinitも共通終端へ進み、途中最高値採用・18000超・追加initが起きないこと。
12. NaN／skip／中断／resume／未実行のstep・分母・費用が正しく記録されること。
13. 全5本が18000で合格する人工fixtureでもcandidate／child／RG3／REC-005が呼ばれないこと。
14. 原artifact上書き・共有cache更新・sealed access・evaluationデータによるsampling変更を拒否すること。

実験の3分程度といった旧wall-clock値を今回の見積りや実測へ転記しない。CPUのtiny結果と本番5本の学習結果を区別する。

focused tests後にrepo標準検証を実行する。

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

現repoのcanonical commandが異なる場合はそちらを使う。過去2197件PASSを今回の試験結果へ転記しない。他ジョブを無断で停止せず、未実施・resource不足は明記する。

### 7.3 必須成果物

```text
config.yaml / protocol.json / system.json / summary.json / report.md
source_manifest.json / source_replay.json / resume_state_audit.json
late_progress_protocol.json / late_checkpoint_index.json
late_length_metrics.jsonl / position_error_trajectory.json
improvement_vs_residual_decomposition.json
same_phase_audit.json / late_progress_decision.json
freeze_audit.json / side_effect_audit.json / cost_accounting.json
# 条件成立時だけ：
extension_protocol.json / extension_data_manifest.json
I01..I05/P_LENGTH_POSITION_BIAS/{checkpoints,training_states}/step12500..18000.pt
learning_curve.jsonl / lr_trace.jsonl / data_trace.jsonl
paired_extension_comparison.json / per_length_position_metrics.json
terminal_floor_status.json
# 常に作成可、提案のみ：
next_step_recommendation.md
```

未到達段階のファイルは捏造せず、summaryへNOT_EXECUTEDと理由を書く。次の修正提案を生成しても、それ自体を実行許可と扱わない。

### 7.4 完了報告の形式

```text
Task: B-C005REC-004H
Source / Core / P12000のhash・再生結果:
Init | overall@12000 | len10 C@8000/10000/12000 | E | L | progress判定
実位相の照合 / 改善内訳と残差集中 / I03の位置誤り / 未確定事項:
continuation_decision / その根拠:
（延長時）Init | EM@12000/18000 | 全長別値 | 95%到達 | regression | updates
新規／累計費用 / 欠測 / 完全resume / stream一致 / freeze:
実tests・command / artifact / 新ADR / 旧結果の保持:
selected_init=null / child_bundle=null / RG3 NOT_EXECUTED:
REC-005、R3-011/012、B-C006、Task Inference: blocked
次の一件の提案（未実行）:
```

**STOP：この一件の完了後は結果を報告する。成績にかかわらず、候補採用・RG3・REC-005・別の修正・更なる予算追加を自動実行しない。**
