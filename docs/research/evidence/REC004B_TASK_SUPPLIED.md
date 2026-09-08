# AI Coding Task — MIRROR_HALVES Schedule Comparison & RG3 Recheck

**正式ID：B-C005REC-004B**  
**版：1.0 / 2026-09-08 / 状態：新しい追加指示。実装・実験は未実施。**  
**位置：REC-004Aのvalidation未達後、REC-005の前。実行対象は本タスク一件だけ。**

## 0. 目的・権限・開始条件

MIRROR_HALVESについて、**同じ初期重み・同じ訓練入力・同じ6000 optimizer updatesで、学習率スケジュールだけが違う2条件**を比較する。Coreやoperator構造を変更せず、現在のcompact operatorが回復基準へ到達できるかを検査する。

CYCLE_FOUR／ROTATE_TRIPLETS／SWAP_ENDSは、REC-004Aの保存済み4000-step checkpointを固定候補として再利用する。これらを再学習せず、MIRROR_HALVESの候補が揃った場合だけchild bundleを作り、独立queryでRG3を再判定する。

ユーザーが本タスクを指定した場合、以下のA〜Eを、前提とSTOP条件に従って一件の中で実行できる。各段階で改めて承認を求める必要はない。**REC-005、R3-011／012、B-C006、Task Inferenceや別の修正タスクは自動実行しない。**

```text
REC-004 / ADR-0095：RG3 FAIL                         ← 保存
REC-004A / ADR-0096：VALIDATION_TARGET_NOT_MET
                    RG3再判定はNOT_EXECUTED          ← 保存
    ↓
REC-004B
 A. artifact・実レシピ・LR／データ契約の確認
 B. 2条件・固定3候補・選択規則・queryを事前登録
 C. MIRROR_HALVESのみ、A/B各6000更新を実行
 D. 最終checkpointから規則どおり選びchildを準備
 E. 全16操作の独立query・fresh-processでRG3再判定
    ↓ 全条件PASSの場合だけ
REC-005へ改訂レシピを引き継げる。実行は別の明示指示。
```

### 0.1 読む資料とその役割

| 区分 | 資料・役割 |
|---|---|
| S1：実行担当者の報告 | [ADR-0096の原文](research/evidence/REC004A_ADR0096_SUPPLIED.md)。checkpointや数値は報告事項で、現repoで実在を照合する |
| S2：旧タスク契約 | 現repoの`docs/CODEX_TASKS_PHASE_B_B2_INCREMENTAL_BUDGET_CALIBRATION.md`。配布時snapshotは[こちら](research/evidence/REC004A_TASK_SUPPLIED.md) |
| S3：復旧受入条件 | 現repoの`docs/EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`。配布時snapshotは[こちら](research/evidence/RECOVERY_ACCEPTANCE_SUPPLIED.md) |
| 実装時の依存契約 | 現repoのModelBundle契約、復旧task／plan／agent addendum、REC-003のprotocol、REC-004のmanifest、REC-004Aの実config／JSON |
| W1：外部API確認 | PyTorchのCosineAnnealingLR公式仕様。適用範囲とURLは[Source notes](research/B2_MIRROR_SCHEDULE_SOURCE_NOTES.md) |

この文書は、**明示した新設条件だけ**S2の変更禁止／選択規則を改訂する。元文書・ADR・runは上書きしない。実API名、physical ID、file hash、optimizerの未確認値を捏造しない。

### 0.2 報告値・解釈・新設条件を混同しない

S1はMIRROR_HALVESについてvalidation EM=0.2725→0.6875（4000→6000）、6000時点のtrain-fit=0.7061、長さ6で196/204、長さ10で46/206と報告している。またT_max=1000を6000まで進め、2000／4000／6000時点の記録LRが0.0008へ戻ったとしている。[S1: Design、Evidence 1–2]

S1には「length-dependent capacity limit」という帰属もある。**本タスクはその帰属を再実証済みの事実として採用しない。** 長さ依存の失敗は観測だが、容量・最適化・位置対応の寄与は未分離として、scheduleを一つの仮説として比較する。S1の原文は保持し、新しい解釈は追記ADRにする。

本タスクで新設するものは、T_maxのA/B比較、6000-step終端選択、500-step計測、保存済み3操作の4000-step採用、候補選択規則、REC-004B用の独立queryである。これらは過去実績ではない。非SHIFT15操作の各EM>=0.95、query各1024例、SHIFTだけの既定例外、freeze／履歴保存／loader規則はS2・S3から継承する。

---

## 1. 許可される変更の正確な範囲

| 区間 | 変更可能 | 必ず不変 |
|---|---|---|
| A/B学習 | 隔離したMIRROR_HALVES primitiveの重み。B条件だけT_maxを6000へ変更 | Core／task側／decoder、他15操作、router全key、ArgumentScorer、schema、formula、controller、verifier |
| 3候補固定 | REC-004Aの3つの4000-step重みをread-only import | その重み、学習レシピ・来歴の原本。追加学習やcheckpointの選び直しをしない |
| 最終assembly | MIRROR_HALVES＋固定3操作の計4 physical slotを新versionへ置換 | 元parentの残り12操作と、全ての共通component・実行formula |
| 計測・文書 | LR／RNG／checkpointログ、focused tests、薄いdispatcher追加、ADR／navigation追記 | 過去run・bundle・共有cacheの原本 |

**今回新規に学習する操作はMIRROR_HALVES一つ。最終bankで親から変わるslotは最大4つ。** この二つをfreeze監査で混同しない。physical primitiveの総数は16、ID mappingとoperator型は変更しない。

保護対象12操作は、SELECT、COUNT、BIND、COPY、NEGATE、SWAP_PAIRS、INVERT_HALF、REVERSE、SORT、ALTERNATING_NEGATE、INCREMENT_MOD、SHIFT。実physical IDとparameter prefixはmanifestから取得する。

Coreは`h_content=f(content)`を維持する。buffer・mode・formula・dtypeも監査する。単に`requires_grad=False`と設定したことを、不変性の証拠にしない。

**禁止：** Core再学習、operator大型化／新位置bias／oracle gatherの導入、LR上限・下限変更、別optimizer／loss／batch／精度への変更、warmup追加、curriculum、データ再重み付け、別初期化の探索、途中checkpointの最良値選択、6000超への延長、router再較正、verifier調整、未達操作の除外、SHIFT以外への例外追加。

---

## 2. 段階A — 実artifact・レシピ・契約を確認する

### A1. 親と既存候補をread-onlyで確定

資料が報告した確認対象は次のとおり。実ファイルが存在しhash／Core依存が一致することを確認する。latest探索や別seedへのfallbackは禁止。

```text
# parent / REC-004
runs/phase_b_b2_model_bundle_recovery/bundles/
  2356543740ce566640f72767e73bd83955bc27bb825bb06c8fcffab03cf53995/manifest.json

# source / REC-004A
runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/
  budget_protocol.json
  operation_contract_audit.json
  learning_curve.jsonl
  error_breakdown.json
  selected_recipe.json
  qualification_scope_audit.json
  checkpoints/
```

上記manifestの改行は表示用。configには実際の完全pathを保存する。REC-004Aはchildを作っていないため、REC-004Aの「完成bundle」を探したり存在を仮定したりしない。

次の3候補を**各4000 updatesのsnapshot**として固定する。ファイル名はcheckpoint index／metadataから特定し、命名規則を推測しない。

| 固定候補 | 採用step | S1のvalidation EM | 今回の扱い |
|---|---:|---:|---|
| CYCLE_FOUR | 4000 | 0.9912 | 重みを再利用。旧1000-step選択を履歴から消さない |
| ROTATE_TRIPLETS | 4000 | 0.9980 | 重みを再利用。旧2000-step選択を履歴から消さない |
| SWAP_ENDS | 4000 | 0.9971 | 重みを再利用。旧2000-step選択を履歴から消さない |

元の最小step候補よりvalidation上の余裕がある保存済み候補へ切り替える**新しい選択判断**であり、独立query合格を主張しない。4000が最適だとも主張しない。3候補のCore／decoder／schema一致と実学習receiptを検査する。weight-only保存でも、再学習しないimport用途ならoptimizer状態は不要。

元の固定validationを再生成できる場合は、3候補のsource予測／指標を一度照合する。これは出自確認であり候補探索ではない。合わなければ理由を報告し、別stepへ切り替えない。source metrics／datasetが不足して出自・互換性を確認できないなら`SOURCE_CANDIDATE_UNVERIFIABLE`で停止する。

### A2. 実レシピ・操作定義を読む

既存`incremental_budget_calibration.py`、`_train_single_primitive`相当、MIRROR_HALVESのfactory、generator、loss、評価関数を確認する。既存の名前・呼び出し先が変わっていれば現コードに合わせ、同等のhelperを重複実装しない。

optimizerの全引数、LR、eta_min、scheduler呼出し順、batch、dtype、gradient accumulation、dropout、train/eval mode、data seed導出をREC-004Aの実configと照合する。報告値は`lr=0.0008`、`eta_min=0.00001`、AdamW、T_max=1000。異なる場合は、新旧差を残し`RECIPE_MISMATCH`で止める。本文にないweight decay等は実値を継承し、一般的なdefaultで補わない。

MIRROR_HALVESの合法length、半分・端数・mask・padding、出力順はsourceと小さな合法fixtureで確認する。名前から変換規則を決めない。教師と評価の契約違反が見つかれば、schedule比較と同時修正せず停止する。

### A3. 初期化とqualification

MIRROR_HALVESは**両条件とも新しい共通初期stateから学習する**。旧6000-step重みのfine-tuneやweight-only resumeは本タスクでは行わない。旧runの初期stateが記録されていなくても作業可能だが、旧runとの完全再現とは呼ばない。

parent／候補を明示diagnostic scopeでロードし、未達parentに「全非SHIFT15操作がfloor合格」という資格を要求すると拒否または未認証になることを確認する。新childのqualificationは別途作る。scopeが未確定なら診断比較だけは可能だが、RG3 PASS／通常capability公開は不可。既存loaderが不十分なscopeを通す契約違反は別タスクにし、ここで緩和・修正しない。

出力：`source_audit.json`（parent／3候補hash・実レシピ・操作仕様・qualificationの項目を含む）。

---

## 3. 段階B — 比較protocolを固定する

### B1. 2条件と予算

| 項目 | A：対照 | B：schedule修正候補 |
|---|---|---|
| condition_id | `A_FIXED_TMAX_1000` | `B_SINGLE_DECAY_6000` |
| scheduler | CosineAnnealingLR | CosineAnnealingLR |
| T_max | 1000のまま6000まで進める | 6000 |
| 最大／最小学習率 | 0.0008 / 0.00001 | 同左 |
| 新規updates | 6000 | 6000 |
| 初期state／訓練入力 | 同じ保存stateとstream | 同左 |
| optimizer／loss／構造 | REC-004A実レシピ固定 | 同左 |
| 主要判定checkpoint | **6000のみ** | **6000のみ** |

model seedは10。一つの事前固定初期化をA/Bで共有する。**新規学習は最大12000 updates、MIRROR_HALVES 2本だけ。** 旧3候補の既往4000 updates等は来歴・累積費用へ残すが、本タスクの新規学習と混同しない。

これは同一計算予算のschedule比較であり、LRの積分量まで同じにする実験ではない。Bでは最初の1000 updatesのLR列も変わる。S2の「最初の1000を同一に保つ」制約は、本タスクのB条件に限り明示的に解除する。その他の旧制約を解除しない。

### B2. LRの時刻定義を固定する

現repoのPyTorch versionを保持する。ライブラリをupgradeして解決しない。公式APIの説明はW1、実際のLR traceはインストール済みversionのCPU試験で検査する。

`u`を完了したoptimizer update数とする。schedulerはoptimizer update後に一回呼ぶ。外部からLRを変更しない条件の検査用式は、

```text
eta_T(u) = 0.00001 + (0.0008 - 0.00001)/2 * (1 + cos(pi*u/T))
```

とする。[W1] 次の表は**u回更新後、schedulerを進めた後の「次回用LR」**であり、直前のupdateで使用したLRではない。

| u | A：T=1000 | B：T=6000 |
|---:|---:|---:|
| 0 | 0.0008000000 | 0.0008000000 |
| 1000 | 0.0000100000 | 0.0007470800 |
| 2000 | 0.0008000000 | 0.0006025000 |
| 3000 | 0.0000100000 | 0.0004050000 |
| 4000 | 0.0008000000 | 0.0002075000 |
| 5000 | 0.0000100000 | 0.0000629200 |
| 6000 | 0.0008000000 | 0.0000100000 |

update uの実使用値は、上記順序なら`eta_T(u-1)`に対応する。**Bの6000回目updateが厳密にeta_minを使った、と誤記しない。** `lr_used`、`lr_after_scheduler`、scheduler内部counterを別々に保存する。

Aは内部counterやoptimizerをリセットしない。CosineAnnealingWarmRestartsへ置換しない。再上昇とoptimizer restartは別概念。6000で両条件を停止する。

この呼び出し順とREC-004Aが食い違う、またはpreflight traceが式と一致しない場合、誤差許容は測定前に固定した小さな数値許容だけを使い、`LR_TRACE_CONTRACT_FAILURE`で止める。複数の変更を黙ってAへ入れて「同レシピ対照」と呼ばない。

### B3. 共通初期化・訓練乱数

MIRROR_HALVESの初期weights／buffersを一度だけ作り、保存してA/Bに完全loadする。factoryがbank登録や他重み初期化を行うなら副作用を隔離する。**seedラベルの一致だけで同一初期化としない。**

A/Bはoptimizerの新しい空状態から開始する。optimizer hyperparametersは同じ、scheduler stateのT_maxだけが意図的に異なる。condition開始時のCPU／CUDA RNG、データ位置、dropout等に使用するtraining乱数を同じ状態へ戻す。eval用乱数は独立にし、途中evalでtraining streamが変わらないようにする。

各updateで同じsample ID・content・target・batch順となることをdigestで確認する。data seedにA/B条件名を混ぜない。実際のaugmentation／shuffleがあればその乱数も一致させる。学習順A→BとB→Aの契約はCPU tiny fixtureで確認し、本番で順番探索をしない。

### B4. データの役割

| role | 内容 | 候補選択 |
|---|---|---|
| `train` | REC-004Aの生成レシピと同じ分布。6000-step入力列をA/Bで共有 | 勾配計算のみ |
| `train_fit` | 両条件で実際に露出済みの同じ固定例。最大1024例 | 診断のみ |
| `schedule_validation` | MIRROR_HALVES各1024例。REC-004Aの固定budget_validationを再利用しhash照合 | **終端6000のA/B選択にのみ使用** |
| `source_validation_replay` | 固定3候補の旧validation | 出自確認のみ。別step選択に使用しない |
| `rec004b_recheck_query` | **新namespaceで全16操作各1024例**。親・選択済みchildに同じ入力 | 使用禁止。候補固定後に一回評価 |
| `reference` | nominal認証を要求する場合の既定契約 | 本タスクのschedule選択に使用しない |

旧validationは既に適応的に利用されているdevelopment資料であり、新たなholdoutとは呼ばない。新しい最終queryの分布・生成version・seed導出・sample数・hashを学習前に固定し、モデル出力は段階Eまで計算しない。旧REC-004Aのrecheck_queryには「重複監査で言及」と「未生成」の両記述があるため、その未露出を推測して流用せず新namespaceを使う。[S1: Design、Evidence 4]

A/B間は同じ訓練例を意図的に共有する。一方、trainとvalidation／最終query、旧3候補の訓練streamと最終queryの重複はsample IDに加え`(task, arguments, content)` digestで確認する。重複除外はモデル出力を見ない決定的手順で最終query側に適用し、A/Bの訓練streamを途中で変えない。履歴Core等の全露出を確認できない範囲はUNKNOWNと開示する。

現repoのdevelopment／validation／sealed policyを継承し、旧sealedや新sealedデータを読込・学習・選択・再評価しない。data seedとmodel seedを混同しない。

### B5. ロックしてから実行

`protocol.json`へparent hash、固定3候補hash、初期state hash、全レシピ、A/BのLR列、データrole／hash、選択規則、最大updates、判定対象、許容差、software／precisionを保存する。run開始後に条件を変えた場合は同じ実験として継続せず停止する。

本タスクの2つの改訂――MIRROR_HALVESのschedule比較と、旧3操作の4000-step固定――を実行前の新ADR／protocol revisionに記載する。既存ADR番号は実repoで次の未使用番号を確認し、0096以降を推測して固定しない。

---

## 4. 段階C — MIRROR_HALVESの2本を実行・計測

### C1. 保存と計測

各条件で0、500、1000、…、6000 updatesを記録する。step0は学習前の対照、主比較は6000のみ。途中の高性能checkpointを選ばない。

- 毎update：累計updates、使用sample/batch digest、loss、`lr_used`／`lr_after_scheduler`、NaN／skip、実parameter更新対象。
- 500ごと：同じtrain-fit／validationのEM、token accuracy、loss、length別の分子／分母、first-error位置、実系列長とpadding位置。
- MIRROR_HALVESの合法な位置対応をsourceから評価専用に取得できる場合、構造上の移動位置と見かけ上のtoken変化位置を分けて集計する。正解対応をprimitiveのforwardへ入れない。
- Correct／None／Wrong-familyなど既存の適用可能なcausal controlsを最低限6000で測る。等価な出力が生じる例へ一律のgapを要求しない。
- Core／他15primitive／router／scorer等のhash、実更新parameter ID集合、wall time、peak VRAM、学習例数／重複を記録する。

各観測点でweightsに加え、optimizer、scheduler、AMP scaler（使用時）、CPU／CUDA RNG、data cursor／sample位置、累計updates、protocol hashを保存する。モデルだけを保存して厳密なresumeと呼ばない。保存と再開は既定の安全なcheckpoint機構を利用する。[W2]

NaN、AMPのupdate skip、対象外更新、入力列の不一致等でpaired contractが崩れた場合は、その事実と実行済み更新数を記録し停止する。勝手なdtype変更・別初期化再試行はしない。装置の中断から再開する場合だけ、**完全state・同じprotocolの厳密resume**を許可する。再開で上限をリセットしない。

### C2. 因果帰属と成功を分ける

A/Bを同じ1024例で比較し、終端EM差（B−A）、Aのみ正解／Bのみ正解の例数、length別差、train-fitの差を出す。token平均への置換や、平均長だけの比較をしない。

| 終端6000の結果 | 記載できる解釈 |
|---|---|
| A未達・B合格 | 同じ初期化・データ・更新数のこのpilotではschedule修正の有効性が支持される |
| A/Bとも合格 | 現構造で到達例を得た。Bが不可欠、または一般に優れるとは結論づけない |
| A合格・B未達 | B案はこのpilotで支持されない。Aの成功は旧runの誤りの証明ではない |
| A/Bとも未達 | この2条件では解決しなかった。容量限界の証明でも、追加stepが無効という証明でもない |
| pairの前提が不成立 | 数値が高くてもschedule効果への帰属は不可。paired実験として停止 |

今回のAは**旧scheduleレシピの新しい共通初期化での対照**であり、ADR-0096のMIRROR_HALVES=0.6875と同じ重み・曲線の再現を要求しない。旧数値との比較は履歴列に置く。

長い入力だけの未達、lossがまだ減少中、train-fit／validation双方の停滞などを記録し、どれかを結果前に「容量不足」と決めない。one seed・one initializationであり、5-model一般化や長さ外挿の証拠ではない。

---

## 5. 段階D — 固定規則で候補を選び、childを準備

### D1. 選択規則（事前固定）

両条件の6000 updatesが正常に完了し、paired／freeze／source検査が成立することを前提とする。

```text
if validation_EM(B at 6000) >= 0.95:
    selected = B at 6000
elif validation_EM(A at 6000) >= 0.95:
    selected = A at 6000
else:
    selected = null
```

**Bをprimaryとし、B未達・A合格のときだけAを事前登録fallbackとして選ぶ。** 両方合格ならBを選ぶが、それだけでschedule優位を主張しない。Aを選んだ場合は`CONTROL_SELECTED_REPAIR_NOT_SUPPORTED`と記録する。数値の高い方を無条件に選ぶ規則ではない。

selected=nullなら`VALIDATION_TARGET_NOT_MET`。途中checkpoint、旧REC-004Aの6000-step、別T_max、別初期化へ切り替えず、child／最終query／fresh-process再判定は`NOT_EXECUTED`として停止する。

固定3候補は4000-stepのまま。最終query前でも、6000-stepへ引き上げたり再学習して余裕を増やしたりしない。

### D2. Recipe revisionとchildの内容

`selected_recipe.json`を新runへ作り、以下のoperation別レシピとartifactの完全hashを固定する。

| Operation | 採用updates／source | schedule |
|---|---|---|
| MIRROR_HALVES | 本タスクの選択された6000-step snapshot | 選択AのT_max=1000、またはBのT_max=6000 |
| CYCLE_FOUR | REC-004Aの4000-step snapshot | **元のT_max=1000**。T_max=4000へ変更しない |
| ROTATE_TRIPLETS | REC-004Aの4000-step snapshot | 同上 |
| SWAP_ENDS | REC-004Aの4000-step snapshot | 同上 |
| 他12操作 | REC-004 parentの同じartifact | 変更なし |

旧REC-004Aの`selected_recipe.json`を上書きしない。固定3候補への切替は過去validationから選んだ新しい判断として開示し、LR比較のcausal deltaと混同しない。純粋なLR効果の比較対象はMIRROR_HALVES A/Bだけである。

新規の隔離childへ4slotをstrictに読み込み、version／training receipt／bank hash／execution signature／依存参照を更新する。関係するcache・機能certificateは無効化し、旧certificateを新しいbankへ無条件copyしない。コピー済み3候補は`IMPORTED_TRAINED_CHECKPOINT`等、現schemaに合う来歴で記録する。

他12操作・Core・router／keys・ArgumentScorer・formulaの数値は不変。新bankへの再較正が実契約上必要なら`DEPENDENCY_REQUALIFICATION_REQUIRED`として停止し、暗黙に学習しない。更新hashと互換性証拠は区別する。

準備済みchildはstaging／diagnostic scopeに留める。最終機能評価前に「全操作合格」や`NOMINAL_VALIDATED`を付与しない。名目上のbank完成と機能資格を分離する。

---

## 6. 段階E — 独立queryでRG3再判定

### E1. 評価の入口

選択したMIRROR_HALVESと固定3候補、child manifest、query hashをロックしてから評価する。**最終queryで評価する修正版は選択済みchild一つだけ。** parentは対照として同じqueryで測る。未選択A/Bを最終queryで競わせない。

全16操作各1024例でCorrect raw EM／token accuracy／lossを測り、適用可能な既存causal controlsを保存する。原floorは**非SHIFT15操作の各EM>=0.95**であり、平均やtoken accuracyへ変更しない。SHIFTも全表に表示し、既定のSHIFT限定例外以上の資格を付与しない。[S2・S3]

verifierはraw性能の失敗を隠すためのフィルタとして使わない。router top-1/top-kとargument診断は既存の非学習評価を別表に置き、L3/L4の研究Gateをここで免除・修正しない。

### E2. 不変性と組込みの確認

- 親の保護12操作：state／formula不変に加え、同じqueryでparentとchildの予測一致を要求する。
- 固定3候補：standalone＋同じCoreでのsource候補予測と、child内の予測が一致することを確認する。親の失敗予測との一致は要求しない。
- MIRROR_HALVES：選択された6000-step候補のstandalone予測とchild内予測を照合する。
- 親・共有cache・REC-004A snapshotは開始／終了時hash不変。childの4slot以外の意図しない変更はFAIL。

保護対象が同じ予測を返していても、新queryでそのEMがfloor未達ならRG3はFAILになる。不変性と十分な機能を混同しない。

### E3. Fresh-process確認と資格

学習processのメモリに依存せず、別working directoryの新processから明示manifestをloadする。全16操作の同一sample IDの離散予測一致、component hash、execution signature、scopeを検査する。

`get_or_build`、trainer、optimizer構築、router較正、別cache探索、欠落stateのrandom fallbackが呼ばれたら失敗する**dynamic guard**を使う。文字列scanだけで「実行されなかった」と証明しない。model skeletonの通常初期化は完全stateのstrict loadに先行できるが、欠落重みの補完には利用しない。

次を別フィールドにする。

```text
structural_load_pass
recovery_floor_pass
fresh_process_pass
nominal_reference_qualification
```

RG3の1024例point-estimate floorは、confidence-boundによる`REF_ADEQUATE`やB2全体のnominal認証の代替ではない。nominal scopeを要求するなら既定のreference契約の独立証拠が必要。未要求／未測定は`NOT_REQUESTED / NOT_EVALUATED`でよく、過大認証しない。[S2・S3]

### E4. 再判定とその後

全て必要：依存・学習来歴の成立、非SHIFT15操作の個別floor、SHIFTの既定扱い、保護対象不変、fresh-process一致、query非利用による選択、過大capability付与なし。

旧REC-004のRG3 FAILと、REC-004Aの`VALIDATION_TARGET_NOT_MET / NOT_EXECUTED`は保存する。新runの結果だけを`RG3_RECHECK_PASS / FAIL / UNRESOLVED`として追加する。

**最終queryで失敗した後、未選択条件や別stepに切り替えない。** 今回のqueryはそこで使用済みの結果として保存する。機能不合格なら不合格childをdiagnostic用に保持し、nominal公開しない。

RG3_RECHECK_PASSの場合だけREC-005へ、この操作別schedule／予算／初期化・データ生成規則を引き継げる。ただし本タスク内では5-model展開を開始しない。seeds11〜14での成功は未実証、結果を見てseedや予算を選び直さない。旧3候補の初期化履歴の確認可能性もmanifestに残し、コホート再構築に必要な情報が不明なら`COHORT_RECIPE_PROVENANCE_REVIEW_REQUIRED`を別blockとして報告する。

---

## 7. 実装先・CLI・必要なテスト

### 7.1 配置

現repoの等価なmodule／dispatcherを優先し、原レシピをglobal defaultで変更しない。下記の新規名は提案であり、実在APIの主張ではない。

```text
docs/CODEX_TASKS_PHASE_B_B2_MIRROR_SCHEDULE_REPAIR.md
src/apc/evaluation/incremental_budget_calibration.py       # 現在の実装を検査・必要最小限再利用
src/apc/evaluation/mirror_schedule_comparison.py           # 分離が必要な場合だけ追加
scripts/run_phase_b_b2_model_bundle_recovery.py            # --task REC-004Bを明示追加
scripts/rec004_fresh_process_check.py                     # manifest入力で再利用
configs/phase_b_b2_model_bundle_recovery_rec004b.yaml
tests/test_mirror_schedule_comparison.py
runs/phase_b_b2_model_bundle_recovery/rec004b/<run_id>/
```

実装後の推奨入口（現時点の実在commandとは限らない）：

```bash
python scripts/run_phase_b_b2_model_bundle_recovery.py --task B-C005REC-004B --config configs/phase_b_b2_model_bundle_recovery_rec004b.yaml
```

run-dirは既存CLI契約に従い、新namespaceへ出す。`--all`、latest自動探索、REC-005への自動継続を追加しない。既存REC-004Aの数値・task動作は後方互換を保つ。

### 7.2 CPU focused tests

1. A/B初期stateが完全一致し、factoryの構築順とcondition実行順で変化しない。
2. A/Bの6000-step入力列が同一で、途中eval／resumeがtraining RNG・data位置を変えない。
3. 実optimizerのparameter ID集合がMIRROR_HALVESだけと一致し、他componentがtiny training後も不変。
4. optimizer→scheduler順、実使用LR／更新後LR、上のLR表との一致、Bの単調減衰、Aの再上昇を検査する。
5. weightsだけのresumeを拒否し、完全checkpointの分割再開が連続実行とCPU tiny fixtureで一致する。
6. 6000終端のみが候補になり、B優先／A fallback／両未達／片方未完了が仕様どおり判定される。
7. validation途中最高値やfinal-queryをselection APIが受け取れない。query後に別候補を採用できない。
8. 固定3候補のstep=4000／Core／schema不一致を拒否し、欠落時に再学習・別step fallbackをしない。
9. childの変更が4slotだけであり、保護12と固定3の予測一致をそれぞれ正しい対照で検査する。
10. unqualified full capability、古いcertificate、raw floorとreference資格の混同、fresh-load中のbuilder呼出しを検出する。
11. NaN／skip／resource停止の該当stepをmissingとして残し、未実行を0%やPASSで埋めない。
12. 旧run上書き、shared cache書込み、sealed access、上限超過、暗黙次taskを拒否する。

標準検証：

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

focused testsから開始する。現repoのcanonical commandが異なるならそれを使い実commandを報告する。他ジョブを無断停止しない。旧2025件PASSを今回のcommitの試験結果へ転記しない。

---

## 8. 成果物・完了報告・停止規則

### 8.1 保存物

```text
config.yaml / protocol.json / system.json / source_audit.json
initial_state.pt / data_manifest.json / lr_trace.jsonl
A_FIXED_TMAX_1000/   checkpoints/ と training_states/
B_SINGLE_DECAY_6000/ checkpoints/ と training_states/
learning_curve.jsonl / error_breakdown.json / paired_comparison.json
fixed_candidate_manifest.json / selected_recipe.json / recipe_revision.json
freeze_audit.json / side_effect_audit.json / summary.json / report.md
# D/Eへ到達した場合のみ：
bundle_lineage.json / all_primitive_execution.json / protected_regression.json
rg3_recheck.json / fresh_process_report.json / qualification.json
```

source commit、framework version、device、precision、各条件updates／例数／時間／VRAM、各指標の分子／分母を記録する。未到達段階は`NOT_EXECUTED`と理由をsummaryへ書き、ファイルや測定値を捏造しない。

### 8.2 状態は独立に報告

```text
implementation_status: COMPLETE / PARTIAL
paired_comparison_status: COMPLETE_VALID / INVALID / NOT_EXECUTED / RESOURCE_BLOCKED
mirror_candidate_status: B_SELECTED / A_SELECTED / VALIDATION_TARGET_NOT_MET / BLOCKED
rg3_recheck: PASS / FAIL / UNRESOLVED / NOT_EXECUTED
nominal_reference_qualification: <既定scope別status、未評価を明記>
rec005_eligible: true/false
rec005_executed: false
```

「2条件を実行できた」「schedule仮説が支持された」「独立queryでRG3合格」は別の結果。どれか一つで他をPASSにしない。

### 8.3 報告する要点

```text
Task: B-C005REC-004B
Parent / Core / source3 candidates / initial stateのhash:
A/Bの共通条件と、意図した唯一のLR差分:
LR table / lr_usedとlr_afterの定義 / update skip:
Condition | train-fit@6000 | validation@6000 | length別結果 | 選択
MIRROR_HALVESのpaired delta / A-only vs B-only correct:
3候補の4000-step固定と、旧最小stepからの改訂:
独立query：全16操作、非SHIFT各floor、SHIFT状態:
保護12の一致 / 固定3のsource一致 / MIRRORの組込み一致:
Fresh-process / qualification / shared-cache side effects:
Implementation / comparison / candidate / RG3の各status:
Tests / commands / artifacts / ADR / deviations:
残るblock / 次タスク（未実行）:
```

### 8.4 終了時の規則

| 状況 | 処理 |
|---|---|
| source／意味論／LR／pairing／freeze違反 | 原因を保存して停止。比較成立・RG3 PASSにしない |
| A/B終端とも未達 | 学習比較完了、RG3再判定未実施。6000超へ自動延長しない |
| 候補はあるがdependency／qualification未確定 | diagnostic artifactを保持し、認証・公開・RG3 PASSを停止 |
| 最終queryで未達 | 新runのRG3 FAIL。query後のcheckpoint再選択・追加学習なし |
| 全再判定条件PASS | 改訂レシピと資格を保存。REC-005の実行は別指示 |

未達でもoperator容量不足とは確定しない。次の提案は、観測された最も明確な機構一つに限定する。位置対応診断・小さな位置bias・容量比較は候補になり得るが、本タスク内では実装しない。

**STOP：本タスク終了後は結果を報告する。REC-005、別の修正、R3-011／012、B-C006、Task Inferenceを自動開始しない。**
