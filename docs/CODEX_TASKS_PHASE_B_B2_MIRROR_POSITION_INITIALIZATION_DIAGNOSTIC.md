> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# AI Coding Task — MIRROR_HALVES Position Correspondence & Initialization Diagnostic

**正式ID：B-C005REC-004C**  
**版：1.0 / 2026-09-08 / 状態：追加指示。コード実装・実験は未実施。**  
**位置：REC-004Bの候補未達後、REC-005の前。診断専用の一件。**

## 0. 目的と終了境界

保存済みcheckpointの位置別誤りを先に調べ、その後、**同一Core・同一訓練stream・同一レシピで、MIRROR_HALVESの初期重みだけを変える5試行**を行う。区別するのは、実装／評価契約の不整合、初期化に伴う学習のばらつき、共通する位置対応の失敗、終盤の学習進行である。

**成功する初期化を探して採用するタスクではない。** 途中または終端でEMが0.95以上になっても、candidateを選択・bankへ組込み・通常公開しない。RG3再判定も実施しない。Core／operatorの大型化、位置biasの追加、別LR探索は行わない。

ユーザーが本タスクを指定した場合、以下のA〜Eを前提成立の範囲で一件として実行する。段階ごとの再承認は不要。ただしSTOP条件では依存作業を止める。

```text
REC-004：RG3 FAIL                                       ← 保存
REC-004A：VALIDATION_TARGET_NOT_MET / RG3 NOT_EXECUTED    ← 保存
REC-004B：VALIDATION_TARGET_NOT_MET / RG3 NOT_EXECUTED    ← 保存
    ↓
REC-004C
 A. source・モデル・実行契約の監査（新規学習なし）
 B. 保存済みモデルの位置対応／padding／曲線診断（新規学習なし）
 C. 新しい5初期化の診断protocolを固定
 D. 固定レシピで5本を完走し、全試行を分析
 E. 証拠と未確定事項を整理してSTOP
    ↓
新しい修正・候補採用・RG3再判定は別の明示指示
```

REC-005以降、R3-011／012、B-C006、Task Inferenceは結果にかかわらずblockedのまま。診断が完了したことと研究Gateが合格したことを混同しない。

---

## 1. 根拠、解釈、新設条件

### 1.1 読む資料

- **S1：ADR-0097の実行報告。** [原文snapshot](research/evidence/REC004B_ADR0097_SUPPLIED.md)。本タスクの開始点。
- **S2：ADR-0096の実行報告。** [原文snapshot](research/evidence/REC004A_ADR0096_SUPPLIED.md)。旧曲線・6000-step重みの比較元。
- **S3：REC-004Bの指示書。** 現repoの `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_SCHEDULE_REPAIR.md`。[配布時snapshot](research/evidence/REC004B_TASK_SUPPLIED.md)。parent、freeze、データ役割、LR契約を照合する。
- 現repoのModelBundle契約、復旧task／plan／受入条件、REC-004A/Bの実config・manifest・JSON、root `AGENTS.md`。

資料の数値は実行担当者の報告であり、この文書の作成時に実repoのコード・checkpoint・JSONを独立再実行した結果ではない。実在path、physical ID、全optimizer引数は実装時に照合する。見つからない情報を一般的defaultや過去の別実験から補わない。

### 1.2 報告された事実

| 対象 | Validation EM @6000 | Train-fit EM @6000 | 扱い |
|---|---:|---:|---|
| REC-004AのMIRROR_HALVES | 0.6875 | 0.7061 | 履歴比較。今回の新試行へ混ぜない |
| REC-004B A：T_max=1000 | 0.4717 | 0.5029 | 同一初期化A/B比較の対照 |
| REC-004B B：T_max=6000 | 0.4336 | 0.4268 | このpilotで修正有効性は示されなかった |

S1はA/Bの初期stateと各stepの入力が一致したと報告している。長さ別EMはAが `{6:41.7%, 7:40.2%, 8:57.7%, 9:59.2%, 10:37.9%}`、Bが `{6:53.9%, 7:52.8%, 8:30.9%, 9:28.2%, 10:50.0%}`。以前の単調な長さ依存は再現しなかった。[S1 Evidence 1, 3, 4]

固定3候補の4000-step validationは、CYCLE_FOUR=0.9912109375、ROTATE_TRIPLETS=0.998046875、SWAP_ENDS=0.9970703125としてsource replayで再現済み。REC-004A/BともchildとRG3再判定は未実施。[S1 Evidence 5, 7]

### 1.3 本タスクで確定事項として採用しない解釈

- 「容量／attention spanが原因」「初期化だけが原因」は仮説のまま。
- train-fitとvalidationが近いことだけから、学習進行・停滞・全種類の汎化問題を断定しない。
- Wrong／Noneが低いことを、あらゆるleakageの不存在証明にしない。
- 旧runと新runの差を初期化へ帰属するには、他の条件の一致証拠が必要。同一`model_seed`だけでは不十分。
- attention weightの最大位置、表現距離、出力の相関だけを、内部で実際に使われた入力位置の因果的証明としない。

S1/S2の原文は保存する。解釈を修正する場合は新しいADRへ、対象範囲と根拠を追記する。

### 1.4 今回新設する診断条件

| 項目 | 新設する条件 |
|---|---|
| Core | REC-004と同じmodel seed10の固定Core一つ |
| 新規初期化 | **I01〜I05の5個**。実行前に全初期stateを作成・hash固定 |
| 新規学習 | MIRROR_HALVESのみ、**各6000 updates、合計最大30000 updates** |
| 固定schedule | REC-004B Aと同じ `CosineAnnealingLR(T_max=1000)` の機械的延長 |
| LR | max=0.0008、min=0.00001。その他optimizer／loss／batch／dtypeはS1の実configを継承 |
| 計測点 | 0、500、1000、…、6000。主要集約は全試行の6000のみ |
| 共通主評価 | REC-004A/Bの既存固定validation 1024例を再利用・hash照合 |
| 長さ均等診断 | 報告された長さ6〜10のうち実仕様上合法な各長さ256例。最大1280例 |
| 位置識別用診断 | 同じ合法長さごと最大64例。最大320例。合法語彙で位置を区別可能な場合のみ |
| counterfactual用 | 同じ合法長さごと最大16基底例。各有効位置につき最大1回の合法token置換 |
| 新しい最終query | **生成・評価しない**。RG3/B2の確認用データを消費しない |
| 候補選択・公開 | **なし**。5本の中の最高値を採用しない |

これは新たな診断予算であり、REC-004Bの上限をそのまま延長する操作ではない。A固定は比較軸を減らすためであり、Aの優位性が一般に確立したという主張ではない。

---

## 2. 変更・更新の許可集合

### 2.1 許可するもの

- 評価専用の位置対応表、合法入力fixture、mask／padding検査、既存checkpoint読込、誤り集計、forward hookによる読み取り計測。
- 新しい5個の隔離MIRROR_HALVES candidateの、既存レシピによる学習。
- 完全な学習状態の保存、再現監査、focused tests、薄いdispatcher追加、ADR／navigation更新。

### 2.2 不変にするもの

- Core、task側、decoder/readout、vocabulary、argument／position schema、学習済みbuffer。
- **parent bank内の全16 primitive原本**。新candidateはbankへ登録せず隔離する。
- とくにMIRROR_HALVES以外の**15操作**、router全keyと重み、ArgumentScorer、lambda／formula、controller、verifier。
- REC-004Aの3固定候補、REC-004BのA/B・初期state・学習state原本、共有cache。

実学習optimizerは、隔離したMIRROR_HALVES一つのparameter ID集合だけを受け取る。prototype factoryの呼出しがbank登録・他component初期化へ波及しないようにする。

**数え方：** 以前の「child assemblyで4slotを置換し、残り12を保護」という規則は今回は使わない。childを作らないため、旧parent全16slotと固定3候補はread-only。新学習中の非対象操作は15個。S1の「12 protected non-SHIFT + SHIFT」はphysical ID一覧で照合し、表記と実更新集合を別々に検査する。

### 2.3 禁止する変更

Core再学習、operator型／容量変更、position bias・relative position機構追加、oracle gatherでのruntime置換、別LR／optimizer／loss、curriculum、データ再重み付け、seed探索の追加、学習済み旧重みからのfine-tuning、新probe学習、router再較正、verifier変更、bundle資格の緩和。

**教師・indexing・padding等の本物の実行契約違反が再現した場合：** 最小再現例を保存して依存学習を止める。バグ修正と初期化比較を同時に行わない。新規診断コード自身の集計バグはsourceモデルを変えない範囲で修正できるが、その差分と再集計を記録する。

---

## 3. 段階A — Sourceと実行契約の監査（新規学習なし）

### A1. Source manifest

次はS1/S3記載の確認対象。ファイルの存在とhashを確認して明示configへ保存する。最新runや別seedを自動探索して置き換えない。

```text
# REC-004 parent（1行の実pathとして扱う）
runs/phase_b_b2_model_bundle_recovery/bundles/
  2356543740ce566640f72767e73bd83955bc27bb825bb06c8fcffab03cf53995/manifest.json

# REC-004A
runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/
  budget_protocol.json / operation_contract_audit.json
  learning_curve.jsonl / error_breakdown.json / checkpoints/

# REC-004B
runs/phase_b_b2_model_bundle_recovery/rec004b/run_001/
  source_audit.json / protocol.json / config.yaml / data_manifest.json
  initial_state.pt / lr_trace.jsonl / learning_curve.jsonl
  error_breakdown.json / fixed_candidate_manifest.json
  A_FIXED_TMAX_1000/checkpoints/ + training_states/
  B_SINGLE_DECAY_6000/checkpoints/ + training_states/
```

REC-004A/Bは完成child bundleを持たない。parent、対象primitiveの型・physical ID・Core依存、旧snapshotの完全path、raw hash・canonical state hash・execution signatureを `source_manifest.json` に列挙する。

最低限の開始条件は、parentとREC-004Bの初期state／A・B終端checkpoint／実レシピ／validation入力の再現が確認できること。欠落は `SOURCE_ARTIFACT_UNAVAILABLE` で新規学習前に停止。REC-004Aの対象checkpointだけが欠落なら、その履歴比較を `UNAVAILABLE` とし、確認できるREC-004B経路と新試行は継続可能。欠落sourceを再学習して過去結果の代用にしない。

### A2. 履歴比較の成立範囲

保存済みMIRROR_HALVES各終端を同じ旧validationへ一度通し、保存指標・可能なら予測digestと照合する。差があれば丸め、入力hash、型／formula、mode、precision、checkpointを調べる。

- 同じsource checkpoint／入力を再現：`SOURCE_REPLAY_VERIFIED`。
- 入力／stateを同一に復元できない：`DESCRIPTIVE_ONLY`、差の因果帰属不可。
- 確認済み同一条件なのに不一致：`SOURCE_REPLAY_MISMATCH`で依存学習STOP。

旧REC-004AとREC-004B Aの比較は、訓練sample列だけでなく、初期化以外のtraining RNG、dropout、optimizer、実更新数、LR列、batch順、精度の一致可能性を表にする。不明は `UNKNOWN`。完全一致が確認できない旧比較を「初期化だけを変えた実験」としない。

### A3. 実際のMIRROR_HALVES定義を読む

現operation定義、generator、`operation_contract_audit.json`、factory、forward、loss、decoder、position encoding、mask構築を読む。**名前から変換を補わない。** 次を記録する。

- 合法な長さ・token範囲・引数、奇数長／区間境界／端数の扱い。
- 出力長と順序、padding対象、loss／EMのmask、絶対position IDの付与。
- 内容と無関係な固定位置写像か、値依存か。pure permutationだと仮定しない。
- primitiveとCoreのattention参照範囲、実装上のmask、長さ情報の渡し方。

小さな合法例をsource仕様から手で追えるfixtureにして、教師・評価の両側を検査する。抽象的な位置ラベルを仕様テストに使う場合も、それを未登録tokenとしてneural入力へ注入しない。

内容非依存の写像が成立する場合だけ、評価用に `pi_n(i)`（出力位置iが取るべき入力位置）を定義する。位置の向きを明記し、`y[i] = x[pi_n(i)]` とsourceの出力が合法例で一致することを検証する。写像を導出できなければ `POSITION_MAP_NOT_APPLICABLE_OR_UNRESOLVED`。適用できる出力誤り・初期化比較は継続できるが、写像に基づく原因確定は行わない。

### A4. 契約監査の出口

意味論の不一致、Core依存不一致、対象外の更新、full capability過大認証を認めるloaderに依存せず診断loadも成立しない場合はSTOPする。明示diagnostic scopeのloadは許可するが、未達parentへnominal全機能資格を付けない。

出力：`source_manifest.json`、`historical_comparability.json`、`operation_contract_audit.json`、`freeze_scope_manifest.json`。

---

## 4. 段階B — 保存済みモデルの位置対応・実行診断（新規学習なし）

### B1. 既存曲線と終端の比較

対象はREC-004AのMIRROR_HALVES終端（取得できた場合）、REC-004B A/B終端。曲線は保存ログを優先し、存在しない観測点を補間して実測扱いしない。

終盤4000〜6000について、500ごとのloss／EM／token accuracyと長さ別値を読み取り、同じLR位相となる点の差も分けて報告する。終端の一値、train-fitとvalidationの差だけで「停滞」「まだ十分に進行中」を決めない。曲線が取れなければ `LATE_CURVE_UNAVAILABLE`。

既存観測点を再評価する場合は、最大でS1記載の13点×2条件とS2記載の4点のみ。新学習・checkpoint追加・6000超の続行はしない。過去最高のcheckpointを採用候補へ格上げしない。

### B2. 出力誤りと位置の取り違え

主評価1024例について、最低限、以下を全て分子／分母付きで保存する。

- sequence EM、有効出力token accuracy、loss、長さ別成績。
- `(実系列長n, 出力位置i)` ごとの正解率、first-error位置、各系列の誤り数。
- source仕様の区間境界／中心／端の失敗と、その他位置の失敗。
- **構造的移動位置**（写像がある場合）と、**見かけ上の変更位置** `x[i] != y[i]` の別集計。
- COPY／Noneのような既存対照が解ける入力の比率。同値入力を主評価から除外しない。

純粋な位置写像の場合、予測token `y_hat[i]` と同じ値を持つ入力位置集合を `J_hat(i)={j | x[j]=y_hat[i]}` として扱う。

- 要素数1：一意な出力値由来の位置推定。
- 要素数2以上：`AMBIGUOUS_SOURCE_TOKEN`。都合のよい一位置へ割り当てない。
- 要素数0：`PREDICTED_TOKEN_NOT_IN_INPUT`。最寄り位置へ強制分類しない。

一意な場合のみ、正解位置、off-by-one、区間取り違え等をsource定義に即して分類する。分類でunknownを消さない。**これは出力値と整合する入力位置の分析であり、内部attentionの使用位置の証明ではない。**

### B3. 診断用の合法入力

各suiteのrecipe・count・入力hashを評価前に固定し、モデル出力を見て例を選ばない。

1. `length_balanced_diagnostic`：合法な6〜10の各長さ256例。語彙／content生成規則は現在のgeneratorを維持し、長さquotaだけ均等にする。元validationとは別分布として集計し、RG3 EMに置き換えない。
2. `position_identifiable_diagnostic`：合法語彙で全位置の値を一意にできる場合、各長さ最大64例。できなければその長さを `NOT_APPLICABLE` とし、語彙を増やさず上記の集合値分析を使う。高precisionの特殊入力上の成功を通常分布の成功と呼ばない。
3. `content_counterfactual_diagnostic`：各長さ最大16の合法な基底例を固定。各有効位置のtokenを一度だけ別の合法値へ変え、元入力・変更入力・各teacher出力を保存する。

3では、teacher上の変化位置とmodel出力／logit変化位置の対応を調べる。内容非依存写像なら、入力位置jから出力位置iへの期待される対応と比較できる。変更で合法性を失う例は作らない。modelの過剰反応・無反応は機能誤りの証拠だが、それだけで特定attention headやCoreの原因と断定しない。

生成可能性不足なら、出力と無関係な試行上限を事前固定して不足件数を報告する。成功例が揃うまで無制限生成しない。これらの診断例を新試行の訓練へ混ぜない。

### B4. Padding／batchingのmetamorphic検査

各合法長さから事前固定した最大16例について、同じcontentを次の**API上合法で、意味を変えない条件**で比較する。

- 単体実行と、他の合法例を混ぜたbatch実行。
- 実系列のposition IDを変えない右padding長の変更。
- 明示maskがある場合の、maskされた領域の合法な置換。

有効tokenと位置を同じにし、crop後の `h_content`、primitive出力、decoder予測を段階別に照合する。eval modeと既存device／dtypeを固定し、数値許容差はREC-004Bの既存契約を事前記録する。浮動小数点の微小差やargmaxのnear-tieを即座にindexing bugとしない。

APIがmixed-length batchやpad値変更を許可しない場合、そのケースは `UNSUPPORTED_METAMORPHIC_CASE` とする。左paddingで絶対positionを変えた入力等を同じ意味だと勝手に扱わない。

明示契約に反する安定した差が最小例で再現すれば `EXECUTION_CONTRACT_FAILURE`。原因componentの修正はせず、新規5試行を止める。数値再現性・意味の同一性を解決できなければ `EXECUTION_CONTRACT_UNRESOLVED` とし、同様に学習を止める。モデルが通常入力の変換を間違えること自体は、実装契約違反とは別であり、初期化比較を止める理由にしない。

### B5. Attention観測の上限

既存APIで、計算を変えずにattention／中間stateを観測できる場合だけ、位置誤りとの関連を評価専用に記録してよい。hook有無で予測が一致することをテストする。

attention取得のためのkernel変更、mask／softmax変更、追加位置特徴の注入、学習可能なprobeは許可しない。取得不能なら `INTERNAL_ATTENTION_NOT_OBSERVABLE` で十分。attention最大値だけから「その入力が原因」「attention span不足」と結論しない。

出力：`historical_replay.json`、`late_learning_curve_audit.json`、`position_error_summary.json`、`position_confusion.json`、`padding_batch_audit.json`、`counterfactual_dependency.json`、`diagnostic_data_manifest.json`。

---

## 5. 段階C — 5初期化のprotocol固定

### C1. 入口

A/Bの監査が成立し、未解決のsource不整合・実装契約違反がない場合だけ進む。位置写像・内部attentionが観測不能でも、合法な入出力と実行契約が確定していれば、出力に基づく初期化比較は可能。

### C2. 初期化と訓練乱数を分離

- `model_seed=10`はCoreの識別子として固定。
- `init_id=I01...I05`は新しい初期重みの識別子。**データseedやCore seedへ変換しない。**
- 実repoのversion付きseed utilityを用い、`rec004c_init:MIRROR_HALVES:<init_id>` 相当の固定namespaceから各init seedを導出する。導出法をprotocolに保存し、Pythonの組込みhashを使わない。
- primitive factoryの既存の初期化分布・型・容量は変更せず、局所RNGで5stateを先に生成する。`initialization_manifest.json`と初期weightsを保存し、5個のcanonical hashが異なることを確認する。state生成後、成績を見てseedを差し替えない。
- 各試行は空のoptimizer状態から開始。初期weights作成後に、**全試行共通のtraining RNG状態**へ戻す。dropout・augmentation等の訓練乱数を、init抽選の消費量と切り離す。
- data streamはREC-004B Aの同じ純粋生成規則を利用し、各updateで全5試行のsample・content・target・batch順digestが一致することを確認する。init_idをdata seedへ混ぜない。

新5試行内の違いを初期stateへ限定する。旧REC-004A/Bは補助的履歴であり、新5本の統計量へ混ぜない。共通training RNGが旧runと違う場合も開示し、旧runとの差を初期化のみへ帰属しない。

### C3. レシピと観測点

全5本にREC-004B Aの既存optimizer／loss／batch／precision／LR traceを適用する。`T_max=1000`、lr=0.0008、eta_min=0.00001、6000 updates。reset／warm restart／warmup追加なし。6000を超えない。

学習前のCPU preflightで実versionのLR traceを旧Aと照合する。`lr_used`と`lr_after_scheduler`を別に記録し、optimizer更新後にschedulerを進める既存順序を維持する。LR preflightの警告を理由に本学習の更新順を変更しない。違えば `TRAINING_RECIPE_MISMATCH` として停止する。

0、500、…、6000で同じ計測をする。0は学習前、主比較は6000。全試行を6000まで実行し、0.95到達時の早期打切り・低成績試行の除外・途中最高値選択をしない。

### C4. データ役割と露出

| role | 使い方 | 禁止する使い方 |
|---|---|---|
| `train` | 既存分布・全試行共通の6000-step列 | 初期化ごとの分布変更 |
| `train_fit` | 実際に露出した同じ固定例、最大1024 | 新しい例をtrain精度と呼ぶこと |
| `existing_validation` | 旧固定1024例・全試行共通 | 新holdout／最終Gateと呼ぶこと |
| B3の3診断suite | 層別位置／counterfactual分析 | 勾配計算・curriculum化 |
| RG3/B2 final query・reference・sealed | **本タスクでは使わない** | 診断に流用すること |

既存validationは繰り返し利用済みのdevelopment資料。新初期化診断も探索的な一Core上の知見として扱う。

`train_fit`は全試行の共通訓練prefixから、sample位置を固定して採取する。取得不能な履歴train-fitは `UNAVAILABLE`。新しい同分布の例で埋めない。

別role streamと `(task, arguments, content)` digestで、現在の訓練とvalidation／新診断suiteの重複を監査する。新診断例の重複除外はモデル出力を見ない決定的手順で行い、訓練streamは途中変更しない。旧validationの不整合・重複が見つかれば、分母を書き換えて旧値を再現したふりをせず、問題を保存し比較を停止する。Core等の全履歴露出が追跡できない範囲はUNKNOWN。

長さ均等・位置識別・counterfactual suiteの結果は別表とし、通常validationへの合算や得点改善には使わない。

### C5. Protocol lock

`initialization_protocol.json`へ、Core／source／5初期state hash、共通training RNG、各実引数、LR trace、データroleとhash、計測点、上限、許容差、metric定義、比較対象、STOP条件を保存してから学習する。

profileやBの結果で新しいLR・モデル容量・init分布を選ばない。この文書の固定条件が実装できない場合は、条件を変更せず `PROTOCOL_BLOCKED` で停止する。

---

## 6. 段階D — 新規5試行を実行・集約

### D1. 保存と失敗の扱い

各updateのloss、sample/batch digest、実LR、累計更新・例数、optimizer対象を記録する。500ごとにweights、optimizer、scheduler、使用時AMP scaler、CPU／device RNG、data cursor、protocol hashを保存する。

装置中断からは完全state・同一protocolで厳密resumeできる。上限をリセットしない。旧6000-step重みからの学習延長や、weight-only restartを継続と呼ぶことは禁止。

source改変・対象外更新・data／LR契約破綻は全依存学習をSTOP。NaNなどその初期化固有の数値失敗は、発生stepと欠測を保存し、その試行を打ち切る。他に契約違反がなく隔離が保たれる場合のみ、残りの事前指定試行を続ける。失敗trialの交換・追加は行わない。

### D2. 出力指標

各試行・各観測点で、train-fitとvalidationのsequence EM、token accuracy、loss、長さ×位置別誤り、先頭誤り位置、累計費用を報告する。B3の診断suiteは終端6000で全試行に対して実行する。中間を追加測定して最高値を選ばない。

終端で、None／Wrong-family等の既存causal controlsを同じ例で評価する。対照callを明示実行することと、primary forwardが非選択primitiveを裏で実行することを区別する。無効な引数対照や、等価入力に対する強制gapを作らない。

### D3. 初期化別のばらつき

最低限、以下を `initialization_summary.json` に出す。

- planned=5、completed、diverged、missing、各initの初期／終端hash。
- 各initの終端validation EM、長さ別EM、train-fit EM、loss。
- 完了trialのmean、sample SD、min、max、range。完了数と算出規則を明記し、2未満ならSD=null。
- 0.95以上だった本数／完了本数、および全予定5本に対する観測達成本数。未実行をEM=0やPASSで埋めない。
- 同じ入力上の試行pairごとの正誤不一致率、両方誤り／片方だけ誤りの件数。
- 各例／位置を誤ったtrial数の分布（0〜5。欠測を分ける）。length×position heatmapと分母。
- 誤り集合のJaccardを使う場合、両集合が空ならnullとし、「完全一致する失敗」と見せない。

最大値を代表値にしない。5試行は**同じCore・同じデータ列・同じレシピ上の初期重みの反復**であり、5独立Coreでも、5種類の未見task familyでもない。標本数5と共通入力の制約を開示し、大規模な初期化分布の成功率を確定しない。

### D4. 0.95到達と診断完了の関係

0.95は従来の復旧floorに対応する**記述用の参照値**として本数を数えるだけ。1本でも5本でも超えてよいが、`selected_init=null`、`child_bundle=null`、`rg3_recheck=NOT_EXECUTED`を維持する。

特定initで高精度に到達したなら「この固定Core／構造／レシピに到達例がある」と報告できる。初期化を選べば復旧が終わる、容量問題が全て否定された、他Coreでも安定するとは言わない。

---

## 7. 段階E — 診断結果と次の一手を整理してSTOP

次の表に、数値・raw artifact path・適用scope・未確定事項を記載する。条件付きの推論を、報告事実に混ぜない。

| 観測 | 許される解釈・次タスク候補 | このタスクでは行わないこと |
|---|---|---|
| 合法な同一入力のpadding／batch条件で、契約違反が再現 | 実行契約の不具合、または再現性の未解決。最小修正を優先 | 修正してそのままinit比較を続ける |
| 初期state以外の一致が確認でき、終端・誤り位置に試行差 | このCore／固定streamで初期化依存のばらつきを観測 | 最良seedの採用、安定性の一般化 |
| 複数initで同じ位置・境界に誤りが集中 | 共通する位置対応の学習困難が候補。位置情報／scoringの診断・修正を次に検討 | 直ちにcapacity／attention spanと断定 |
| initごとに失敗位置が異なり、一部が高精度 | 最適化／初期化に対する安定性改善が候補 | 成功initだけ公開、失敗trialを除外 |
| 終盤にも同位相の曲線で改善が続く | この予算内で未収束の可能性。次の有限延長の是非を提案可能 | 6000超を自動実行 |
| train-fit高、独立例だけ低い | 測定scopeでのgeneralization gapを調べる | 表現能力だけへ帰属 |
| 全trial低い／曲線停滞 | 現レシピの失敗が反復。原因はなお未分離になり得る | 「小型operatorには原理的に不可能」と結論 |

### E1. 回答必須の問い

1. MIRROR_HALVESの実仕様は何か。内容非依存の位置写像は成立するか。
2. 旧A/Bの結果は同じ入力で再現したか。旧REC-004A比較の交絡は何か。
3. padding／batch／maskで意味を変えずに実行したとき、同じ結果になるか。
4. どの長さ×出力位置が誤りやすいか。反復tokenの曖昧さを除くと何が分かるか。
5. counterfactual出力はどの対応を示すか。内部attentionへの帰属はどこまで可能か。
6. 新5試行の違いは本当に初期stateだけか。共通訓練乱数と入力一致を検査したか。
7. 終端性能と失敗位置はinit間でどれだけ変わるか。共通する失敗はあるか。
8. 終盤の曲線は改善／停滞のどちらをどの範囲で支持するか。
9. 容量、位置学習、初期化、実行契約のうち、何が支持され何が未確定か。
10. 次に変更すべき一つの機構は何か。証拠が不足すれば `UNRESOLVED` でよい。

初期化・位置の単一原因を必ず選ぶ必要はない。未確定を明示した診断も有効な完了結果。source／pairingが壊れたまま因果診断完了としない。

### E2. 完了条件と状態

本タスクの完了条件は性能0.95ではなく、source／実行契約、保存済み診断、事前指定試行の実行状況、freeze・露出・分母、結論のscopeが明示されていること。数値失敗で打ち切ったtrialも結果として残す。

```text
implementation_status: COMPLETE / PARTIAL
source_audit_status: VERIFIED / PARTIAL_HISTORY / BLOCKED
historical_diagnostic_status: COMPLETE / PARTIAL / BLOCKED
execution_contract_status: PASS / FAILURE / UNRESOLVED
initialization_experiment_status:
  COMPLETE / COMPLETE_WITH_NUMERICAL_FAILURES / PARTIAL / NOT_EXECUTED
mechanism_diagnosis: <supported labels + unresolved questions>
selected_init: null
child_bundle: null
rg3_recheck: NOT_EXECUTED
rec005_eligible: false
```

`DIAGNOSTIC_COMPLETE`と表示してもRG3 PASSにしない。全5試行を完了しても新規学習済みcheckpointは診断用途のみ。単一capacity limit等を否定／支持した範囲を限定して、実repoの次の未使用ADRへ記録する。ADR番号を0098と決め打ちしない。

---

## 8. 配置・CLI・テスト・成果物

### 8.1 配置

既存の等価なmoduleを優先し、原レシピやglobal defaultを変更しない。

```text
docs/CODEX_TASKS_PHASE_B_B2_MIRROR_POSITION_INITIALIZATION_DIAGNOSTIC.md
src/apc/evaluation/mirror_schedule_comparison.py              # reported existing; 計算部分を再利用
src/apc/evaluation/incremental_budget_calibration.py          # reported existing; source／data参照
src/apc/evaluation/mirror_position_initialization_diagnostic.py # 必要な場合のみ追加
scripts/run_phase_b_b2_model_bundle_recovery.py              # --task REC-004C 明示dispatch
configs/phase_b_b2_model_bundle_recovery_rec004c.yaml
tests/test_mirror_position_initialization_diagnostic.py
runs/phase_b_b2_model_bundle_recovery/rec004c/<run_id>/
```

新module／config名は提案。実APIを確認して最小差分にする。推奨CLIは次の形だが、実装済みとの主張ではない。

```bash
python scripts/run_phase_b_b2_model_bundle_recovery.py --task B-C005REC-004C --config configs/phase_b_b2_model_bundle_recovery_rec004c.yaml
```

run-dir指定は既存CLIに従う。旧`run_001`を上書きしない。`--all`、latest自動選択、次task自動開始を追加しない。root `AGENTS.md`、復旧task/plan、`docs/DECISIONS.md`と現在の復旧ADR logへ分岐・block状態を追記する。旧本文を消さない。

### 8.2 必須CPU tests

1. 位置写像の向きとsource teacher一致、合法な奇数長・境界の検査。
2. 重複tokenのsource位置を集合で扱い、unknown／empty denominatorをnullとして残す。
3. moved位置とchanged位置を区別し、特殊診断suiteと主評価を合算しない。
4. 意味を変えないpadding／batch条件だけを比較し、unsupported条件をbug扱いしない。
5. 読取hookの有無で予測が一致する。診断teacher／対応表がruntime forwardへ入らない。
6. init_idがCore／data seedを変えず、5stateを先に固定し、同一labelから同じstateを再生成できる。
7. init作成後に共通training RNGへ戻り、試行順・途中evalで訓練streamが変わらない。
8. optimizer対象が隔離MIRROR_HALVESだけ。parent16・他component・固定3候補が不変。
9. 固定LR・6000上限・500計測、合格しても早期採用／追加seed／追加stepが起きない。
10. 完全stateの分割再開がCPU tiny fixtureで一致し、weight-onlyを厳密resumeとして受け入れない。
11. NaN／欠測試行の分母と保存、min/max/SD・pairwise比較が正しい。失敗trialを削除しない。
12. 偶然全5試行が0.95以上でも、bundle publish／RG3 query／REC-005が呼ばれない。
13. evaluation内のbuilder／optimizer／較正呼出しをdynamic guardで検出する。明示D学習区間とは分ける。
14. original artifact上書き、共有cache write、sealed accessを拒否する。

小さなCPU fixtureの成功を実GPU上のMIRROR_HALVES成功として扱わない。focused tests後にrepo標準検証を実行する。

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

現repoに別のcanonical commandがあればそれに従い、実command・exit statusを報告する。旧2048件PASSを新commitの結果へ転記しない。他の実験processを無断で停止しない。

### 8.3 成果物

```text
config.yaml / protocol.json / system.json / summary.json / report.md
source_manifest.json / historical_comparability.json / historical_replay.json
operation_contract_audit.json / freeze_scope_manifest.json
late_learning_curve_audit.json / diagnostic_data_manifest.json
position_error_summary.json / position_confusion.json
padding_batch_audit.json / counterfactual_dependency.json
initialization_manifest.json / initialization_protocol.json
initial_states/I01.pt ... I05.pt
I01/ ... I05/  # learning_curve、lr/data trace、checkpoints、完全training_states
initialization_summary.json / cross_init_error_overlap.json
freeze_audit.json / side_effect_audit.json / final_diagnosis.json
plots/  # 必要最小限のcurve・length×position heatmap
```

未到達段階のファイルを捏造しない。JSONの件数・分母を正とし、plotだけから主張しない。source commit、device、framework version、精度、updates／training examples、時間、VRAMを記録する。

### 8.4 完了報告の書式

```text
Task: B-C005REC-004C
Implementation / diagnostic status:
Parent Core / source checkpoint hashes / required diagnostic scope:
Historical replay / cross-run comparability / unresolved RNG differences:
MIRROR_HALVES semantics / mapping applicability / legal lengths:
Padding/batch/length contract findings:
Historical length×position errors / counterfactual observations:
Init ID | initial hash | updates | train-fit | validation EM | per-length EM | status
Planned/completed/failed/missing trials / distribution / error overlap:
Protected parent16 / non-target15 / fixed3 / router/Core freeze audit:
Interpretations supported / not supported / still unresolved:
Tests / commands / artifacts / ADR / deviations:
selected_init=null / child_bundle=null / rg3_recheck=NOT_EXECUTED:
REC-005 onward / R3-011/012 / B-C006 / Task Inference: blocked
Next single mechanism recommendation (not implemented):
```

**STOP：新しい修正や候補採用を自動で実装しない。診断結果をユーザーへ報告して、この一件を終了する。**
