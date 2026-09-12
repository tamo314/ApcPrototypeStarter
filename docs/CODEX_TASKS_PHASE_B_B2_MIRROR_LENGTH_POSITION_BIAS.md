> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.
> Closeout provenance note: three historical supplied-attachment links below are missing; see the final ledger audit note. They are preserved as gaps and are not reconstructed evidence.

# AI Coding Task — MIRROR_HALVES Length-Conditioned Position Bias & RG3 Recheck

**正式ID：B-C005REC-004D**  
**版：1.0 / 2026-09-09 / 状態：新しいタスク指示。プロジェクトの実装・実験は未実施。**  
**位置：REC-004Cの診断完了後、REC-005の前。一件の限定的な修正比較。**

## 0. 目的・実行権限・終了境界

固定位置の診断指標を確認・必要なら訂正し、**MIRROR_HALVESの既存compact operatorへ、小型の長さ条件付き位置スコアを一つ追加する**。REC-004Cで保存した5初期stateを使い、現行operatorと同じCore・同じ訓練入力・同じ6000更新で比較する。

本タスクは「位置スコアが根本原因」「容量不足だった」と事前に断定しない。新機構が到達性能と初期化に対する安定性を改善するかを検証する。採用条件を満たした場合だけ、事前固定した一候補をbankへ組み込み、全16操作の独立queryとfresh-processでRG3を再判定する。

ユーザーが `B-C005REC-004D` を指定した場合、下記A〜Fを前提成立の範囲で一件として実行できる。段階ごとの再承認は不要。ただしSTOP条件を飛ばさず、REC-005や別の修正へ自動継続しない。

```text
REC-004：RG3 FAIL                                          ← 保存
REC-004A/B：VALIDATION_TARGET_NOT_MET / RG3 NOT_EXECUTED      ← 保存
REC-004C：DIAGNOSTIC COMPLETE / candidateなし / RG3未実施    ← 保存
    ↓
REC-004D
 A. source確認・固定位置指標の是正・length経路の確認
 B. 小型position-bias実装・初期等価性・保存契約の検査
 C. 5ペアの初期state／データ／レシピ／選択規則を固定
 D. U（現行）対P（位置スコア追加）、各5本×6000更新
 E. 全5本のPがvalidation floor合格の場合のみ、P/I01を固定
 F. 固定3操作とchildを準備し、独立query・fresh-processでRG3再判定
    ↓
STOP。合格してもREC-005は別の明示指示。
```

R3-011／012、B-C006、Task Inferenceは本タスクでは解除しない。relation不足、他CoreのSHIFT、legacy streamのN membership衝突は別blockである。

---

## 1. 根拠と、今回新たに設計した条件

### 1.1 読む資料

- **S1：ADR-0098実行報告。** [原文](research/evidence/REC004C_ADR0098_SUPPLIED.md)。5初期化の結果、実位置写像、保存artifactの根拠。
- **S2：REC-004C指示書。** [snapshot](research/evidence/REC004C_TASK_SUPPLIED.md)。初期化・訓練stream・診断suite・freeze契約を継承する。
- **S3：ADR-0097／REC-004B指示。** [結果](research/evidence/REC004B_ADR0097_SUPPLIED.md)、[指示](research/evidence/REC004B_TASK_SUPPLIED.md)。固定3候補、親bundle、独立queryとassembly規則。
- **S4：復旧受入条件・ModelBundle契約。** [受入条件](research/evidence/RECOVERY_ACCEPTANCE_SUPPLIED.md)、[設計snapshot](research/evidence/MODEL_BUNDLE_CONTRACT_SUPPLIED.md)。実装時は現repoとのrevision差を確認する。
- **今回の実装設計：** [B2_LENGTH_CONDITIONED_POSITION_BIAS_V1.md](design-docs/B2_LENGTH_CONDITIONED_POSITION_BIAS_V1.md)。API補足・報告／推論の区別は[source notes](research/B2_MIRROR_POSITION_BIAS_SOURCE_NOTES.md)。

本パック作成時には実repoのコード・checkpoint・JSONを独立実行していない。資料記載pathは確認対象であり、存在確認の代わりではない。実API・physical ID・未記載のoptimizer値を推測しない。

### 1.2 報告が直接支持すること

S1によれば、同じCoreと同じ6000-step訓練streamで、I01〜I05の終端validation EMは約0.2041／0.3135／0.4346／0.4814／0.5029。全て0.95未満で、5本から候補は選ばれていない。初期重み以外を固定した範囲で初期化感度が観測された。[S1 Evidence 6–7]

`MirrorHalvesOp.apply`は `mid=n//2; reversed(seq[:mid]) + reversed(seq[mid:])`。出力←入力の位置写像は、`i<mid`なら `mid-1-i`、それ以外は `n+mid-1-i` と報告され、oracle interpreterとの一致が確認されている。[S1 Evidence 2]

padding／batch条件、hook非干渉、Core／他15操作のfreezeは、S1で検査された範囲では合格している。あらゆる実装バグの不存在証明とはしない。[S1 Evidence 3, 5, 10]

### 1.3 報告内の矛盾を明示して扱う

S1 Evidence 4は「fixed位置は奇数長だけ、長さ7/9だけが寄与」と記載する。しかし、**S1自身の位置写像から計算すると、長さ6/10にも固定位置が2つある**。これは資料の書換えではなく、今回の契約監査で解消する相違である。誤りが文章だけか、診断コードか、限定subsetの説明不足かは、現時点で未確認。

原因として「容量限界」「attention span不足」「位置スコアが必須」を確定事項にしない。追加位置スコアの成功も未実証である。

### 1.4 今回の新設条件

| 項目 | このタスクでの新設・継承 |
|---|---|
| 新機構 | 汎用座標4入力→ReLU隠れ32→スカラー、追加192 parameters。head間共有 |
| 入力 | 内容から得る実系列長n・出力位置i・入力位置jだけ。正解写像は禁止 |
| 初期化 | 共通部分はREC-004CのI01〜I05。新bias部分は5本共通の事前固定state、初期出力0 |
| 比較 | U=現行operator、P=同じoperator＋位置スコア。5ペアを今回のrunnerで新規実行 |
| 新規更新 | 各条件・各init 6000、**最大60000 optimizer updates**。MIRROR_HALVESだけ |
| 主要時点 | 6000終端のみ。0/500/…/6000を計測するが途中最高値を採用しない |
| 暫定採用条件 | 正常比較＋Pの全5本で既存validation EM>=0.95。平均だけの合格不可 |
| 採用artifact | 条件成立時も**P/I01の6000-stepを固定採用**。最高値のinitを選ばない |
| 固定3候補 | CYCLE_FOUR／ROTATE_TRIPLETS／SWAP_ENDSはREC-004Aの4000-stepを維持 |
| 最終確認 | 新しい独立queryを各16操作1024例。非SHIFT15操作の各EM>=0.95を継承 |

全5初期化でのvalidation合格条件、192-parameter案、P/I01固定採用は**今回の新規設計**。既存資料に確定していたと説明しない。既定RG3の数値floor、SHIFTだけの例外、reference資格との区別は変更しない。

---

## 2. 変更範囲と禁止事項

### 2.1 許可する変更

1. 診断集計・文章の誤りを、保存済み予測と実写像に基づいて是正する。旧JSON／ADRは保存し、新revisionを作る。
2. MIRROR_HALVESのcandidateに限り、[設計書](design-docs/B2_LENGTH_CONDITIONED_POSITION_BIAS_V1.md)の加算位置スコアを導入する。
3. 必要なら実content length／positionを伝える後方互換の最小layout引数と、新operator typeをstrictに再構築するfactory／manifest対応を追加する。Coreの計算や旧operatorの既定動作を変更しない。
4. 新しい隔離candidateを学習し、条件成立時のみ新childにMIRROR_HALVES＋既定3候補を組み込む。

**これは既存operator本体の大型化ではないが、192個の学習parameterを追加する構造変更である。** パラメータ数不変の実験とは呼ばない。新構造の明示許可は本タスクだけに限定する。

### 2.2 Freezeの数え方

| 区間 | 更新可能 | 不変 |
|---|---|---|
| A/B監査・CPU診断 | metricコード／新candidateの試験用状態 | 元artifactすべて |
| U学習 | 隔離MIRROR_HALVESの従来parameters | Core・他15操作・parent全16slot原本・router／scorer等 |
| P学習 | 隔離MIRROR_HALVESの従来parameters＋192 bias parameters | Uと同じ保護対象 |
| 最終assembly | childの4slotだけversion置換。MIRRORの型・architecture署名も更新 | 親の残り12操作と共通componentの重み・formula |

保護12はSELECT、COUNT、BIND、COPY、NEGATE、SWAP_PAIRS、INVERT_HALF、REVERSE、SORT、ALTERNATING_NEGATE、INCREMENT_MOD、SHIFT。実IDはmanifestから取る。親のMIRROR slotも原本はread-only。

### 2.3 禁止

Core再学習、query/key/value/FFNの幅・層数・head数増加、別LR／loss／batch／optimizer／精度、curriculum、追加init、6000超の学習、router再較正、verifier変更、position-map教師による補助loss、oracle gather、REVERSE＋SHIFTへのruntime置換、機能floor緩和は禁止。

新biasに `mid=n//2`、半区間ID、正解位置 `pi_n(i)`、正解positionとの距離、同じhalfかのhard mask、操作専用lookup tableを入力・埋込みしない。教師は従来のtoken出力学習にのみ使う。

---

## 3. 段階A — Sourceとmetricを確定する（本格学習なし）

### A1. 入力artifact

次は報告された確認対象。実際の完全path・hashをconfigへ保存し、latest探索や別seedへのfallbackをしない。

```text
runs/phase_b_b2_model_bundle_recovery/bundles/
  2356543740ce566640f72767e73bd83955bc27bb825bb06c8fcffab03cf53995/manifest.json
runs/phase_b_b2_model_bundle_recovery/rec004c/run_001/
  source_manifest.json / operation_contract_audit.json / initialization_protocol.json
  initial_states/I01.pt ... I05.pt
  I01/ ... I05/  # checkpoint・完全training state・学習曲線
  position_confusion.json / position_error_summary.json / diagnostic_data_manifest.json
runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/checkpoints/
  # 固定3操作の4000-stepを実indexから特定
runs/phase_b_b2_model_bundle_recovery/rec004b/run_001/fixed_candidate_manifest.json
```

parent／Core／decoder／schema、初期state5個、旧validation、完全レシピ、固定3候補を確認する。初期state欠落を同じseedで作り直して同一とみなさない。欠落なら `SOURCE_ARTIFACT_UNAVAILABLE`。

旧数値の再現は元JSONの整数分子／分母を正にする。0.4717等の丸めから架空の高精度値を作らない。I01〜I05終端と3固定候補を同じ旧validationで再生し、source値と照合する。

### A2. fixed／movedの是正

評価専用に、source interpreterとの一致を検査した `pi_n(i)` を使って、

```text
fixed(i,n) = valid_output(i,n) AND (pi_n(i) == i)
moved(i,n) = valid_output(i,n) AND (pi_n(i) != i)
changed(i) = valid_output(i,n) AND (x[i] != y[i])
```

を別々に定義する。位置番号はcontent中の0始まり。padding／特殊tokenを分母へ含めない。

| n | piの固定位置 | 固定数 |
|---|---|---:|
| 6 | 1,4 | 2 |
| 7 | 1 | 1 |
| 8 | なし | 0 |
| 9 | 6 | 1 |
| 10 | 2,7 | 2 |

この表はS1の式からの導出であり、観測結果ではない。長さ2〜12の合法source fixtureで位置写像・全単射・自己逆・mask分割を検査する。neural語彙へ未登録位置ラベルを入れない。

- 文章だけ誤り：`REPORT_ONLY_ERRATUM`として新ADRへ訂正を追記。
- 診断集計が誤り：評価コードのみ修正。旧checkpointの同じ入力・同じ予測から再集計し、新metric versionで保存。
- 限定subsetだった：全位置指標とsubset指標を別名・別分母で記録。
- maskが訓練loss／runtimeにも影響していた：`TRAINING_OR_EXECUTION_CONTRACT_FAILURE`で停止。バグ修正と新bias比較を混ぜない。

fixed数0はaccuracy=null・分母0。通常sequence EMとvalid-token accuracyは訂正前後の予測不変で一致することを確認する。追加学習は不要。旧moved-vs-fixed gapの数値・符号が維持されるかを結果から報告し、維持されると仮定しない。

### A3. 位置・長さが現在どこにあるか

実際の `CrossPositionPrimitive`、`cross_attn`、factory、content mask、decoder入力、旧診断を読む。次を `position_access_audit.json` へ保存する。

- query行と出力content位置、key列と入力content位置の対応。
- 実length nの取得元と、padding幅／batch最大長との違い。
- 既存のposition特徴、length特徴、score項、mask、projection／residual／readout。
- 追加biasをscaled scoreのsoftmax前へ加える最小経路。
- 既存機構と新項が重複する場合、その具体的な差分。

長さ情報が無いと事前に断定しない。既存状態に含まれていても、新しい経路で直接利用しやすくする実験として扱える。実query/keyがcontent位置に対応せず設計を実装できないなら `POSITION_LAYOUT_CONTRACT_UNRESOLVED`で停止し、oracle対応を使って補わない。

`REVERSE + length-dependent SHIFT`の分解可能性は、既存compositionが長さ依存引数を合法的に扱えるかというread-only確認だけに留める。レシピの追加やruntime差替え、別composition性能実験をここへ混ぜない。

### Aの出口

`source_audit.json`、`fixed_position_audit.json`、`metric_erratum.json`、`corrected_historical_position_metrics.json`、`position_access_audit.json`を保存。source不整合・teacher不一致・実行契約違反があれば新規学習STOP。正しいmetricに直したこと自体はprimitive修正成功ではない。

---

## 4. 段階B — 新しい位置スコアと互換性を実装する

詳細は[設計書](design-docs/B2_LENGTH_CONDITIONED_POSITION_BIAS_V1.md)を正とする。構造は一案だけで、結果を見てhidden幅や特徴を変えない。

### B1. 位置項

有効content位置で、`d=max(n-1,1)`とし、

```text
phi(i,j,n) = [ i/d, j/d, (j-i)/d, n/L_ref ]
b_theta(i,j,n) = w2^T ReLU(W1 phi(i,j,n) + a1)
score_new = score_existing + b_theta
```

- hidden width=32、出力biasなし、head間で共有。parameters=128+32+32=**192**。
- `L_ref`はmodelの合法最大content長の固定値を実schemaから取得し、学習前に記録。batch幅やvalidationの最大長で変えない。
- 使用するのは汎用座標だけ。teacher、正解写像、半分IDはfeatureへ入れない。
- 従来parametersと192個を同じ既定optimizer recipeで学習する。既存モデルを凍結してbiasだけ学習する別実験にはしない。
- 追加数は実数で検査し、192個かつ元primitiveの5%以下を小型条件とする。満たさなければ `POSITION_BIAS_BUDGET_MISMATCH`。

### B2. 初期は元の関数と等価にする

5本共通の新bias初期stateを一度作り、保存する。隠れ層は固定namespaceの局所RNGで初期化し、出力重みw2だけを0とする。元I01〜I05の全tensorは同じ値でU/Pへloadする。元stateをnew factoryの乱数消費に依存させない。

初期bias=0で、UとPのlogits／有効出力／共通parameterの勾配が既定許容差内で一致することをCPUと本番deviceの小fixtureで検査する。新branchの勾配が流れることも検査する。両層全ゼロや `detach()` により学習不能な項を作らない。

### B3. Mask／kernelと副作用

既存APIでfloat additive biasを渡せる経路を優先する。MHAとSDPAのbool maskの意味は同じと仮定しない。paddingへの禁止は維持し、biasで禁止keyを復活させない。異なる実lengthを持つ例で同じbias matrixを誤共有しない。[W1/W2はsource notes]

API経路の変更に伴うzero-bias差を確認し、機能差が残るなら本格比較を始めない。近似的な浮動小数点差は実数・許容差・backendを記録し、数値計算経路まで完全同一と主張しない。許容差をFAIL後に拡大しない。

### B4. 保存・再構築を先に通す

新operatorは明示のarchitecture type／feature version／192個のstateを保存し、fresh processでその型からstrictに構築する。**旧クラスへstrict=Falseでloadし、biasを落として通すことは禁止。** 旧bundleの型・default・資格は変更しない。

診断candidateにもarchitecture署名を付ける。軽量なCPU artifactでnew-type round-tripと、bias tensor欠落／未知type／layout version不一致の拒否を確認する。新型のloadには対応させるが、名目上の全機能資格を先に与えない。

出力：`architecture_spec.json`、`zero_bias_parity.json`、`gradient_path_audit.json`、`mask_layout_tests.json`、`operator_serialization_contract.json`。これらが成立してからprotocolをlockする。

---

## 5. 段階C — 5ペアのprotocolを事前固定する

### C1. ペアと予算

```text
I01: U/I01 6000 updates  vs  P/I01 6000 updates
I02: U/I02 6000 updates  vs  P/I02 6000 updates
I03: U/I03 6000 updates  vs  P/I03 6000 updates
I04: U/I04 6000 updates  vs  P/I04 6000 updates
I05: U/I05 6000 updates  vs  P/I05 6000 updates
```

Uも今回のrunnerで新規実行し、保存済みREC-004C終端を代用品にしない。共通初期stateは同一のI01〜I05。Pの新bias初期stateは全5本で同じ。10本にまたがるtraining RNG・sample列・batch順は共通にし、condition/initをdata seedへ混ぜない。

recipeはREC-004Cと同じAdamW、lr=0.0008、eta_min=0.00001、`CosineAnnealingLR(T_max=1000)`、6000 optimizer updates。loss、weight decay、batch、dtype等の残りは実configから継承する。scheduleの再上昇を含め、そのまま固定する。

更新上限はMIRROR_HALVESの10本合計60000。成功・未達にかかわらず終端まで走らせる。途中の最高値採用、成功initだけの停止、別init追加、weight-only restartはしない。保存は0/500/…/6000のweightsと完全training state。

Uが旧REC-004Cを再現しない場合、data／RNG／initial weights／optimizer／mode／backendを照合する。丸め値で厳密一致を要求しない。未説明の差が比較の唯一介入を崩すなら `CONTROL_REPLAY_MISMATCH`で停止する。微小な数値非決定性をどこまで許すかはBで事前固定し、旧値を新しい値で上書きしない。

### C2. データ役割

| role | 内容 | 用途 |
|---|---|---|
| train | REC-004Cの6000-step生成規則・同じ分布 | 従来token lossの勾配だけ |
| train_fit | 実露出済みの固定例、最大1024 | 診断のみ |
| existing_validation | 旧固定1024例 | 全5本の終端採用条件。既に適応的利用済みのdevelopment資料 |
| diagnostic_suites | REC-004Cのlength-balanced／位置識別／counterfactual | 是正metricで評価。主EMと合算しない |
| rec004d_recheck_query | 新namespace、全16操作各1024例 | 候補固定後、一度だけ正式確認 |
| reference／B2 sealed | 既定の別研究契約 | 本比較には使わない |

新queryの分布・generator version・role seed・sample hashは学習前に登録する。query生成担当だけが教師を保持し、訓練・selection APIはその出力へアクセスしない。新queryは旧未実施queryの未露出を推測して流用しない。

trainと評価の重複はsample IDと `(task, arguments, content)` digestで監査する。query側の決定的な重複処理を出力観測前に固定し、訓練streamは比較中に変更しない。Core等の過去学習で露出不明の部分はUNKNOWNと記載する。

旧sealed0〜4／20〜24、新sealed30〜34その他現repoの封印済みdataを読込・学習・選択・評価へ利用しない。model seed=10とdata seedを混同しない。

### C3. Lockするもの

`position_bias_protocol.json`へ次を固定する：source hashes、metric v2、architecture、layout／L_ref、bias初期state、I01〜I05、共通training RNG、LR trace、全optimizer引数、データhash、実行順、計測点、上限、終端条件、I01採用規則、固定3候補、query仕様、qualification scope、比較許容差。

新しい192-parameter構造と、REC-004Cの診断専用状態から今回の条件付き採用へ移ることを、実行前の新ADRへ記録する。次ADR番号は実repoの索引から取得し、0099等を推測して予約しない。

---

## 6. 段階D — 学習・比較・限定ablation

### D1. 記録

毎updateの実LR、更新回数、入力digest、loss、実optimizer対象を保存する。500更新ごとにtrain-fit／validationのEM・token accuracy・loss、length×position別分子／分母、正しいfixed/moved/changed集計を保存する。

weightsだけでなくoptimizer、scheduler、AMP使用時scaler、CPU／CUDA RNG、data cursor、protocol hashを保存する。外部中断からは同一protocolの完全resumeだけを許す。

NaN等は該当trialを失敗として保存する。独立trialの数値失敗なら残り予定trialは隔離されたまま続行できるが、失敗trialを交換しない。source／freeze／stream契約違反は全依存学習を停止する。

### D2. 主要比較

終端6000について、各initのU/Pのsequence EM、P−U、Uのみ正解／Pのみ正解の件数（`U_only_correct`／`P_only_correct`）、length別の差を示す。mean、sample SD、min、max、range、0.95到達数を分母付きで集計する。

全試行を同じ重みで扱う。長さ別平均を通常validation全体EMへ置き換えない。failed/missingを0やPASSへ埋めず、予定5ペアに対する完了数を明示する。データ例数が多くても独立Coreが5個あることにはならない。

- Pが全ペアで改善：`PAIRED_GAIN_ALL_FIVE_OBSERVED`。
- 改善・悪化が混在：`MIXED_PAIRED_EFFECT`。
- Pが全ペアで改善しない：観測値をそのまま報告する。
- Pの5/5終端EM>=0.95：この旧validation上の `FIVE_INIT_VALIDATION_FLOOR_PASS`。

改善ラベルと採用条件は別。全ペアで改善しても0.95未達なら候補なし。検定p値や母集団成功率の保証は必須とせず、同一Core・同一データの小規模比較であることを明記する。

### D3. 追加biasへの依存性を確認する

全5本のP終端について、同じdiagnostic validationで**学習済みbiasだけを0にしたforward**を追加評価する。再学習・更新・checkpoint変更はしない。元biasを戻すと元予測へ戻ることをテストする。

これは学習済みPがbiasを利用するかの介入であり、Uと同じ重みへ戻す操作ではない。zero-biasで劣化しても「元の失敗原因は位置情報だけだった」と断定しない。attention argmaxと正解写像一致は補助指標であり、合格条件にはしない。

Correct／None／Wrong-familyの既存対照、padding／batch不変性、counterfactual依存、入出力位置の曖昧性は継続して記録する。正解写像へattentionを強制するloss・maskは追加しない。

### D4. 計算量・主張の範囲

UとPのresident／active parameters、追加bias計算のpair数、FLOPs推定、学習費用、warmup後のmedian/p95 forward latency、VRAM、選択外primitive callsを別記する。parameterが192増えただけだから無視できると仮定しない。biasは通常n×nの位置対について計算する。

本比較は「汎用座標への明示経路＋小型parameter追加」という複合した構造介入であり、同数parameterの非位置対照を実施していない。純粋なparameter数効果との完全分離、未知長への外挿、他操作への一般化、自律的architecture discoveryは主張しない。

---

## 7. 段階E — 候補固定とassembly

### E1. 採用条件

次の全条件が必要。

```text
metric/source/layout/serialization/zero-bias契約がPASS
U/Pの全5ペアが6000まで正常完了し、paired comparisonが成立
全P/I01..I05の終端existing_validation EM >= 0.95
固定3候補のsource・4000-step・Core依存が確認済み
過大capability付与やquery利用による選択がない
```

いずれか未達なら `POSITION_BIAS_VALIDATION_NOT_MET` または対応するBLOCKED。**selected_init=null、child_bundle=null、rg3_recheck=NOT_EXECUTED**で停止する。4/5成功でも最良initを選ばない。単一原因が確定しなくても、有効な比較結果は保存する。

全条件成立時だけ **P/I01、step=6000** を採用する。この指定は結果前に固定され、I01が最低／最高のどちらでも変えない。Uや旧REC-004A終端へのfallback、平均weight、ensemble、再学習はしない。

今回の5/5条件はdevelopment上の安定性確認であり、全5本が独立queryでも合格したと主張しない。正式queryで測る修正版はP/I01から作るchild一つだけ。

### E2. 固定3候補とchild

| slot | 採用artifact |
|---|---|
| MIRROR_HALVES | P/I01の6000-step。新architecture＋共通部＋192 bias重み |
| CYCLE_FOUR | REC-004Aの4000-step snapshot |
| ROTATE_TRIPLETS | REC-004Aの4000-step snapshot |
| SWAP_ENDS | REC-004Aの4000-step snapshot |
| 他12操作・Core等 | 元REC-004の完全に同じartifact |

新しいchildの4slotだけを置換し、physical ID数16とrouter key mappingを維持する。新しいMIRROR型をmanifestへ登録し、architecture signature／state ABI／feature version／L_ref／Core・decoder依存を保存する。旧generic classへbiasを黙って追加しない。

変更bankに依存するrecipe、intermediate、reference certificate、function qualificationは無効化する。旧証拠は保存するが新bankへ丸ごとコピーしない。raw weightsと資格のhashを循環参照させない。

router/scorerの数値・formulaは不変。新bankに対する互換性を証明できず、再較正が必要なら `DEPENDENCY_REQUALIFICATION_REQUIRED`で停止する。本タスクで再較正はしない。

childは新IDのstaging／diagnostic scopeに留める。MIRRORを取り込んだ同型standalone予測と、bank実行予測が一致することを確認する。これをloaderが対応できなければ `SERIALIZATION_OR_RUNTIME_BLOCKED`で止める。

---

## 8. 段階F — 新queryとfresh processによるRG3再判定

候補・architecture・固定3・child manifest・query hashを固定してから、全16操作の `rec004d_recheck_query` 各1024例を一度だけ評価する。未選択initやUを最終queryで競わせない。親は同じ入力の対照として評価する。

### F1. 全16操作の機能確認

- raw direct/oracle-call sequence EM、token accuracy、loss、既存causal controlsを表示。
- **非SHIFT15操作ごとにEM>=0.95**。平均・token accuracy・fixed位置精度で代替しない。
- SHIFTも数値を表示。既存のSHIFT限定例外を維持し、他操作へ拡張しない。
- 親から不変の12操作は、state／formula一致に加え同じqueryで予測一致を確認。
- 固定3候補とMIRRORは、それぞれ正しいsource candidateからの予測とassembly後予測を照合。
- routing／ArgumentScorerの非学習診断は別表。raw executionの不足をverifierで除外しない。

保護対象が新queryでfloor未達なら、そのままRG3 FAIL。学習していないことと機能floor合格は別。MIRRORが未達なら別initへの切替、step変更、query再生成をしない。

### F2. Fresh processと資格

別working directoryの新Pythonプロセスから、manifestを明示してloadする。新operator typeと192 bias tensorを含むstrict load、全16の離散予測一致、component hash、execution signatureを検査する。

trainer／optimizer生成／get-or-build／router較正／global cache探索をdynamic guardで禁止する。通常のmodel skeleton生成後の完全loadはよいが、欠落tensorをランダム値で補わない。原parent・旧run・固定3・共有cacheは不変。

`structural_load_pass`、`recovery_floor_pass`、`fresh_process_pass`、`nominal_reference_qualification`を別項目にする。point-estimateのRG3合格だけで統計的 `REF_ADEQUATE` やB2全機能nominal資格を付与しない。

### F3. 判定と引き継ぎ

全条件合格なら、新runとして `RG3_RECHECK_PASS`。これは**人が設計した小型heterogeneous operator変更による、seed10上の復旧pilot合格**である。元generic operatorの予算延長だけで直った、未知familyへの自律的発見が成立した、5Coreが復旧したとは言わない。

採用レシピは旧1000-step／004A/Bのmanifestを上書きせず、新architecture revisionとして固定する。REC-005へ進む前に、残りseeds11〜14へ適用する同じ固定レシピ・init導出・新型build／load契約が書けることを確認する。I01の訓練済み重みを別Coreへcopyしない。Coreごとに初期化を複数試して最良値を選ぶ方式へ変更しない。

`RG3_RECHECK_PASS`かつ未解決のrecipe／dependency blockがなければREC-005を別指示で実行可能と報告できる。自動実行はしない。R3-011／012、B-C006、Task Inferenceは引き続きblocked。

---

## 9. ファイル配置・テスト・成果物

### 9.1 推奨配置

以下の新規名は提案。既存の同等abstractionを優先し、原defaultを変えない。

```text
docs/CODEX_TASKS_PHASE_B_B2_MIRROR_LENGTH_POSITION_BIAS.md
docs/design-docs/B2_LENGTH_CONDITIONED_POSITION_BIAS_V1.md
src/apc/primitives/length_conditioned_position_bias.py        # 必要なら新規
src/apc/evaluation/mirror_position_initialization_diagnostic.py # metric訂正だけ
src/apc/evaluation/mirror_position_bias_repair.py             # 一task orchestration
src/apc/utils/model_bundle.py                                # 新型のstrict contractだけ
scripts/run_phase_b_b2_model_bundle_recovery.py               # REC-004D dispatch
scripts/rec004_fresh_process_check.py                         # 型に応じる既存入口
tests/test_mirror_position_metric_contract.py
tests/test_length_conditioned_position_bias.py
tests/test_mirror_position_bias_repair.py
configs/phase_b_b2_model_bundle_recovery_rec004d.yaml
runs/phase_b_b2_model_bundle_recovery/rec004d/<run_id>/
```

実装後の推奨CLI形（現在の実在commandとは断定しない）：

```bash
python scripts/run_phase_b_b2_model_bundle_recovery.py --task B-C005REC-004D --config configs/phase_b_b2_model_bundle_recovery_rec004d.yaml
```

既存CLIに従って新run-dirを使い、`--all`、latest探索、自動次taskを追加しない。root AGENTS、復旧plan/task、DECISIONS indexと現在の復旧ADR logへ分岐・状態を追記する。S1の報告では現在のlogは `docs/DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`。実repoを確認して次番号を使う。

### 9.2 必須テスト

1. source写像とfixed mask：n=6/10の固定2個、n=8の分母0、fixed+moved=valid、changedとの区別。
2. 集計訂正で旧予測／sequence EMが変わらない。旧JSON／ADRの保存。
3. position featureはi,j,n,L_refのみで、内容値・teacher・pi・half labelへアクセスしない。
4. 長さの異なるbatch、padding幅変更、特殊tokenがある合法layoutで位置対応が正しい。
5. 初期bias=0のforward／logit／共通勾配等価性。新branchに勾配が流れ、key方向に異なるscoreを生成できる。
6. softmax前の加算位置、既存scale・mask・dtypeの維持。禁止paddingへのmassを作らない。
7. I01〜I05の共通部一致、bias初期state共通、init生成のRNG消費が訓練streamへ影響しない。
8. optimizer対象がUは対象本体、Pは対象本体＋192のみ。parent原本／他15／router／Core不変。
9. 全10本の同一sample列、6000上限、途中eval非干渉、完全resume再現、weight-only拒否。
10. Pが4/5合格なら採用なし。5/5合格でもI01固定。最終query後のinit切替が拒否される。
11. bias-zero介入が重みを変更せず、通常forwardへ戻すと元予測が復元する。
12. 新型stateのfresh-process round-trip。旧型＋新重み／bias欠落／未知architectureを拒否。
13. childは4slotだけ変更し、残り12不変。fixed3とselected MIRRORのsource予測を照合。
14. evaluation／load中のtraining・fallbackをdynamic guardで検出。capability過大認証・旧certificateを拒否。
15. NaN／未実行／空分母を結果から消さない。source不整合時は学習せず、旧artifactを上書きしない。

CPU tiny fixtureとGPU学習結果を別証拠にする。focused tests後にrepo標準検証：

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

現repoの新canonical commandがあれば従う。他ジョブを無断停止しない。旧2076件PASSを新commitへ転記しない。未実施・resource不足は明記する。

### 9.3 成果物

```text
config.yaml / protocol.json / system.json / summary.json / report.md
source_audit.json / fixed_position_audit.json / metric_erratum.json
corrected_historical_position_metrics.json / position_access_audit.json
architecture_spec.json / zero_bias_parity.json / mask_layout_tests.json
gradient_path_audit.json / operator_serialization_contract.json
position_bias_protocol.json / data_manifest.json / shared_bias_initial_state.pt
U/I01..I05/ / P/I01..I05/  # 曲線・LR/data trace・weights・完全training states
paired_comparison.json / per_length_position_metrics.json / bias_ablation.json
initialization_stability.json / cost_accounting.json / freeze_audit.json
side_effect_audit.json / candidate_decision.json
# 採用条件成立時だけ：
selected_recipe.json / recipe_revision.json / bundle_lineage.json
all_primitive_execution.json / protected_regression.json
rg3_recheck.json / fresh_process_report.json / qualification.json
```

### 9.4 完了報告

```text
Task: B-C005REC-004D
Implementation / source / metric correction / paired experiment status:
Source-derived facts / erratum / new decisions / deviations:
Core / five base-init hashes / shared bias-init hash / architecture signature:
Fixed-position audit: prose-only / metric-fixed / subset / blocked:
Added parameters / valid-length source / padding and zero-bias parity:
Init | U EM | P EM | P-U | P train-fit | per-length EM | status
Planned/completed/diverged/missing; P>=0.95 count; paired-effect label:
Bias-zero ablation / attention and counterfactual limits:
Candidate: null or P/I01@6000; rationale from fixed rule:
Independent RG3 query / source4 assembly / protected12:
Fresh-process / strict new-type load / qualification:
Updates/examples / FLOPs-estimate scope / latency / VRAM:
Tests / actual commands / artifact paths / ADR:
RG3_RECHECK: PASS / FAIL / UNRESOLVED / NOT_EXECUTED
REC-005 eligible: true/false with reasons; executed: false
R3-011/012 / B-C006 / Task Inference: blocked
```

**STOP：この一件が終了したら結果を報告する。未達でも学習延長・別bias設計・別init採用を自動実装しない。**
