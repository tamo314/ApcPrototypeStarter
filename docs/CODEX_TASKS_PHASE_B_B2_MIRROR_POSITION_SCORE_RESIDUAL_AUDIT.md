# AI Coding Task — MIRROR_HALVES Position-Score Residual Audit & Next-Repair Contract

**正式ID：B-C005REC-004E**  
**版：1.0 / 2026-09-09 / 状態：新規指示。プロジェクトでの実装・実行は未実施。**  
**位置：REC-004Dの候補未達後、REC-005の前。保存済みモデルを用いる診断・修正仕様策定の一件。**

## 0. このタスクで行うこと・行わないこと

REC-004Dの学習済みP（192-parameterの位置bias追加版）について、位置biasそのもの、既存attention scoreとの組合せ、終盤の学習曲線を調べる。**次に変更すべき一箇所を、最大一案の修正契約として出力するところまで**を本タスクとする。

本タスクでは、新しいprimitiveの学習、重みの更新、位置feature／L_ref／temperatureの本番変更、候補採用、child assembly、RG3再判定を行わない。後述のforward介入は、隔離された評価専用copy上の診断だけであり、採用候補ではない。

```text
REC-004D：全5ペアで改善、Pは0/5合格、RG3 NOT_EXECUTED（保存）
    ↓
REC-004E
 A. Source replay・計測経路・データ／予算を固定
 B. 全位置対のbiasとReLU状態を列挙
 C. 実score分解・固定されたforward介入・終盤曲線を照合
 D. 診断結果と、一案までの次修正契約を出力
    ↓
STOP。次修正の実装・学習は別の明示指示。
```

ユーザーが本タスクを指定した場合、A〜Dは前提とSTOP条件を守って一件として実行できる。段階ごとの確認は不要。**新規optimizer updates=0**。REC-005、R3-011／012、B-C006、Task Inferenceは引き続きblocked。

### 0.1 なぜここで修正方式を固定しないか

添付結果は「192-parameterの座標経路が改善した」ことを支持するが、残差がfeature値の不整合、学習されたbias形状、既存scoreとの競合、出力側の計算、未収束のどれによるものかまでは示していない。`L_ref`の実値と生のscoreも本文にはない。根拠のないfeature変更を先に実装せず、保存済みartifactからこれらを区別する。

この一件は広範な再診断ではない。Core・他操作・router・verifierの調査を再開せず、**位置スコアの残差**に限定する。

---

## 1. 根拠・継承・新設条件

### 1.1 読む資料

- **S1：ADR-0099実行報告。** [原文snapshot](research/evidence/REC004D_ADR0099_SUPPLIED.md)。今回の開始点。
- **S2：REC-004D指示書。** 現repoの `docs/CODEX_TASKS_PHASE_B_B2_MIRROR_LENGTH_POSITION_BIAS.md`。[配布時snapshot](research/evidence/REC004D_TASK_SUPPLIED.md)。source、5初期化、freeze、旧採用条件を確認する。
- **S3：位置bias v1設計。** 現repoの `docs/design-docs/B2_LENGTH_CONDITIONED_POSITION_BIAS_V1.md`。[配布時snapshot](research/evidence/REC004D_POSITION_BIAS_V1_SUPPLIED.md)。現実装とのrevision差を確認する。
- root `AGENTS.md`、復旧task／plan／ModelBundle契約、REC-004Dの実config・protocol・architecture spec・data manifest。
- [Source notes](research/B2_MIRROR_POSITION_SCORE_RESIDUAL_SOURCE_NOTES.md)：報告・計算による整理・今回の新規設計の区別。

資料の数値は実行担当者の報告であり、本指示書作成時に実repoのcheckpointやJSONを独立実行した値ではない。path・API・物理ID・optimizer値・L_refは実装時に確認し、不明を推測で補わない。

### 1.2 引き継ぐ観測とその限界

S1によれば、Pの終端validation EMはI01〜I05で約0.6768／0.7070／0.5713／0.7373／0.9023、平均0.7189。全5本でUより改善したが0.95到達は0/5。candidate・child・独立query再判定は存在しない。[S1 Evidence 6–7, Consequences]

位置biasを0にした学習済みPはEM=0〜0.007へ崩れ、元biasを戻すと予測が復元した。これはPがbiasを利用している証拠であり、Uの失敗原因が位置情報だけだったことの証明ではない。[S1 Evidence 9]

S1は長さ10を主な残差とし、長さ6〜9を大きく改善した領域と説明する。一方、同じ表ではP/I03の長さ7／8／9は0.85／0.57／0.39、P/I02の長さ9は0.53である。**報告の強調点を保持しつつ、7〜9を解決済みとは扱わない。主な注目対象は10、全評価範囲は6〜10。**[S1 Evidence 6, 8]

fixed位置に関する旧ADR-0098の矛盾は `REPORT_ONLY_ERRATUM` として解消済み。JSON値は一致しており、本タスクで同じ問題を再調査する必要はない。正しいmetricを継承し、その版だけ検査する。[S1 Evidence 1]

### 1.3 今回の新設条件

| 項目 | 条件 |
|---|---|
| 学習 | **0 updates**。新初期化・学習延長・probe学習もなし |
| 中心モデル | REC-004DのP/I01〜I05、終端6000。元Uは必要な対照のみ |
| bias全列挙 | 合法なn=6〜10、全i,j。1モデルあたり330位置対 |
| 曲線 | 保存された0／500／…／6000の13時点×5 Pモデル |
| 実score観測 | 既存length-balanced suiteから各長さ32例、計160例×5モデル |
| 介入評価 | 同じ既存length-balanced suiteの各長さ256例、計1280例×5モデル |
| forward介入 | §5で固定するscore倍率・score成分除去・長さfeature置換だけ |
| 新しい最終query | **生成・評価しない**。RG3／B2 sealedを消費しない |
| 完了条件 | 計測の妥当性、raw根拠、未確定事項、次の一案が明示されること |
| 性能合格条件 | 本診断に新しいEM Gateは設けない。旧0.95／5本条件は変更しない |

330は `6²+7²+8²+9²+10²` の算術上の位置対数であり、330個の独立な学習試行ではない。

---

## 2. 不変条件・許可範囲

読み取り専用で保存する対象：parent bundle全16slot、固定Core／task側／decoder、REC-004Cの初期state、REC-004Dの全U/P checkpoint／training state、新bias初期state、固定3候補、router／keys／ArgumentScorer／controller／verifier、全共有cache。

**学習対象は一つもない。** Pの評価copyに対し、§5の指定したscoreだけを一時的に変えてforwardすることを許可する。`state_dict`、buffer、formulaの元ファイルを書き換えず、介入値をcheckpoint・bank・manifestへ保存しない。

評価copyの学習modeは固定eval。lossを測るためのtarget参照は評価器だけに限定する。勾配計算・optimizer生成・training API呼出し・新規fitを禁止する。teacherと位置写像は指標計算にのみ用い、scoreやforwardの入力へ渡さない。

禁止するもの：

- 長さ10専用のruntime分岐、正解位置lookup、half ID、`pi_n(i)`を使うbias、oracle gather。
- 新しい座標、L_ref、hidden幅、head別bias、学習可能なgateの導入・試行。
- Core再学習、operator容量増加、LR／loss／訓練分布変更、未見長の訓練。
- 介入でよかったalphaや入力改変を、そのまま採用candidateとすること。
- 全5本の平均重み、最良init選択、途中最高checkpointの採用。
- 過去runの上書き、shared cache削除、latest探索、別seedへのfallback。

診断コード自身のバグはsourceモデルを変えずに修正・再集計できる。**本番feature・mask・loader等の実行契約違反を発見した場合は、修正と診断を混ぜず、最小再現例を保存して依存作業を停止する。**

---

## 3. 段階A — Source replayと計測契約の固定

### A1. 正確な入力を確定する

S1が報告したrun root：

```text
runs/phase_b_b2_model_bundle_recovery/rec004d/run_001/
  source_audit.json / architecture_spec.json / position_bias_protocol.json
  data_manifest.json / shared_bias_initial_state.pt
  I01/.../U_CURRENT_OPERATOR/  および P_LENGTH_POSITION_BIAS/
    checkpoints/ / training_states/
  paired_comparison.json / per_length_position_metrics.json
  bias_ablation.json / learning_curve.jsonl
```

I01〜I05配下の完全pathは実directoryとmetadataから特定する。`U/I01`等の別表記を存在確認なしに作らない。source manifestが参照するREC-004 parent、Core、REC-004Cの診断データ、REC-004Aの固定3候補もhash台帳に記録する。

次を照合する：architecture type、feature version、L_refの実値と取得元、実content最大長、特殊token offset、全192 bias tensors、Core／decoder dependency、旧validation hash、13 checkpointのstepと学習レシピ。

`L_ref=10`や「長さ10がschemaの上限」と仮定しない。訓練範囲の上限とmodelの合法上限は別。nは実content mask由来とし、padding幅・target長・batch最大長と混同しない。

P終端5本は必須source。欠落なら `SOURCE_ARTIFACT_UNAVAILABLE` でSTOP。中間checkpointだけの欠落はその時点を `UNAVAILABLE` とし、終端診断は実行可能。旧結果を再学習して補わない。

### A2. 元の予測を再現する

Pと取得可能なUの終端を、REC-004Dと同じ旧validation1024例で再生する。元JSONの整数分子／分母・予測hashを正とし、ADRの丸め値へ無理に完全一致させない。

同じsource・同じ入力で説明できない予測差が出る場合、追加介入を始めず `SOURCE_REPLAY_MISMATCH`。装置・dtype等の既定許容差を事前記録し、後から拡大しない。

### A3. Observerが何を測っているかを確認する

現 `CrossPositionLengthBiasPrimitive.forward` と `cross_attn` を読み、query/keyのprojection、head分割、既存scale、mask、位置Embedding、softmax、value／出力経路を特定する。

観測用の式を、実forwardと対応させて登録する：

```text
S_other = 新position-bias以外の既存score項（既存scale適用後）
B       = 新position-biasの実際の加算値
M       = 既存mask（変更しない）
attention = softmax(S_other + B + M, key_dimension)
```

`S_other`をraw hの内積で代用しない。biasがscaleの前か後か、さらに別のscore項があるかを実装から確認する。softmaxの行・列対応はcontent位置のlayoutへ明示変換する。

既存hook／adapterを優先する。副計算でscoreを再構成する場合、同じq/k/v・scale・maskでattention出力と最終logitsを再現することを、CPU tiny fixtureと本番deviceの小fixtureで検査する。**通常Pと無介入observer付きPの離散予測が一致することが、score介入の前提。** 計算backendまで同一でない場合はその制限も記録する。

本番kernel・precision・attention計算を広範に置換しない。妥当な再構成ができなければ `SCORE_OBSERVATION_UNAVAILABLE` とし、B単体・出力・保存曲線の診断へ範囲を限定する。数値の似たproxyを実scoreと呼ばない。

### A4. データ・実行予算を事前ロックする

- `source_replay`：旧validation1024例。新holdoutとは呼ばない。
- `grid`：n=6〜10、各0≤i,j<n。tokenを必要としない全座標。
- `score_observation_subset`：既存length-balanced各256例のsample ID順先頭32例／長さ。出力・正誤を見ず固定。
- `intervention_suite`：同じ既存length-balanced全1280例。
- `reference/final_query/sealed`：使用禁止。

不足時は取得できるsourceと不足数を明示し、別分布で埋めたり「失敗例だけ」を採ったりしない。訓練データ・診断suiteの来歴はREC-004D/Cのmanifestから継承し、既知の露出を消さない。

`residual_audit_protocol.json`にsource／データhash、観測点、観測subset、全介入、許容差、禁止処理、missing規則を固定する。次節以降の出力を見て介入値やsample数を増やさない。

---

## 4. 段階B — 位置biasを全列挙する

### B1. 実forwardと同じfeatureを使う

S3の規則を実実装と照合する：

```text
d = max(n - 1, 1)
phi = [i/d, j/d, (j-i)/d, n/L_ref]
z   = W1 @ phi + a1
h   = ReLU(z)
b   = w2 @ h
```

全5 Pモデルの全保存時点に対し、330位置対を列挙する。全13時点が揃えば **65 grids、21450 scalar bias値**。head共有なのでBをhead数分複製して独立標本に数えない。

bias値は実moduleから取得する。別実装の式による計算は一致検査に使え、実moduleとの差があれば隠さない。実length n、phi、z、h、w2ごとの寄与、b、finite/NaN、maskで使用可能な位置の別フラグを保存する。

### B2. row-centered biasと隠れ層の活性を測る

各query行で、実maskが許す有効key集合K_iに対して、

```text
b_center[i,j] = b[i,j] - mean_{k in K_i}(b[i,k])
```

を計算する。この位置表のK_iは有効content keyに限定し、禁止keyの-inf、padding、特殊tokenを分母へ混ぜない。K_iが空ならnull＋理由。全330の座標表と、実maskで利用される集合は別に記録する。

中心化は記述用でforwardへ戻さない。attentionが有効な特殊tokenも参照する場合、content-onlyの統計と、特殊tokenを含む全許可key上の統計を区別する。softmaxの定数加算不変性は**全許可keyへ同じ定数を加える場合**の性質であり、content keyだけをずらして特殊tokenを据え置く介入の等価性を主張しない。実score分解・§5の介入では、特殊tokenを含む元の全maskを維持する。

各長さ×行×checkpointについて、raw bのrange、row-centered std／range、隠れunitの活性率、key方向のactivation patternの変化数を出す。ReLU閾値直近の数値は別欄にし、事後にepsilonを調整しない。

**補助的な数理チェック：** 固定n,iではphiはjの一次式になる。同じReLU活性集合が行内の全jで続けば、その区間のbもjの一次式になる。該当する行・区間を `ROW_AFFINE_SEGMENT_OBSERVED` と記録してよい。しかし、既存scoreも加わるため、それだけでoperator全体が目的位置を表現不能とは言えない。

「活性が0のunit数」を、そのままネットワーク全体のdead ReLU診断や容量不足の結論にしない。全長で不活性か、特定長・行だけかを区別する。

### B3. 正解位置との比較は評価専用

S1/S2で確認済みの位置写像をmetric側だけに保持する。raw biasと合算scoreに対して、正解入力位置のrank、tie-aware margin、rowのentropy、許可key数を記録できる。

これは記述指標であり、新しい訓練lossにもPASS条件にもならない。Core表現が局所tokenだけを保持すると仮定しない。attention argmaxが正解位置と違うことから、出力誤りや原因を自動確定しない。

fixed位置は n=6:{1,4}、7:{1}、8:{}、9:{6}、10:{2,7} を契約テストにする。fixed/moved/changedを区別し、分母0はnull。

### Bの成果物

`position_grid.npz`等のtyped array、`position_grid_index.json`、`activation_summary.json`、`row_centered_bias_summary.json`。配列順序、dtype、shape、content位置の向き、hashをindexに記載する。

---

## 5. 段階C — Scoreの相互作用と固定forward介入

### C1. 合算する前と後を分ける

P終端5本と `score_observation_subset` を使い、headごとのS_other、B、S_other+B、各行の許可key集合、softmax分布を取得する。

各成分のrow-centered std／range、正解位置のrank／margin、S_otherとBの行内相関、合算後の確率、最終token予測を記録する。std=0では相関をnull。B/Sの比率は分母0ならnullとし、極大値をclampして事実を隠さない。

既存位置Embedding等を含むS_otherを「純粋なcontent score」と呼ばない。head平均だけで差を消さず、head別と集約を併記する。

### C2. 診断専用の介入表（追加探索は禁止）

下記だけを、隔離された同じP終端重み・同じintervention_suiteへ適用する。実際に計測経路を検証できた条件だけ実行する。

| ID | score／biasの変更 | 対象 | 何を調べるか |
|---|---|---|---|
| J0 | S_other+B+M | 全長 | 通常P、基準 |
| J1 | S_other+0×B+M | 全長 | 旧bias-zero結果と、今回の同一入力上の対照 |
| J2 | S_other+0.5×B+M | 全長 | biasを弱めたときの応答 |
| J3 | S_other+2×B+M | 全長 | biasを強めたときの応答 |
| J4 | 0×S_other+B+M | 全長 | 既存score成分の除去。value／residual／decoderはそのまま |
| J5 | phi第4成分だけ9/L_refへ。i/d,j/d,(j-i)/dは真のn=10由来 | n=10のみ | 学習済みMLPのlength featureへの局所感度 |
| J6 | phi第4成分だけ10/L_refへ。残りは真のn=9由来 | n=9のみ | 同じ切替の逆方向 |

J5/J6は**訓練時の整合した座標組合せから外すcounterfactual**であり、有効な通常入力や修正candidateではない。L_ref自体、mask、position IDs、実length、query/output長は変更しない。n=10をn=9へ偽装してpaddingやdecoderまで変えてはいけない。

J4も、attentionのscore成分を除去する診断であり、独立したbias-onlyモデルの性能ではない。内容依存のvalue、Core、residualは残る。既存scoreに分離不能な処理があるなら `INTERVENTION_UNSUPPORTED` で十分。全面attention置換はしない。

J1/J2/J3のいずれかが改善しても、そのalphaを本番設定へ採用しない。J5/J6が改善しても、L_refが誤っていた、正規化が根本原因だったと断定しない。

通常forwardへ戻した後、J0の予測を再度確認する。parameter／buffer／mode hash不変を検査し、介入の順番が結果を変えないことをCPU fixtureで試験する。

### C3. 実行上限と出力

主suite1280例にJ0〜J4：**5モデル×5条件×1280=32000 prediction examples**。
J5/J6は各該当長256例：**5モデル×2条件×256=2560 examples**。
従って本介入matrixは最大34560例。restoreチェックやsource replayは別費用として計上する。診断対象が不足／未対応ならmissingを開示し、勝手に別介入で置き換えない。

各init×介入×長さについて、sequence EM、token accuracy、loss、n、J0とのpaired delta、J0-only正解／介入-only正解、共に誤り、長さ×出力位置の変化を保存する。

長さ10の改善と他長の低下を別々に報告する。最も良かった介入だけの集約を作らない。length-balancedのEMを旧validation EMやRG3の値へ混ぜない。0.95を超えても診断結果として記録し、candidateにはしない。

### C4. 終盤の曲線を同じLR位相でも比較する

Pの全保存曲線を読み、特に4000／4500／5000／5500／6000でtrain-fitとvalidationのEM、loss、各長さの分子／分母を並べる。length別lossがなければ、同じ保存checkpointと固定診断subsetで評価して補足し、元ログとは別欄にする。

既存T_max=1000では位相が繰り返すため、4000→6000や3000→5000など、**実lr_used／scheduler counterで確認した同位相の差**も併記する。終盤を線形補外して必要step数を予言しない。単一の終端やtrain-fitとの近さから「収束済み」「容量限界」としない。

既存保存時点以外の曲線を補間して実測扱いしない。再学習・6000超の継続は行わない。

### Cの成果物

`score_observer_parity.json`、`score_decomposition.npz`、`score_decomposition_index.json`、`intervention_results.jsonl`、`paired_intervention_summary.json`、`intervention_nonmutation_audit.json`、`late_phase_learning_audit.json`。

---

## 6. 段階D — 次の修正を一案に絞って仕様化し、STOP

### D1. 結論は強制しない

以下の表で、証拠と未確定事項を整理する。複数の観測ラベルがあってよいが、次に実装する修正案は最大一件とする。

| 観測 | 記載可能な結論 | 次の一案の方向（まだ実装しない） |
|---|---|---|
| 実feature／mask／layoutが既定契約と不一致 | `EXECUTION_CONTRACT_FAILURE` | 最小の契約修正。設計変更は混ぜない |
| 契約正常、J2/J3で再現性のある長さ別の利害が見える | `SCORE_BALANCE_SENSITIVITY_OBSERVED` | scoreの扱い一箇所の、将来の学習対照を設計 |
| J5/J6で形状・出力が変わり、gridの活性遷移と対応する | `LENGTH_FEATURE_SENSITIVITY_OBSERVED` | 同じ容量での一つの座標表現案を検討。ただし正規化原因確定ではない |
| bias-only scoreは区別できるが、合算でmargin／出力が悪化する | `SCORE_COMPONENT_INTERACTION_LEAD` | 既存scoreとの相互作用に限定した対照案 |
| row-affine／同順位の行が多く位置差を付けにくい | `POSITION_DISCRIMINATION_LEAD` | 小型位置表現の一案。ただし全operatorの表現不能を主張しない |
| 合算attentionの記述指標は良いが出力が誤る | `POSITION_SCORE_SUFFICIENCY_NOT_ESTABLISHED` | value／residual／readout等の限定確認を提案。原因確定ではない |
| 同位相の終盤で改善が続く | `OPTIMIZATION_PROGRESS_OBSERVED` | 別承認の有限予算検証を検討。自動延長しない |
| 観測が混在／不足 | `UNRESOLVED` | 最も小さい追加確認一件まで。根拠のないrepairを捏造しない |

hard mask、oracle position table、長さ10専用例外、Core再学習、大型化、複数loss探索をこの表の「修正」として紛れ込ませない。

### D2. `next_repair_contract.md`の必須内容

次に試す一案を、以下の項目で実行可能な仕様案として書く。ただし**このファイルの生成は、実装・学習の許可ではない**。

```text
status: PROPOSED_NOT_AUTHORIZED（またはEVIDENCE_INSUFFICIENT）
対象mechanismと、今回のraw artifact／metricへの参照
観測／推論／提案の明示的区別
変更する一箇所、featureまたはscoreの具体式と適用位置
修正前後で不変にするもの
追加parameter数・計算量の変化
同一親／共通init／入力列を使う対照、介入の交絡
既存5初期化を将来どう比較するか（最高値採用は禁止）
学習を要する場合の有限上限、明示的なレシピ差分
採用条件と、長さ6〜10全体を評価する指標
独立queryの扱い、旧FAIL保存、STOP条件
現APIでの変更対象と必要テスト
```

具体式や予算を資料から決められず、新しく設計する場合は「次タスクの提案値」と明記する。複数案を実験してから最良を選ぶ契約にしない。十分な証拠がなければ無理に修正案を完成扱いにせず、不足項目を明示する。

**旧REC-004Dの採用規則（P全5本の終端validation EM>=0.95、I01固定）およびRG3の非SHIFT15操作各EM>=0.95を、この診断の成績で緩和しない。** 将来の異なる修正に適用する規則は、その次の明示指示で確定する。

### D3. 完了条件と状態

本タスクは性能Gateではない。source・計測・実施範囲・raw結果・不変性・未確定事項・次の一案が明示されれば、未解決の原因があっても診断の完了を報告できる。

```text
implementation_status: COMPLETE / PARTIAL
source_replay_status: VERIFIED / BLOCKED
position_grid_status: COMPLETE / PARTIAL / NOT_EXECUTED
score_observation_status: VERIFIED / UNAVAILABLE / INVALID
intervention_status: COMPLETE / PARTIAL / NOT_EXECUTED
late_curve_status: COMPLETE / PARTIAL / UNAVAILABLE
residual_diagnosis: <evidence-backed labels + unresolved>
next_repair_contract: PROPOSED_NOT_AUTHORIZED / EVIDENCE_INSUFFICIENT
new_optimizer_updates: 0
selected_init: null
selected_intervention: null
child_bundle: null
rg3_recheck: NOT_EXECUTED
rec005_eligible: false
```

通常Pのsource replayや介入非干渉が破綻した場合は `DIAGNOSTIC_BLOCKED`。内部scoreが取得不能でも、その範囲を隠さず、妥当なgrid／出力診断から得た限定結果は報告できる。

---

## 7. 実装先・テスト・成果物

### 7.1 配置

既存の同等module／dispatcherを優先し、元forwardのdefault動作を変更しない。次の新規名は提案であって、実在APIの主張ではない。

```text
docs/CODEX_TASKS_PHASE_B_B2_MIRROR_POSITION_SCORE_RESIDUAL_AUDIT.md
src/apc/evaluation/mirror_position_score_residual_audit.py  # 必要なら新規
src/apc/evaluation/mirror_position_bias_repair.py          # 読込／計測の最小再利用
src/apc/primitives/primitive.py                           # 原則read-only。観測が必要でも演算非変更
scripts/run_phase_b_b2_model_bundle_recovery.py            # REC-004E明示dispatch
configs/phase_b_b2_model_bundle_recovery_rec004e.yaml
tests/test_mirror_position_score_residual_audit.py
runs/phase_b_b2_model_bundle_recovery/rec004e/<run_id>/
```

推奨入口（実装後の形）：

```bash
python scripts/run_phase_b_b2_model_bundle_recovery.py --task B-C005REC-004E --config configs/phase_b_b2_model_bundle_recovery_rec004e.yaml
```

run-dir等は既存CLIに従う。`--all`、latest自動選択、別修正の暗黙実行を追加しない。旧runを上書きしない。root AGENTS、復旧task/plan、DECISIONS索引と現ADR logへ新branchと結果を追記する。ADR番号は実repoから次の未使用番号を取得する。

### 7.2 必須CPU／小fixtureテスト

1. 全位置対が330で重複せず、5モデル×13時点のindexに欠測を正しく表す。
2. 実lengthとpadding幅を区別し、L_ref／特殊token offsetを実schemaに従って使う。
3. fixed/moved分割、重複tokenの曖昧さ、分母0=null、maskされたkeyの除外。
4. rowへの定数加算でsoftmaxが変わらないこと、中心化に-infを混ぜないこと。
5. 無介入observerが予測を変えず、再構成score／出力が事前許容内で一致すること。
6. J1〜J4が指定したscore成分だけを変更し、value／mask／decoderを維持すること。
7. J5/J6が第4featureだけを変え、真のn／position IDs／mask／L_refを変えないこと。
8. 任意の介入後に通常forwardへ戻ると元の予測が復元し、weights／buffersが不変であること。
9. query target／piがbias・介入入力へ流れないこと。評価専用写像はruntimeから隔離すること。
10. optimizer／trainer／builder／router較正をdynamic guardで禁止し、更新0を検査すること。
11. 観測subsetを正誤で選ばず、最高EMのinit／介入を自動採用しないこと。
12. source mismatch・NaN・unsupported・missingをPASSや0%で埋めず、query／child／REC-005が呼ばれないこと。

全13時点の不足が理由で勝手に再学習してはならない。通常の模型構築は完全stateのstrict loadに先行できるが、欠落重みを初期化値で補完しない。

focused tests後、repo標準検証を行う：

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

現repoに新しいcanonical commandがあれば従う。旧2112件PASSを今回の結果に転記しない。全suiteを省略した場合は明示する。他ジョブを無断で停止しない。

### 7.3 保存物

```text
config.yaml / protocol.json / system.json / summary.json / report.md
source_manifest.json / source_replay.json / data_manifest.json
residual_audit_protocol.json / score_observer_parity.json
position_grid.npz / position_grid_index.json
activation_summary.json / row_centered_bias_summary.json
score_decomposition.npz / score_decomposition_index.json
intervention_results.jsonl / paired_intervention_summary.json
intervention_nonmutation_audit.json / late_phase_learning_audit.json
residual_diagnosis.json / next_repair_contract.md
freeze_audit.json / side_effect_audit.json / cost_accounting.json
plots/  # bias・score・曲線の必要最小限。raw配列と分母が正本
```

source commit／checkpoint／device／dtype／framework version、実行例数、時間、VRAMを残す。未実施物は捏造しない。

### 7.4 完了報告

```text
Task: B-C005REC-004E
Source hashes / P終端replay / L_refの実値とschema根拠:
Grid: planned / completed / missing / masks / activation patterns:
Score observer parity / unavailable部分 / 許容差:
Init × length × intervention: EM, paired delta, 分子/分母:
長さ10の残差と、7〜9に残る不足:
Bias・既存score・合算の観測、attention解釈の限界:
終盤・同LR位相の進行、補間や未観測の扱い:
根拠がある原因候補 / 未確定 / 否定できないもの:
次の一案と変更する一箇所（未実装）:
Weights・Core・parent16・固定3候補・cacheの不変性:
Updates=0 / tests / commands / artifacts / new ADR:
selected_init=null / selected_intervention=null / child_bundle=null:
RG3 NOT_EXECUTED / REC-005以降blocked:
```

**STOP：診断結果と次修正の仕様案を報告して、この一件を終了する。高性能の介入が見つかっても採用・学習・独立query実行へ自動移行しない。**
