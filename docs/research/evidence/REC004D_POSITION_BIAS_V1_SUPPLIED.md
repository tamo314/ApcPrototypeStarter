# Design — Length-Conditioned Position Bias v1

**適用タスク：B-C005REC-004D / 新規の実装案。実repoに既に存在するAPIや実験成功を示す文書ではない。**

## 1. 介入仮説と非主張

REC-004Cは、同一Core／訓練列／予算でも初期重みによってMIRROR_HALVESの性能と長さ別失敗が変わると報告した。ここでは「現在のposition/length情報を、attentionの位置指定へ直接使える小さな経路が学習を助けるか」を検証する。

情報が元表現に存在しないと決めない。位置項が成功しても、追加parameter数の寄与を完全に分離した証拠でも、元構造の原理的表現限界の証明でもない。操作定義を知った人間が選んだ構造を一操作へ試す実験であり、自律的なoperator discoveryとは別である。

## 2. 演算の境界

```text
content + valid-content mask / position layout
  ├── 既存のf(content) ── 既存query/key/value/residual/readout
  └── n,i,jの汎用座標 ── 小型MLP ── attention scoreへの加算

TaskSpec / operation ID / pi_n / 正解出力
  └── position biasの入力には使用しない
```

task-blind Coreは不変。新項は、選択済みprimitiveの実行時だけ計算する。選択に関係なくbank全体のbiasやprimitiveを実行しない。

teacherは従来の訓練target生成・評価用に使う。位置写像はmetric専用。runtime moduleがoperation interpreter、metric helper、target position mapへ依存する構造を禁止する。

## 3. 座標契約

`i`は出力contentの0始まり位置、`j`は入力contentの0始まり位置、`n`はその例の有効content長とする。`n`はpadding幅でもbatch最大長でもquery targetの長さでもない。

MIRROR_HALVESはlength-preservingとのsource報告に基づき、input maskからnを得られる。特殊tokenがある場合、attention上の行・列からcontent位置へ写すlayoutを明示する。queryが単純なcontent position gridではない場合は、位置対応を実コードから確定し、未確認のoffsetを埋めない。

- 有効content対：下記MLPを適用する。
- 特殊tokenの行・列：追加biasは0。既存maskの意味を変えない。
- padding key：既存の禁止を維持する。
- padding query：既存の出力・loss maskで処理し、無意味なall-masked softmaxによるNaNを増やさない。
- length0等の不正入力：現在の合法入力契約に従いtyped error。黙ってn=1とみなさない。

`L_ref`は既存model schemaの最大content長を読み取り、正の固定値としてmanifestへ保存する。position embeddingの行数に特殊token分が含まれる場合、そのままcontent最大長とせず対応を明記する。値が確定できない場合は設計前提未確定として止める。

batchや評価suiteにより`L_ref`を変えない。対応範囲外を勝手にclipしない。学習範囲外の長さについて、動作可能と一般化成功を区別する。

## 4. 固定する新項

有効位置で、

```text
d = max(n - 1, 1)
phi = [i/d, j/d, (j-i)/d, n/L_ref]
h = ReLU(W1 @ phi + a1)
b = w2 @ h
```

を計算し、既存のscaled attention scoreへ加える。

```text
S'[batch, head, i, j] = S_existing[batch, head, i, j] + b[batch, i, j]
A = softmax(S' + existing_mask, key_dimension)
```

`W1: (32,4)`、`a1: (32,)`、`w2: (32,)`。追加学習parameterは **4×32+32+32=192**。全head・全長で同じparameterを共有する。出力bias、head別MLP、learned gate、追加temperature、別norm、dropout、残差FFNは追加しない。

出力側だけに依存する一様row offsetではなく、key方向にも変化可能な非線形項であることをテストする。queryごとに全keyへ同じ定数を加えるだけの実装にしてはならない。

MLPへ正解対応やその補助表現を渡さない。特に `n//2`、半区間分類、`pi_n(i)`、正解位置との差、same-half mask、操作固有のposition tableをfeatureとして追加しない。汎用座標からの学習だけを許可する。

元のcontent-driven attentionを0にしたり、bias-onlyへ置換したりしない。projection・residual・decoderの計算は元のまま。出力tokenへ直接補正を加える新経路も禁止する。

## 5. 初期値とparameter更新

共通base stateはREC-004Cの `initial_states/I01.pt`〜`I05.pt` を厳密にloadする。旧訓練済みcheckpointをwarm-startにしない。

新MLPの隠れ層は、実repoの通常Linear初期化規約を一つ選んで仕様固定する。固定namespace `rec004d_shared_position_bias_init_v1` 相当の局所RNGから一度作り、そのstateを全5本へcopyする。規約・init seed・全tensor hashを保存し、モデル構築で消費したRNGは訓練開始前に戻す。

`w2=0`で、初期bを全位置で0にする。隠れ層まで全ゼロにして学習不能にしない。最初のupdateでw2へ勾配が入り、その後必要な隠れparameterへ伝わることをtiny fixtureで確認する。first updateで隠れ層gradientが0でも、出力層ゼロ初期化に由来する場合は不具合と混同しない。

Pではbase parametersとMLP parametersを一つの既定optimizer設定で更新する。追加parameter専用の大きいLRやweight decay例外は作らない。Uは元parametersのみ。共通Coreやbank内の非対象parametersをoptimizerへ渡さない。

## 6. 既存attention APIとの接続

sourceで`cross_attn`の実型・入力shape・maskを確認する。外部APIの参照はsource notesのW1/W2。固定version資料の確認は、現repoのPyTorchをupgradeする指示ではない。

`nn.MultiheadAttention`を使っているなら、float additive attention maskを通す既存契約を優先する。例ごとのnが異なる場合、2-D maskの誤broadcastを避け、必要なbatch×head layoutへ正しく展開する。既存のkey-padding maskとattention maskのdtype・意味を整合させる。[W1]

`scaled_dot_product_attention`を使う場合は、そのAPIのbool maskの意味とbroadcast規約に合わせる。MHAのbool maskと同じ意味だと仮定しない。[W2]

新たな独自CUDA/Triton kernel、attentionの手書き全面置換、ライブラリのupgradeは不可。小さなadapterでは保存契約を保ったscore加算を実現できない場合は `POSITION_SCORE_INJECTION_UNSUPPORTED` とし、別設計を勝手に選ばない。

## 7. Zero-bias等価性の検査

変更前operatorと新operatorの共通stateを同じにし、b=0で比較する。

- eval/train modeのlogitsと有効出力。
- 同じscalar lossからの共通parameter／入力への勾配。
- 異なる実length・合法padding幅・batch組成。
- 既存のNone/Wrong対照とmask処理。

CPU tiny fixtureおよび本番device/dtypeで検査する。数値許容は既存の再現契約から引用し、測定前に固定する。exactに一致しないbackendなら差を記録し、許容される丸め差と機能差を区別する。Pだけ計算精度・batch・determinismを変更して等価性を作らない。

Pの非ゼロbias状態でもpadding/batch不変性を確認する。隠れたnの算出ミスはb=0状態だけでは検出できない。

初期b=0は「予測関数が最初は同じ」という意味であり、更新後も同じになる要求ではない。Pでは追加parameterの勾配が更新されるため、以後のtrajectoryが変わることは意図した介入である。

## 8. 保存とbundleの扱い

新typeの提案例は `cross_position_length_bias_v1`。実repoの型体系へ合わせ、manifestに以下を含める。

```text
architecture_type / architecture_version
base_architecture_signature
position_feature_version
hidden_width: 32
head_sharing: shared
output_bias: false
L_ref / content-layout version / position-offset contract
full_state_hash / bias_state_hash
core_dependency_hash / decoder_dependency_hash
parameter_count_base / parameter_count_added / parameter_count_total
training_recipe_hash / init identity / data role hashes
```

保存されたarchitectureからnew-type moduleを組み立て、全stateをstrictにloadする。名前がMIRROR_HALVESなら自動で新型に変えるのではなく、明示typeでdispatchする。既存MIRROR artifactは従来型で読み込めるままにする。

依存bankのcontent hash・ABI・実行署名は新しくするが、parent原本を変更しない。形が同じだから別Coreのbiasを流用してよいとはしない。reference／recipe cacheの資格はversion依存として無効化し、新queryまたは既定referenceで再認定する。

新型がforward上動くだけでは採用可能ではない。manifest指定・新process・builderなしで同じ出力を再現できることが必要。

## 9. 解釈と次の分岐

Pが改善すれば「位置／実lengthへの小型追加経路が、このCore・訓練条件で有効」と言える。元の位置情報が失われていた、容量が192個だけ不足していた、他のoperationや未知長でも有効だとは言えない。

学習済みPからbiasを0にした比較は、Pが新項を利用するかの観察であり、元Uとの学習経路の違いまで取り除く対照ではない。

Pが未達なら全曲線・gradient・bias range・位置別結果を保存して停止する。hidden幅、入力feature、LR、budgetを同じtask内で追加探索しない。5試行中の最高値を採用して安定性成功ともしない。
