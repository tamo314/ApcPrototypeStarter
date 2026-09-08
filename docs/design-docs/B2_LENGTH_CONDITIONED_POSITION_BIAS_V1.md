# 設計書 — Length-Conditioned Position Bias V1（B-C005REC-004D）

**対象operator：MIRROR_HALVES（`CrossPositionPrimitive`）のみ。他15 operationのarchitectureは変更しない。**

## 1. 目的

REC-004/REC-004A/REC-004B/REC-004Cを通じて、既存compact operator（`CrossPositionPrimitive`）でMIRROR_HALVESを0.95 recovery floorまで学習させる試みは、budget延長（REC-004A）・schedule変更（REC-004B）のいずれでも成功していない。REC-004Cの5初期化診断は0.20〜0.50の範囲に留まり、budgetやscheduleではなく、operatorが実位置情報へ直接アクセスする経路を持たないことが未検証のまま残っている（position/lengthの明示的特徴が既存architectureに存在しない、`docs/research/B2_MIRROR_POSITION_BIAS_SOURCE_NOTES.md`参照）。

本書は、その仮説を検証するための最小限の構造変更（新規192 parameter）を定義する。位置スコアの追加が根本原因を解決するという結論は事前に採用しない。

## 2. Architecture

### 2.1 新機構

```
phi(i, j, n) = [ i/d, j/d, (j-i)/d, n/L_ref ]     where d = max(n-1, 1)
b_theta(i, j, n) = w2^T ReLU(W1 * phi(i, j, n) + a1)
score_new[i, j] = score_existing[i, j] + b_theta(i, j, n)
```

- `i`：出力（query）content位置。`j`：入力（key）content位置。`n`：その例の実content長（padding幅ではない）。
- `W1 ∈ R^{32x4}`, `a1 ∈ R^{32}`（`nn.Linear(4, 32, bias=True)`）、`w2 ∈ R^{32}`（`nn.Linear(32, 1, bias=False)`）。
- hidden width固定32、head間で完全共有（1つのbias値がn_head=4全headへブロードキャストされる）。
- 追加parameter数：`4*32 + 32 + 32*1 = 128 + 32 + 32 = 192`。既存primitive本体17,098 parametersに対し1.12%（budget上限5%以内）。

### 2.2 入力の禁止事項

`phi`は`i, j, n, L_ref`のみから構成する。次を明示的に禁止する：教師位置写像`pi_n(i)`、正解出力との距離、`mid = n//2`、同一half判定（hard mask）、operation専用lookup table、content token値そのもの。これらを入力・埋め込みへ混入させないことをtestで検査する（`tests/test_mirror_position_bias_repair.py`の該当テスト）。

### 2.3 注入経路

既存`CrossPositionPrimitive.forward`は`nn.MultiheadAttention`へ`key_padding_mask`（bool）を渡している。新機構はこれを、`ShiftRelativePrimitive`/`ReverseRelativePrimitive`が既に採用している加算float `attn_mask`（padding位置`-inf`、有効位置はbias値そのもの）に置き換える。この経路は`softmax`の直前（scaled dot-product attentionのscore）へ加算され、既存のscale（`1/sqrt(d_operator/n_head)`）やdtypeを変更しない。

`b_theta`の出力層`w2`をゼロ初期化することで、この置き換え自体がbias=0時に既存`key_padding_mask`と数学的に同一の結果を生む（padding位置は両者とも`-inf`相当、有効位置は両者とも加算0）。これはStage Bの`zero_bias_parity.json`で forward が bit-exact（両device）、勾配がdevice依存の浮動小数点許容差内（CPU上約2e-8、CUDA上0.0、事前登録許容差1e-6）で一致することを実測確認する。

### 2.4 初期化

- 共通部分（`content_in_proj`、`content_position_embedding`、`answer_query_embedding`、`cross_attn`、`attn_norm`、`ffn`、`ffn_norm`、`readout`）：REC-004Cが保存した`initial_states/I01.pt`〜`I05.pt`から`strict`にload。
- 新bias部分（`position_bias_hidden.weight/bias`、`position_bias_out.weight`）：5本共通の単一state（`shared_bias_initial_state.pt`）を1回だけ生成し、全5 pairへ同一値でload。`position_bias_hidden`は通常のPyTorch既定初期化（非ゼロ）、`position_bias_out`は明示的にゼロ初期化。
- 両層を同時にゼロ初期化しない（`w2=0`かつ`W1=0`だと`w2`自身の勾配も恒久的に0になり、学習不能になる）。`W1`が非ゼロである限り、`w2`は初期stepからその場で非ゼロ勾配を受け取り、1 update後には`W1`にも勾配が伝播する（`gradient_path_audit.json`で実測）。

## 3. Masking / Layout

- padding key（`j >= 実content長`）は既存同様、あらゆる長さで`-inf`。
- 異なる実長を同一batch内に混在させる場合、bias tensorは`[batch, out_max, lmax]`で各exampleごとに独立計算する（同一形状のbias行列を異なる長さのexample間で誤って共有しない）。
- `L_ref`はmodelの合法最大content長（`CrossPositionPrimitiveConfig.max_sequence_length`の既定値32、`DEFAULT_MAX_SEQUENCE_LENGTH`）に固定し、batch毎の最大長やvalidationの最大長では変えない。

## 4. Serialization / Strict Contract

新architecture `CrossPositionLengthBiasPrimitive`は`CrossPositionPrimitive`のsubclassとして実装する（共通submoduleをそのまま継承し、`forward`のみ完全に上書き）。これにより：

- `state_dict()`は共通17,098 parametersに加え、`position_bias_hidden.weight`、`position_bias_hidden.bias`、`position_bias_out.weight`の3 keyを追加で持つ。
- 旧`CrossPositionPrimitive`へこのstate_dictを`strict=True`でloadすると`unexpected key(s)`で失敗し、逆に新classへ旧state_dictを`strict=True`でloadすると`missing key(s)`で失敗する。`strict=False`によるbias破棄loadの経路はどこにも実装しない。
- `PrimitiveManifestEntry.architecture_signature`を`"cross_position_length_bias_v1"`（旧`"cross_position_v1"`と区別）として記録し、`state_abi_hash`は`weights_hash`の複製ではなく、`apc.utils.model_bundle.compute_state_abi_hash`（新規追加、REC-002が定義した13 checkは変更しない）による実ABI（key名／shape／dtype、architecture_signatureでtag付け）を記録する。
- bank／fresh-processでの再構築（`scripts/rec004_fresh_process_check.py`）は、manifestが宣言する`architecture_signature`ごとに正しいclassを選んで構築する。旧genericクラスへの黙示的追加は行わない。

## 5. スコープ外

Core再学習、query/key/value/FFNの幅・層数・head数変更、別LR/optimizer/loss/batch/精度、curriculum、追加init、6000超の学習（10本合計60000上限）、router/argument scorer再較正、verifier変更、位置教師付き補助loss、oracle gather、REVERSE+SHIFTのruntime置換、機能floor緩和。これらは`docs/CODEX_TASKS_PHASE_B_B2_MIRROR_LENGTH_POSITION_BIAS.md`§2.3で明示的に禁止されている。
