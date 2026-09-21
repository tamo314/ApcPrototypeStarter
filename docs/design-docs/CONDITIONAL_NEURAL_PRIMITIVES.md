# Conditional Neural Primitives — v1実装設計

日付: 2026-09-21 / CNP-000 / `DESIGN_BASELINE_RECORDED`

親契約: [研究方針](../research/CONDITIONAL_NEURAL_PRIMITIVES_CHARTER.md)。
実行順・判定: [実装計画](../exec-plans/active/CONDITIONAL_NEURAL_PRIMITIVES.md)。
以下のファイル名・APIは作成予定であり、実装済みとは扱わない。

## 1. 型と情報境界

新コードは `src/apc/cnp/` に隔離する。既存の整数トークン用Operation registry、
operation ID、Core vocabulary、Phase D executorを変更しない。
既存 `PrimitiveCall` は `get_operation()` と整数列に結びついているため、そのまま流用しない。
`apc.cnp.contracts.PrimitiveCall`（外部import名 `CNPPrimitiveCall`）を連続値用の型付き実装とする。
同じ「family + arguments」の契約を維持し、NNはこの呼び出しの `arguments` を実際に消費する。
旧型との暗黙変換は禁止。CNPの型はCNPにだけ適用する設計上の拡張である。

| 型 | 必須フィールド・意味 |
|---|---|
| `SetState` | `values: float32[B,N,8]`, `valid: bool[B,N]`, `item_ids: int64[B,N]` |
| `SelectArguments` | `query: float32[B,8]`, `threshold: float32[B]` |
| `PrimitiveCall` | `schema_version=cnp_v1`, `family`, 型付き `arguments`。condition IDは記録専用 |
| `SelectionResult` | `logits: float32[B,N]`, `selected: bool[B,N]`, `state: SetState` |
| `Recipe` | 呼び出しの順序。重み・訓練データを保持しない |

`item_ids` は順序復元・集合一致評価のみに使用し、NN入力にしない。
モデルに渡せるものは `values`, `valid`, `query`, `threshold` のみ。
生成行列、正解mask、出力件数、分割名、world seed、task ID、後段の正解は別の評価用レコード。
forward引数に評価用レコードを受け取らせない。

出力は `selected = valid & (logits >= 0)`。出力件数はこの予測maskから求める。
空集合は正当な状態。全無効maskでは空結果を返し、NaNや仮の一要素を作らない。
`N=0` はバッチ境界でゼロ幅として扱うか、全無効padding一枠として正規化し、両者を等価にする。
padding値、item ID、バッチ内の別例によって有効要素の出力が変わってはならない。

## 2. 最初のデータ領域

### 2.1 生成式

要素 `x` と見本 `q` は8次元、各座標の基本域は `[-1,1]`。
固定の公開特徴変換を次とする（出力16次元、訓練パラメータ0）。

```
phi(x) = concat(x, sin(pi*x))
M = 16 * (A.T @ A + 0.05*I) / trace(A.T @ A + 0.05*I)
A: [4,16], 各成分はworld RNGから標準正規分布で一度だけ生成
d(x,q) = (phi(x)-phi(q)).T @ M @ (phi(x)-phi(q)) / 16
target(x,q,tau) = (d(x,q) <= tau)
```

worldはv1全体で一つ。生成行列Mを条件ごとに交換しない。
教師関係は観測値から決まるため、観測不能なノイズから正解を推定させる問題ではない。
物理世界のノイズ除去や意味理解の実証とは呼ばない。生成器の非線形性は合成的なものである。
phiと生成式の形は全方式に公開。MとAの数値、world RNGは学習器・baselineに非公開。
公開形を使うPSD距離学習を必須baselineにし、生成構造の知識をNNだけに有利に扱わない。

主分布の要素 `x_j ~ Uniform[-1,1]`。見本の初期域は `q_j ~ Uniform[-0.5,0.5]`。
初期学習の閾値は `{0.5,0.8,1.1}`、補間評価の閾値は `{0.65,0.95}`。
要素は置換を許した独立生成。連続値でも一意数とレコードdigestを実測記録する。
入力長は初期学習で `{1,2,4,8,16}` を均等抽出し、padding後にloss対象を限定する。
全正例・全負例の集合も除外しない。ラベルを見て主分布を再サンプルしない。

確認用の見本は初期域で32個固定し、学習見本と独立。各見本について上記5閾値すべてを評価。
学習は固定の64見本と3閾値から均等抽出する。確認用32見本は実装テストや開発選択に使用しない。
開発用の見本32個は別のsplitで作成する。

### 2.2 後続の条件域追加

CNP-004では同じMのまま、初期域の外の見本を四つのblockとして追加する。
block順は `(q1+,q2+), (q1+,q2-), (q1-,q2+), (q1-,q2-)` に固定。
q1,q2の絶対値を `Uniform(0.5,1]`、残り6座標を `Uniform[-0.5,0.5]` とする。
各blockに訓練見本8個、適応に使わない転移見本16個を固定。閾値は学習済みの3値。
見本ID・block IDは学習器の入力にしない。推論時はqとtauだけで実行する。
各blockで学ぶのは既存関係の新しい条件域であり、未知操作の獲得とは呼ばない。

### 2.3 分割と乱数

予定root seed: world `610000`; data `610010`; 開発モデル `{610100,610101}`;
確認モデル `{610200,610201,610202,610203,610204}`。
これは結果前の固定値であり、既存登録との衝突監査はCNP-001で実施する。
衝突時は実行前に理由を記録した新ADRで再登録し、黙って代替seedを使わない。

すべてのdata RNGは `SHA256(cnp_v1 | root | role | block | condition | length | example)`
の先頭8byteのunsigned整数（big-endian）から導出する。roleは
`source_train`, `dev_eval`, `confirm_eval`, `adapt_train`, `replay`, `shadow`, `transfer_eval`,
`stress_eval` に分離。Pythonのプロセス依存hashは禁止。
dataのrootは610010で全model seed・比較方式に共通とし、条件リスト・教師・評価入力を
固定する。model seedは重み初期化とoptimizer以外のモデル側乱数にのみ使う。
`PYTHONHASHSEED`が必要なら別にuint32へ制限し、RNG seedをそのまま環境変数へ流用しない。
整数rootの監査に加え、生成条件、content、labelレコードdigestの分割横断重複も検査する。
同じ条件で別contentを測ることは許すが、訓練/開発/確認/shadow/転移間の同一入力record共有は禁止。
例外は明示的なlineageを持つreplayの訓練record再利用、比較方式間のpaired共有、
16集合armの64集合armへの包含だけ。`replay` RNGはbuffer選定用で、新規教師を生成しない。

確認用データのラベル・スコアはCNP-003の固定完了まで開かない。
CNP-004用の適応・転移分割もCNP-002/003で評価しない。
旧Phase B sealedとCNPの未使用確認データは別物であり、前者へのアクセスは常に0。

## 3. モデル

### 3.1 条件付き選択NN

```
h_i = phi(x_i)                             # 固定・内容のみ
a = phi(query)                            # 条件側で別に計算
v_i = concat(h_i, a, abs(h_i-a), h_i*a, threshold)   # 65次元、primitive内
u_i = GELU(Linear_65_64(v_i))
r_i = GELU(Linear_64_64(u_i))
logit_i = Linear_64_1(r_i)
```

65→64→64→1、biasあり、dropoutなし、合計8,449パラメータ。
同じ重みを全要素に使う。集合内相互作用を必要としない課題なので、初回にTransformerを
導入しない。順序同変性と可変長対応は構造で持たせるが、長い集合の完全一致は別途実測する。
これは系列・グラフ全般のモデルではない。

`ConditionalSelectPrimitive` は既存 `PrimitiveBase` の凍結・状態・呼び出し数管理を継承し、
`PrimitiveBank.add_primitive` / `get` に載せる。初回のfamily数は1。
forwardは選択された実体にだけ行い、inactive候補を計算して出力を消す実装は禁止。
学習は有効要素のBCEWithLogitsを集合ごとに平均し、その後バッチ平均。
主判定ではlogit閾値0固定。確認結果から閾値を調整しない。

### 3.2 局所適応

CNP-004の方式LOCALは基盤8,449重みを凍結し、最初のhidden層に残差を置く。

```
u_i' = u_i + B(GELU(A(u_i)))
A: Linear(64,8,bias=False), B: Linear(8,64,bias=False)
```

追加1,024パラメータ。Aをseedで初期化、Bを0初期化。初回出力は親と一致。
各blockでは親の安定adapterを複製した候補を更新する。最初のblockだけ新規adapterを作成。
blockごとのadapterを積み重ねず、選択NN一個＋adapter一個を安定実行状態の上限とする。
候補更新時の親コピー、optimizer状態、replayを含めたピーク保持量は別会計。
1,024という更新数は約12.1%に相当し、総費用12.1%を意味しない。

全blockで基盤、phi、他の部品は不変。新旧教師を含むリプレイでadapterを学習する。
shadow不合格なら候補を保存して親を継続使用し、依存する次blockを止める。
adapter統合・蒸留はこの段階に含めず、圧縮の成功とは報告しない。

## 4. 連続合成

CNP executorはSetStateを受け渡す。SELECTはvaluesを変えずvalidを予測maskに更新する。
消えた要素は後段で復活しない。毎段のq/tauはその段のargumentsから読み取る。
有効maskは実行制御であり、正解maskの注入ではない。

初期の評価recipeは次の3種類。レシピ学習や合成例による重み更新はしない。

1. `SELECT(q1,t1) -> SELECT(q2,t2)`。
2. `SELECT(q,t) -> COUNT`（COUNTは固定された決定論的な要素数計算）。
3. `SELECT(q,t) -> SUM_FIRST`（選択要素の元の第1座標の和、空集合では0）。

COUNT/SUM_FIRSTは明示的な決定論的部品として登録・計測し、NNとは呼ばない。
同じ予測maskを後段に渡す通常経路を主評価とする。正解maskリセットは診断欄だけ。
二回選択は集合積を実行系が保証する設計であり、ANDの発見や順序推論の証拠にはしない。
この段階で検証するのは予測誤り込みの出力品質と、部品更新の波及だけである。

## 5. baselineと対照

| 名称 | 実装・公平性 |
|---|---|
| ORACLE_REFERENCE | Mを知る独立NumPy参照。上限・評価器検証用。一般推論経路に混ぜない |
| RAW_DISTANCE | raw x,qの平均二乗距離＋学習データのみで適合する尺度とoffset。未学習版も併記 |
| LEARNED_METRIC | 公開phi上で学習するPSD行列 `L L.T /16` と正のlogit温度。16×16のL、256+1重み。初期L=I。Mをロードしない |
| CONDITIONAL_MLP | §3.1そのもの。CNP-003の通常NN基準とCNPプリミティブは同一実体 |
| UNCONDITIONED | q/tauをゼロ置換した同形NNを同量学習。条件不要なショートカットの検出用 |
| FULL_REPLAY | 同じ事前学習MLPから全重みを更新、LOCALと同じnew/replayミニバッチ |
| FULL_NO_REPLAY | 同じ初期モデル・同じstep上限、new例だけ。保持へのリプレイ寄与を分離 |
| SCRATCH | 同形MLPを各blockで初期化し直接学習。LOCALと同じnew/replayへのアクセス。再利用の補助対照 |
| METRIC_REPLAY | LEARNED_METRICを同じnew/replayで更新。NNの必要性を判定する強い比較対象 |

RAW_DISTANCEの尺度・offset適合は
`logit=tau-(softplus(a)*mean((x-q)^2)+b)` の二変数logistic回帰
（a,b初期値0、AdamW/lr=0.001、batch32集合、最大1,000反復）、
未学習版は `logit=tau-mean((x-q)^2)`。適合費用も計上する。
METRICのlogitは `softplus(s)*(tau-d_L(x,q))`、s初期値0。
訓練終了後の温度・閾値調整は禁止。初期Mの数値を知る方式との優位比較は禁止。

主たる方式比較はLOCAL対FULL_REPLAYおよびMETRIC_REPLAY。
SCRATCHだけへの勝利、リプレイ量が異なる方式だけへの勝利は費用利益の根拠にしない。
「タスク別モデル保存」の容量はSCRATCHで全blockモデルを保持する場合として実測する。
共有一モデルより保存量が少ないとは主張しない。

因果対照はCorrect、None（validをそのまま通す）、Wrong-family（同じ距離条件の補集合を
取る参照family）、Wrong-argument（別のqを渡す）を出力集合のIDで比較する。
Wrong-familyは学習する第二NNの代用品ではなく、誤操作の参照対照。
主スコアは全例。因果gapは参照出力が実際に反転する有効要素だけを固定して、
元の正解に対する要素正解率の差で測る（集合F1差ではない）。
ほとんどの要素が正例のセルでNoneの集合F1が自然に高くなる問題を避けるためである。
wrong qは固定の見本リストを一つ循環させる。効果が出るqを探索して選ばない。
effectfulな要素数・それを含む集合数・母集団比率を必ず報告し、
不足時は`INCONCLUSIVE`として扱う。CIの再標本化単位は引き続き集合とする。

## 6. 評価器と成果物

- mask F1（両方空なら1）、要素balanced accuracy、集合完全一致EM、COUNT完全一致、
  SUM_FIRST絶対誤差を別々に記録。COUNT一致だけで選択集合が正しいとしない。
- balanced accuracyは例ごとでなくセル全有効要素で集計し、片クラスしかないセルはN/A。
- セルは `model_seed × condition_domain × threshold × length × recipe`。
  見本ごとの値も保存し、条件平均だけで欠損を隠さない。
- 反復内の信頼区間は入力集合単位のpaired bootstrap、1,000回、role専用RNG。
  五つのモデルのばらつきを別に表示。要素を独立試行と数えてCIを狭めない。
- 性能、実行時間、メモリー、モデル・config・data・実行コードのhashを保存する。
  評価生成器のハッシュだけでなく件数・ユニーク数・長さ・正例率も出す。
- 全パネルは `population=sampled`。小さな手書きの列挙単体テストだけがexhaustive。
  訓練生成器と参照関数を共有して同じ誤りを隠すテストにしない。

保存は `runs/cnp_v1/<task>/<run_id>/`、存在する出力先は拒否。
静的gateはディレクトリ作成より先に走らせ、生成後の失敗は証拠を残す。
`run_id`は設定hash＋UTC開始時刻。途中再開は同一runのcheckpoint・RNG・optimizer完全一致を
検証できる場合だけ。新seedや新configは新run。科学的な不合格をresumeで再試行しない。

CNP bundleは別schema `cnp_bundle_v1`。既存token-schema loaderへ偽のvocabularyを渡さない。
`canonical_state_hash`などの純粋なhash utilityとPrimitiveBaseの共通機能は再利用する。
manifestはphi/version、family、引数型、基盤/adapter重みhash、dataset/split/config/code hash、
訓練seed、親bundle、使用条件、shadow結果、resident/active/temp数を持つ。
world秘密行列は評価用artifactに別保存し、モデルbundleには含めない。
loaderに学習や不足ファイル生成のfallbackを持たせない。
fresh-loadは同じdeviceでlogits/maskの完全一致を要求する。CPU/CUDA相互では
logitsのatol=1e-5、rtol=1e-5とmask一致を確認し、閾値近傍の不一致も報告して隠さない。
実験の正式なdeviceはCUDA一つに固定し、近傍例を理由にデータを削らない。

## 7. 実装ファイルと意味のあるテスト

| 予定ファイル | 責務 |
|---|---|
| `src/apc/cnp/contracts.py` | 連続値State、型付きCall、Recipe、情報境界 |
| `src/apc/cnp/data.py` | RNG分離、条件生成、データmanifest。oracle実装とは分離 |
| `src/apc/cnp/reference.py` | 独立参照と決定論的下流部品 |
| `src/apc/cnp/primitive.py` | 固定phi、MLP、adapter、厳密疎実行 |
| `src/apc/cnp/baselines.py` | 距離・距離学習・更新方式 |
| `src/apc/cnp/execution.py` | SetStateによる通常連続実行 |
| `src/apc/cnp/training.py` | 一度だけの学習、候補分離、replay |
| `src/apc/cnp/evaluation.py` | パネル、対照、集計、通常/診断経路分離 |
| `src/apc/cnp/artifacts.py` | fail-closed保存・ロード・CNP manifest |
| `scripts/run_cnp.py` | `audit`, `develop`, `confirm`, `adapt`, `report` の分離CLI |
| `configs/cnp/v1.json` | 数値・分割・比較方式・予算の唯一の実行設定 |
| `tests/test_cnp_*.py` | 以下の振る舞い・不変条件の回帰テスト |

必須テスト: 内容／引数分離、oracleメタデータをforwardできない型境界、見本を変える対照、
空集合、重複要素、padding不変性、順序同変性、長さ1、部分バッチ、選択解除後の非復活、
独立参照との小例一致、境界d=tauの規則、split重複検出、未選択forward=0、凍結重み・勾配、
候補棄却後の親同一性、不正hashロード拒否、別プロセス復元、CPU/CUDAのmask一致。
浮動小数の閾値境界例は別記し、主分布からこっそり除外しない。
1optimizer stepによる勾配配線の単体テストは可。単体テストを研究性能の根拠にしない。
