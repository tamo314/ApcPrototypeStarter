# CNP 適応の順序・保持改善設計

日付: 2026-09-22 / ADR-0202 / `IMPLEMENTATION_COMPLETE_EXECUTION_PENDING`

## 1. 目的と範囲

[改善案レビュー](../results/R_CNP001_IMPROVEMENT_REVIEW.md)を具体化する。
設計の対象は、評価の分離、提示スケジュールの固定対照、replay上の機能保持制約、
修正版親からの独立確認への引き継ぎである。設計時は研究コードの実装・モデル評価・
データ生成・学習を行わなかった。後続の実装指示により本書§6のI-1〜I-3を実装したが、
研究データ生成・モデル評価・学習は引き続き開始していない。

v1、R-CNP-001のコード経路・run・測定値・STOPを保存する。既存親は診断用の固定入力であり、
新しい診断の成功をH-CNP2の確認やbundle採択としない。
[実行計画](../exec-plans/active/CNP_ADAPTATION_SCHEDULE_AND_RETENTION.md)を作業・予算の正本とする。

## 2. 判定を分離する

### セルと測定数

- `domain_id`: sourceまたは`q1+q2+`。`query_id`: そのdomain内の固定見本識別子。
- **主セルは(domain_id, query_id, length, threshold)**。個別queryを消して集約しない。
  各panelは8 query×5長さ×3閾値=120セル。domain集約15セルは説明用に別保存する。
- 長さは1,2,4,8,16、閾値は0.50,0.80,1.10。shadowは**128集合/セル**、
  new/oldそれぞれ15,360集合、合計30,720集合。
- trainは既存予算の8 query×3閾値×64集合=1,536。各64集合の長さは従来どおり
  1,2,4,8が各13、16が12。shadow増量は学習教師の追加ではない。
- 増量は将来の新panelだけに適用する。R-CNP-001の12/13集合セルを継ぎ足さない。
  長さ1では一集合のF1差1/128=0.0078125になり得るが、これで統計的精度が
  保証されたとは主張しない。集合単位のpaired bootstrap CIも併記する。

### GateReport

各seedについて次を独立した欄で返す。判定式にCIや平均値による救済を入れない。

| 欄 | 条件 | 不成立時 |
|---|---|---|
| `panel_validity` | 全120セル、各128集合、正負両クラスあり、非重複監査PASS | `INCONCLUSIVE_PANEL`、学習開始前に停止 |
| `parent_old_quality` | 親の各旧セルBA≥0.95かつF1≥0.90 | `PARENT_INELIGIBLE`と明記 |
| `new_quality` | 候補の各新セルBA≥0.95かつF1≥0.90 | FAIL |
| `old_quality` | 候補の各旧セルBA≥0.95かつF1≥0.90 | FAIL |
| `old_retention` | 全旧セルで候補F1−親F1≥−0.01 | FAIL |
| `invariants` | 基盤hash不変、初期adapter出力一致、入力/順序/費用証跡が整合 | `INVALID_RUN_STOP` |

`candidate_gate = new_quality AND old_quality AND old_retention AND invariants`。
新セルの親比F1差は記録するが、保持ゲートにしない。
BAがN/Aのセルを捨てず、panel不成立とする。元計画G0のdomain×length×thresholdに
対する正例率[0.05,0.95]の検査も維持する。ラベルで再抽出・追加生成しない。

固定親診断では親の不足自体が研究対象なので、`PARENT_INELIGIBLE`でも**診断としてだけ**
比較を行える。候補の全セルfloorは下げない。修正版親の正式な確認では親適格性を
開始条件とし、未達の親で適応の確認へ進まない。診断の候補gate PASSと親適格性を区別する。

親→候補を、継続合格・新規失敗・回復・継続失敗の4群で保存する。
保持差は親不合格セルも含めて全旧セルで測る。親合格セルだけの保持率は補助指標に限定。

### 必須証跡

セルごとに集合数、有効要素数、正負例数、TP/TN/FP/FN、BA、集合平均F1、EM、
親との差を保存する。集合別のrecord ID、F1、EM、TP/TN/FP/FN、valid countも保存し、
学習用manifestとは別の評価artifactに置く。空target/空予測のF1=1は従来どおり。
paddingを混同行列に含めない。

CIは各セルの128集合を同じindexで親/全方式にpaired resampleし、1,000反復、
2.5/97.5 percentile（線形補間）。反復で片クラスとなるBAはN/Aとし、有限反復数も報告する。
F1/EMとBAの欠損処理を混同しない。5親はモデル初期化の反復で、独立worldでも独立panelでもない。

## 3. データと再現性

診断はworld=610000、既存CNP-003親seed=610200–610204を意図的に再利用する。
new/old panelは新しいdata rootとroleを使う。queryはrole間で分離し、new側は既存の
`q1+q2+`生成分布、old側はsource query分布を維持する。

| 診断 | data root | スケジュールroot | bootstrap root | role prefix |
|---|---:|---:|---:|---|
| R-CNP-001S（順序） | 620010 | 620020 | 620030 | `cnp_repair_schedule_v1` |
| R-CNP-001R（保持） | 620110 | 620120 | 620130 | `cnp_repair_retention_v1` |

各prefixに`new_train`、`new_shadow`、`old_shadow`を置く。new/old shadowは別query。
role、query ID、長さ、threshold、example indexから生成seedを導出し、生成順序に依存させない。
root引数を受け取る新生成器を用意し、v1のmodule定数を書き換えない。
新生成seedはnamespace/root/role/query ID/length/threshold/example indexのcanonical JSON配列の
SHA256先頭8 bytesをbig-endian uint64化する。query生成はlength/threshold/exampleを含めず
末尾`query`、bootstrapはroot/panel/cell/replicateを用いる。thresholdキーは小数2桁の文字列。
上記整数は本設計の予約で、既存configs/src/docsの登録との衝突を実装時にも静的検査する。

replayはR-CNP-001修正版と同じsource 128,000 recordの192 query×threshold stratumから
各8件、計1,536件。既存の選定規則・hashを継承し、比較方式間で同一bufferを使う。
source lineageと親が学習した範囲をmanifestで照合する。
新規oracleでsource教師を増やさず、旧sourceの決定的再構成費用は会計する。

データ分離監査はrole名だけでなく、model-visible query hash、入力record hash、
既存の入力＋target digestを用いる。hash仕様はdtype/shape/byte orderを含める。
既存source/開封済みpanelとのqueryとrecordの非重複を検査し、同じqueryを使うthreshold/length
セル内の共有、方式間の共有、replayと旧sourceの一致のみallowlist化する。
監査用の過去query再構成はラベル・モデル出力を要求しない。旧sealedは触らず静的境界で除外する。
全panel生成後、モデル実行前にmanifestを固定し、学習前に分離・件数・ラベル分布を検査する。

## 4. R-CNP-001S: スケジュール三方式

共通: 第1hidden後rank8 adapterのみ1,024重み更新、基盤8,449重み凍結、集合等重みBCE、
AdamW lr=0.001/wd=0/betas=(0.9,0.999)/eps=1e-8、clip=1、float32、256更新。
毎batchはnew16＋replay16。各側4,096提示、unique各1,536。
各親からA/B/Cを独立cloneし、adapter初期化の乱数状態を同じに戻す。B行列の0初期化を検査する。

### 4.1 quota（record別提示回数）

| 方式 | quota | 提示順序 |
|---|---|---|
| A `ORDERED` | 連続index循環が定める回数 | `(step*16+offset)%1536` |
| B `DISPERSED_MATCHED` | Aとrecord単位で完全一致 | §4.2の共通分散器 |
| C `DISPERSED_BALANCED` | stratum間の回数差≤1、stratum内record間の回数差≤1 | Bと同じ分散器 |

Aは新しいデータ上で現行順序を再現する。new listはquery→threshold→example、
replay listはquery→threshold→既存選定digest順。Aのrecord quotaは2または3。
R-CNP-001の成績をAの実測値として再利用しない。

Cのquotaは各側独立に計算する。new24 stratumは16個が171回、8個が170回。
replay192 stratumは64個が22回、128個が21回。
余りを受けるstratumを登録rootとside・stratum IDのhash順で固定する。
各stratumではquotaをrecord数で割り、余りを**targetを含まない入力hash**の登録順で配る。
全record最低2回であることを検査する。ラベル、親loss、shadow成績による選択はしない。

### 4.2 共通分散器

newとreplayを別々に作る。stratum sの総quotaをQ_sとする。

1. 各record r（quota m_r）に出現j=0..m_r−1を作り、位置`(2j+1)/(2m_r)`でsortする。
   同点は`H(root, side, s, input_id, j)`、最後は一意なinput_id/jで決める。
   これをそのstratumのrecord queueとする。
2. stratumの出現k=0..Q_s−1に位置`(2k+1)/(2Q_s)`を与える。
   全stratumの出現を位置、`H(root, side, s, k)`、s/kの順にsortする。
3. 得られたstratum列に従って各queueから一件ずつpopし、16件単位でbatch化する。

位置比較は整数の交差積を用い、float丸めを避ける。HはUTF-8のcanonical JSON配列
（区切り`,`/`:`、整数と文字列のみ）に対するSHA256のhex辞書順。
分散器はquotaだけを入力に取り、方式名やmodel seedで乱数を変えない。
B/Cに同じhash keyを使い、quota以外の分散ルールを一致させる。

実装はscheduleを学習前にJSONLへ書き、全256 batchのrecord IDsとSHA256を固定する。
batch内の条件数、最長連続query回数、stratum別quota、末尾16/64 batchの分布も記録する。
「すべてのbatchに全条件が入る」とは要求しない。比較は順序とbatch構成を合わせた介入である。

### 4.3 測定と解釈

主対照はB−A（提示順序）、C−B（同じ分散規則下の提示配分の追加効果）。
全5親・全セルの差、new品質失敗数、old新規失敗数、old保持失敗数、最悪保持差を保存する。
両要因の完全な交互作用を識別したとはしない。

事前に固定する実用候補はC。A/Bの成績を見てCから切り替えない。
全5親のcandidate gate PASSを`S_DIAGNOSTIC_VIABILITY_PASS`とする。
どれか一つでも未達なら`S_DIAGNOSTIC_FAIL_STOP`。平均性能改善だけは`DESCRIPTIVE_IMPROVEMENT`
として併記できるが、採択・次段階の合格条件を満たさない。
候補選択・promotionは全結果で0。これは固定親・単一world・一つのschedule rootに限定した診断。

## 5. R-CNP-001R: 機能保持制約の二方式

Sとは独立したrole/data rootを用い、方式は固定C schedule上のlambda=0とlambda=1のみ。
候補レシピはlambda=1と事前指定し、lambda探索や結果後の係数変更はしない。
1は事前固定の最初の尺度であり、最適値または保持を保証する係数とは主張しない。

集合平均演算を `SetMean(v)=非空集合ごとのvalid要素平均の集合間平均` として、

```text
L_task = 0.5 * SetMean(BCE(new_logits, new_labels))
       + 0.5 * SetMean(BCE(replay_logits, replay_labels))
L_keep = SetMean((replay_logits - stop_gradient(parent_replay_logits)) ** 2)
L = L_task + lambda * L_keep
```

正解BCEを残し、親の出力を正解ラベルで置き換えない。logitのclamp、温度調整、親誤例の除外はしない。
親が誤るreplayと親が正しいreplayのloss・予測変化を診断欄で分け、学習選別には使わない。
有効要素平均の式と集合平均の式を混同しない。

親の出力cacheは既存replayだけでfloat32作成。parent hash/input ID/architecture signatureと
結び、欠損・hash違いなら学習前STOP。shadow/query・targetからcacheを補充しない。
lambda=0にも比較監査用cacheを共通に用意するが、lambda=0の必須学習費用と共通診断費用を
分けて報告し、保持制約の追加費用を隠さない。勾配はadapterだけへ到達させる。

lambda=1の全5親candidate gate PASSで`R_DIAGNOSTIC_VIABILITY_PASS`。
FAILなら`R_DIAGNOSTIC_FAIL_STOP`。lambda=0との差から保持とnew品質のtrade-offを報告する。
RはSの失敗を自動的に救済するrunではなく、S結果を記録した後に実行範囲を明示する独立診断。
SのSTOP後にRを暗黙継続しない。

## 6. 実装境界と検証

追加module（名前は本設計の予定）:

| ファイル | 責務 |
|---|---|
| `src/apc/cnp/repair_protocol.py` | 型付きconfig、role/seed/panel定義、明示rootの生成・分離監査 |
| `src/apc/cnp/repair_schedule.py` | quota・分散器・schedule validator。torch/model依存なし |
| `src/apc/cnp/repair_evaluation.py` | 集合証跡、明示domain/query、親品質とnew/old gate分離 |
| `src/apc/cnp/repair_retention.py` | 集合等重み保持lossと親cacheの検証 |
| `src/apc/cnp/repair_followup.py` | 独立clone、学習、記録、予算監視。旧runnerは呼び替えない |
| `configs/cnp/repair_schedule_v1.json`, `repair_retention_v1.json` | 本設計の数値と許可modeを実装時に固定 |
| `scripts/run_cnp.py` | `repair-schedule` / `repair-retention`を明示dispatch。既定では実行しない |

`repair.py`の修正版MLPと集合lossを再利用する。新gateを旧runnerへ差し込まない。
将来のcheckpoint loadはcorrected signatureを必須にし、v1 checkpointの意味を変更しない。

必要な回帰テストは以下。実装の単なる写しではなく、破ってはいけない性質を検査する。

- A/Bの全record別回数一致、Cの各側総数4,096、stratum差≤1、unique1,536維持。
- 入力listの並び替えに対するB/Cの再現、root変化、別プロセスでのhash/順序一致。
- targetを反転しても固定bufferに対するscheduleが同一。optimizerへの入力順は保存scheduleと一致。
- 旧セル一つの未達が全体FAIL、新セルの相対低下だけではFAILにならない。
  親不合格を候補由来の失敗へ誤分類しない。N/A・欠損・query集約による救済を拒否。
- 長さ1/16、padding、空mask等の手計算fixtureで混同行列・集合F1・保持lossを独立確認。
- lambda=0で既存集合BCEとloss/gradient一致、候補=親なら保持loss0、親にgradなし。
- A/B/C初期出力一致、基盤不変、adapter勾配、fresh processでlogit・選択mask一致。
- train/shadow入力の別roleへの付け替え、同一queryの不正共有、欠けたcache、上書きを拒否。
- resource guardが時間/メモリー超過で証拠を保存して停止し、次候補へ進まない。

## 7. R-CNP-002への引き継ぎ

正式確認へ渡すのは選んだ最良checkpointではなく、hash固定した**一つの学習レシピ**。
S/Rで使ったv1親と開封済みpanelは正式確認に使わない。
S/Rの結果を受けたrecipeの決定理由は別ADRに記録して固定し、正式確認で再選択しない。

修正版親は最初から集合等重みBCEで学習する。同一world内の新root・未使用queryで
source→開発→確認→適応を分離する方針とし、別worldへの一般化は主張しない。
source 4,000更新/batch32、MLPの形状とoptimizer、2開発/5確認seed、FULL_REPLAYと
LEARNED_METRICの比較を維持する。新seed値・全role・費用上限・固定レシピを含む
R-CNP-002の実行登録はS/R後の引き継ぎ成果物であり、本設計から未決レシピを自動補完しない。

G1→訂正済み因果分母によるG2→親のold品質→G3の順を守る。
親がnewを既に解ければ`ADAPTATION_NOT_NEEDED`として更新を省略し、H-CNP2の成功に数えない。
G3は独立転移、各旧条件保持、両順序SELECT合成、改善幅と全5seed条件を
[元計画](../exec-plans/active/CONDITIONAL_NEURAL_PRIMITIVES.md)から継承する。
S/Rのshadow gateをG3全体の代わりにせず、G4は同等品質を満たしてから費用比較する。

この段階で未確定なのはS/Rで調べるレシピの有効性であり、閾値を緩める余地ではない。
