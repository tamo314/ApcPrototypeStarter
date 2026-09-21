# CNP-004 結果レビューと改善順序

日付: 2026-09-22 / ADR-0197 / `REVIEW_COMPLETE_PROTOCOL_DEVIATIONS_IDENTIFIED_STOP_PRESERVED`

## 1. 結論と範囲

**保存されたLOCAL候補は5/5 seedで失敗した。一方、実装と事前設計に重要な不一致があり、
「設計どおりの局所適応が失敗した」「adapter容量が足りない」とはまだ結論できない。**
先に評価・損失・adapter配置・replay選定を契約と整合させ、学習順序の偏りを切り分ける。
層数、rank、教師数、更新数を増やすことを最初の改善策にしない。

本レビューは保存JSON、コード、git履歴、ファイルhash、整数indexの提示回数の確認のみ。
研究コード・設定・重みは変更していない。新規データ生成、model forward、学習、optimizer
update、候補選択、promotion、legacy sealed accessはいずれも0。
過去の`G3_FAIL_STOP`と全runを保持する。転移・合成・block 2–4の適応・G4は未実行のまま。
既存preflightにはblock 2–4の**親shadow評価**が既にあるため、それらまで未評価とは呼ばない。

根拠は[実行計画](../exec-plans/active/CONDITIONAL_NEURAL_PRIMITIVES.md) §3.3–5、
[設計](../design-docs/CONDITIONAL_NEURAL_PRIMITIVES.md) §3、
[機械可読監査](CNP004_RESULT_REVIEW_AUDIT.json)。

## 2. 保存結果の再集計

主run: `runs/cnp_v1/adapt/cnp004_block1_evidence2/`、block `q1+q2+`。
実行commit `be10eed49cb925a35c6ebfd02a7b58a794392705`、config SHA256
`19b73713d29aa90160db221535f120786b134cfd783c3565601685fffbe218be`。
world/data seedは610000/610010、model seedは610200–610204。
新規教師1,536集合、source replay 1,536集合、new/source shadow各1,536集合。
各候補256 updates、LOCAL/FULL_REPLAY/SCRATCHはnew16＋replay16、FULL_NO_REPLAYはnew32。

以下は**長さ・閾値を混ぜた保存shadow指標の5 seed平均**であり、契約の各セル判定や
転移評価の代替ではない。BAはbalanced accuracy、旧F1差は候補−親。

| 方式 | 新BA | 新F1 | 新集合EM | 旧F1差 | 保存shadow合格 |
|---|---:|---:|---:|---:|---:|
| 凍結MLP親 | 0.88495 | 0.86645 | 0.57422 | — | — |
| LOCAL | 0.89418 | 0.87664 | 0.58945 | −0.01180 | 0/5 |
| FULL_REPLAY | 0.93534 | 0.92625 | 0.70339 | −0.02011 | 0/5 |
| FULL_NO_REPLAY | 0.92225 | 0.90786 | 0.66107 | −0.10651 | 0/5 |
| SCRATCH | 0.68841 | 0.76351 | 0.37096 | −0.20978 | 0/5 |

LOCALの各seedは次のとおり。品質floorはBA≥0.95、F1≥0.90、保持許容低下は0.01。

| seed | 新BA | 新F1 | 新F1差 | 旧F1差 | 集約保持条件 |
|---|---:|---:|---:|---:|---|
| 610200 | 0.90728 | 0.89107 | +0.02513 | −0.01582 | FAIL |
| 610201 | 0.90053 | 0.88265 | +0.01238 | −0.01760 | FAIL |
| 610202 | 0.89399 | 0.87266 | −0.00326 | −0.01046 | FAIL |
| 610203 | 0.89394 | 0.87509 | +0.01452 | −0.01461 | FAIL |
| 610204 | 0.87514 | 0.86175 | +0.00220 | −0.00052 | PASS |

LOCALの新F1は全seedで集約値自体が0.90未満なので、各セルを復元しても全セル合格には
ならない。停止の根拠は残る。610204の保持PASSは集約値だけで、各セル保持を保証しない。
基盤hashは全seedで更新前後一致。凍結違反は保存証拠上認められないが、基盤を凍結しても
adapterが旧入力への出力を変えるため、機能保持は自動的には成立しない。

FULL_REPLAYはLOCALより新F1が高いが、全seedで保存新BA<0.95、旧F1低下>0.01。
「全重みなら既に解決」「LOCALだけの容量不足」のどちらも支持しない。
FULL_NO_REPLAYの旧F1低下は大きく、replayを外す方針を支持しない。ただしnew提示回数も
2倍に変わるため、差のすべてをreplayの有無だけの因果効果とは扱わない。
SCRATCHの旧F1差はrunnerが共通MLP親に対して算出した参考値で、SCRATCH自身の親保持ではない。

### 単純な比較方式の既存証拠

`cnp004_preflight1/report.json`のblock 1で、LEARNED_METRIC親は更新前から
BA=0.960930、F1=0.961208、EM=0.800130。各seed欄で同一であり、独立した5回の成功とは数えない。
モデルは257重みで、LOCALの8,449＋1,024重みより小さい。
同じworldの距離構造を使う方式として、次の比較でも必ず残す価値がある。
ただしpreflightも集約shadow判定なので、記録された`ADAPTATION_NOT_NEEDED`を
各セル・旧条件保持・転移・合成・総費用の合格に拡張しない。G4の勝者は未確定。

## 3. 確認できた仕様不一致と観測上の限界

### 3.1 adapterの挿入位置が異なる

設計§3.2は「最初のhidden層に残差」を入れ、`u'=u+B(GELU(A(u)))`とする。
実装[primitive.py](../../src/apc/cnp/primitive.py)の`forward`は
`input_proj → GELU → hidden_proj → GELU → adapter → readout`である。
すなわち第2隠れ層後の補正を試しており、設計の第1隠れ層後の補正ではない。
後段の固定非線形変換に入る前の特徴を調整できるか、という点が異なる。
今回の結果は実装された配置の結果として有効だが、登録された配置の反証には使えない。
配置を直すと改善するという因果証拠もまだない。

### 3.2 損失の集合重みが異なる

設計§3.1は「学習は有効要素のBCEWithLogitsを集合ごとに平均し、その後バッチ平均」。
実装[training.py](../../src/apc/cnp/training.py)の`masked_bce_loss`は
`losses[valid].mean()`で、全有効要素の平均である。
同じ要素当たり誤差なら、長さ16の集合は長さ1の集合の16倍の総寄与を持つ。
長さ別品質と集合平均F1を要求する契約に対して、これは無視できない目的関数の相違。
集合内で平均してから集合間平均する式へ整合させるべきだが、効果は未測定。

このhelperは初期学習にも使われている。従って適応だけ修正してCNP-003親まで
元設計どおりだったと扱わない。親を保存して「実装済み親上の適応修正」とするか、
初期学習から改訂するかを後続契約で明示する。既存H-CNP1の観測値・訂正G2 PASSは保存し、
本レビューだけでその実測を取り消したり、修正版の親を新規学習したりしない。

### 3.3 G3を集約値だけで判定している

計画§4は「各旧domain・各長さ・各閾値のF1低下 ≤0.01」とG2品質floorの保持を要求する。
[adaptation.py](../../src/apc/cnp/adaptation.py)の`evaluate_panel`は全panelに一つの
`MetricsAccumulator`を使い、`quality_floor`と`retention`はその集約値だけを見る。
保存JSONには各セルの指標・混同行列・予測がなく、artifactだけでは正規判定を復元できない。
shadow、親の再利用判定、候補の保持判定を同じdomain×length×threshold単位に揃える必要がある。
閾値を変える修正ではない。LOCAL停止は§2のF1により引き続き妥当。

### 3.4 replay選定が契約のhash規則と異なる

計画§3.3はdomain内をrecord hashの小さい順に選ぶ。
`source_replay_records`はsourceの最初の48 batch（1,536集合）だけを取得し、
その**取得後**にsortする。初期学習全128,000集合からhash順で1,536集合を選ぶ処理ではない。
これは代表性に影響し得るが、今回の保持失敗の原因とまでは未同定。
保存済みsource教師を正本とする選定・lineage・query/threshold/length分布を記録する必要がある。
今回source全件の再生成やbuffer交換はしていない。

### 3.5 一候補・再利用・計測の実装が未完

block1 runnerは親new shadowを測るが、親がfloorを満たす場合の0step分岐がない。
今回のMLP親は全seed不足なので、今回の不要更新を示す所見ではない。後続運用には修正が必要。
`0,16,64,256`のスコアという計画に対し、保存されるのは16/64/256の最終ミニバッチlossと
最終shadow集約だけ。lossは毎回異なるquery/thresholdのbatchなので、同一panelの学習曲線や
収束判定として比較できない。最適化不足・過適合・容量不足をこのログだけで区別できない。

方式別時間、optimizer/replay/親候補の容量、fresh-load parity等のCNP-004証拠も揃っていない。
METRICの転移比較、小予算arm、後続streamはSTOP後に未実行であり、これらを成功扱いしない。
G4は未実行で、約94秒という一括時間からLOCALの費用利益を計算しない。

## 4. 学習順序の偏り：確定事実と仮説の区別

`adaptation_records`はquery→threshold→64例の順に並ぶ。
`train_fixed_candidate`はshuffleせず連続indexを循環する。
LOCAL/FULL_REPLAYでは同じqueryが12更新、同じquery×thresholdが4更新続く。
256更新の新規提示は4,096回で、内訳は次のとおり（ラベルやモデルを使わない整数計算）。

| query index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 提示回数 | 576 | 576 | 576 | 576 | 576 | 448 | 384 | 384 |

全1,536集合を使っているが、最初の1,024集合は3回、残り512集合は2回。
最後のnew batchはquery 5・threshold 0.5のみである。
**queryごとの連続更新と終端偏りが新旧汎化を損ねた可能性**はあるが、原因としては未確認。
shuffleが元契約で明示されたわけではないので、これは仕様違反と区別した改善仮説である。
次に調べるならnew16/replay16、unique教師数、256更新、方式間paired入力を固定したまま、
ラベル非依存のquery×threshold層化・順序分散を一つの事前固定対照として比較する。
まず仕様不一致を整理し、配置・損失・順序を同時変更して一つの原因が分かったとは言わない。

## 5. 改善の優先順位と次の判定

1. **評価と契約を先に整合。** 各セルのBA/F1/保持差、正負例数、集合数、混同行列、
   親再利用、基盤hash、データ選定、学習設定を一つの固定manifestで監査できるようにする。
   誤ったセルを良い平均で救済せず、通常テストでadapter位置・集合重みの意味を検証する。
2. **登録された小型構成を正しく試す準備。** 第1隠れ層adapter、集合平均loss、
   契約どおりのreplayを固定する。これは仕様整合であり、性能改善はまだ約束できない。
   親の学習lossの相違も明示し、既存親を継承する範囲を決める。
3. **学習順序を小さな独立仮説にする。** 開発専用の固定条件で、順序以外を揃えた有限対照を
   事前登録する。教師増量・rank増量・lr探索より先に、偏りと保持の関係を確かめる。
   train/new-shadow/old-shadowを同じ単位で記録し、shadowによるbest checkpoint選択をしない。
4. **保持と単純方式を同時に見る。** 新精度だけでなく全旧セルのF1低下≤0.01を維持する。
   学習済みmetric親の各セル再利用資格を確認し、不要なら0stepとして費用に反映する。
   metricが同等品質で低費用なら、その方式を採ることも正式な改善である。

修正版LOCALが全seedの正規shadowを通過した場合だけ、登録済み手順に従って未使用転移・
合成・費用の確認へ進む。再度失敗なら、その境界で停止して限界を記録する。
容量拡張や保持loss追加は、上記を通じて表現不足または干渉が特定できた場合の別候補であり、
現時点の第一推奨ではない。

これらは**提案であり未実装・未実行**。実行計画§5の
「保存済み確認結果を見た後の変更はv2という別計画であり、v1の再試行として成功させない」
を適用する。適応改訂版は既存の確認専用CNP-V2-001と区別した契約にする。
今回開いたshadowは開発・診断用として扱い、修正後の独立確認を既開封shadowの再測定だけで
済ませない。新しいsplit・有限予算・停止条件を固定し、実行範囲が指定されてから進める。

## 6. 仮説・費用・証拠の扱い

| 対象 | 本レビュー後の解釈 |
|---|---|
| H-CNP1 | 既存checkpointについての訂正panel／未使用query panelの支持を保持。初期lossの仕様差を注記 |
| H-CNP2 | 今回の実装・予算では不支持。登録配置・損失・正規セル判定による確認は未成立 |
| H-CNP3 | 合成評価未実行。合成波及そのものを反証したとは言えない |
| H-CNP4 | G4未実行。パラメータ数だけの費用利益主張は禁止 |

主runは94.046791秒、peak CUDA allocation 68,253,184 bytes、peak process RAM
1,874,268,160 bytes。対象機は既定のRTX 5060 Ti、主runの環境はWSLである。
これらはrun全体の記録値で、方式別値ではない。

さらに保存領域には`cnp004_block1`、`cnp004_block1_evidence1`、`cnp004_block1_evidence2`
という3回の20候補学習があり、各5,120、合計**15,360 updates**を記録している。
git差分は途中でdata manifest/metrics保存追加とLOCAL初期化seedの修正を示す。
主run一回分だけを全研究費用として報告しない。3 runとpreflightの記録wall time合計は
322.093788秒（source事前学習・検証・未記録overheadを含まない可視範囲）。
これは51,200の全stream上限超過を示すものではないが、契約の「256step候補一つ」と
実際の再実行履歴は区別して申告する必要がある。最良runを選んで5seedへ混在させない。
ADR-0196の数値は最終runの記述として保持し、全面的な仕様準拠・実装不備なしという解釈は
本ADR-0197で限定する。

## 7. 今回の検証と成果物

- PASS: 主runの記録コードhash全15件、config hash、親report/manifest hashが現ファイルと一致。
- PASS: `metrics.jsonl`の20行と`report.json`の20候補が一致。集計・seed別判定・提示回数を再計算。
- 保存data manifestは4 panel各1,536 unique、cross-panel overlap 0を報告。
  今回はデータを再生成せず、分割監査そのものを再実行したとは扱わない。
- Python 3.12.13標準ライブラリだけで監査を作成。generator:
  `runs/cnp_v1/review/cnp004_result_review/review_artifacts.py`。
- 成果物: 本文書、[監査JSON](CNP004_RESULT_REVIEW_AUDIT.json)、ADR-0197、decision index、計画の参照追記。
- PASS: ローカルリンク・anchor 11件、方式表4行・seed表5行の丸め値、入力78ファイルと
  コード15ファイルのhash不変、費用集計、`git diff --check`。
  検証記録は`runs/cnp_v1/review/cnp004_result_review/verification.json`。
- 研究実装の変更がないためpytest/ruff/mypyと研究実験は今回実行対象外。
  前回の全体pytestは既存artifact欠損で未完走であり、PASSへ変えない。
