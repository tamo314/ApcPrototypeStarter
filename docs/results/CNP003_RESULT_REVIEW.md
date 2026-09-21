# CNP-003 結果レビューと改善方針

日付: 2026-09-21 / ADR-0191 / `CAUSAL_EVIDENCE_INVALID_MEASUREMENT_REPAIR_REQUIRED`

## 1. 結論と範囲

CNP-003の保存された結果は `G2_FAIL_STOP` のまま保持する。ただし、因果評価器が
契約で指定した介入別のreference-effectful集合を測っていないため、このFAILを
「モデルが条件を使えなかった」という科学的反証には使えない。
因果証拠は `INVALID_EVIDENCE`、H-CNP1は確認未成立、CNP-004は引き続き停止とする。
通常精度の観測値は保持し、修正後のG2 PASSを先取りしない。

対象は既存 `runs/cnp_v1/confirm/cnp003_confirm1/` のreport/data/run manifest/metricsと
実行コードのレビュー、保存集計からの算術的導出のみ。新規データ生成、モデルforward、
学習、checkpoint選択、適応、旧sealedアクセスはすべて0。
実行コード14ファイルのSHA256はrun manifestと全件一致した。
入力hashと全seedの集計は[レビュー監査JSON](CNP003_RESULT_REVIEW_AUDIT.json)に保存した。

## 2. 保存結果

下表は各seedの既知長5×閾値5の25セルにおける最小値。各列の最小セルは同一とは限らない。

| seed | balanced accuracy | 集合F1 | 見本別F1 p10 | 記録上の最小因果gap | G2失敗セル |
|---|---:|---:|---:|---:|---:|
| 610200 | 0.967740 | 0.955855 | 0.927604 | 0.158325 | 25/25 |
| 610201 | 0.970666 | 0.960042 | 0.934896 | 0.159424 | 25/25 |
| 610202 | 0.971642 | 0.957024 | 0.932552 | 0.160034 | 25/25 |
| 610203 | 0.970014 | 0.955169 | 0.934896 | 0.161499 | 25/25 |
| 610204 | 0.973722 | 0.960932 | 0.937583 | 0.161255 | 25/25 |

通常品質の3条件は全125セルで基準を満たす。記録上のWrong-argument gapは125/125セル、
None gapは100/125セルで0.50未満。Wrong-family gapと記録上のCorrect accuracyは
全セルで0.95以上。ただし後者は以下の不正な共通分母での値である。

LEARNED_METRICは257重みで最悪balanced accuracy=0.953387、F1=0.921131。
MLPは8,449重みで、これより高い最悪セル品質を記録した。UNCONDITIONEDの各seedの
最悪balanced accuracyは0.656220～0.662071。条件を渡す価値を示唆する比較ではあるが、
介入別因果検証の代用にはしない。METRICは固定単位行列初期化・同一データで全seedの
評価値が同一であり、5回の独立な初期化再現性とは数えない。

長さ32でのMLPの集合EMは全seed・閾値を通じて0.397705～0.642090。
二回SELECTの最悪集合F1はseed別0.917827～0.929372。
要素単位の高精度から、長い集合の完全一致や正確な記号実行を主張できない。
局所適応・保持・費用利益（H-CNP2/3/4）は未検証。

記録された総時間は5,945.69秒（約99.1分）、peak CUDA 68,528,640 bytes、
process RAM 1,974,112,256 bytes。実行commit、config hash、資源の原値は監査JSON参照。
これはCNP-003全体の測定であり、CNP-004の方式別総費用比較ではない。

## 3. 主要な問題: 因果評価の分母が契約と不一致

[設計§5](../design-docs/CONDITIONAL_NEURAL_PRIMITIVES.md)は
「因果gapは参照出力が実際に反転する有効要素だけを固定して」と規定する。
[計画§3.2](../exec-plans/active/CONDITIONAL_NEURAL_PRIMITIVES.md)は
介入ごとのeffectful集合数を要求する。

しかし [CausalAccumulator.add](../../src/apc/cnp/development.py) は次を全対照で共有する。

```text
changed = valid & ((target != wrong_family)
                 | (target != wrong_argument_prediction)
                 | (target != none))
```

Wrong-familyは正解選択の補集合なので、その項だけでほぼ全有効要素が対象になる。
実際、保存された全方式・全セルで `effectful_items == valid_items`、Wrong-familyの
元正解に対するaccuracyは0である。さらにWrong-argumentの集合定義へ学習器の予測を
使っており、計算済みの `controls.wrong_argument.selected` を渡していない。
[confirmation runner](../../src/apc/cnp/confirmation.py)と開発runnerの双方が該当し、
[evaluation helper](../../src/apc/cnp/evaluation.py)にも同じ和集合パターンがある。

### モデルを改善しても現行判定を通せない証明

全要素が分母で、Noneは全要素を選ぶ。セルの正例率をpとするとNone accuracy=p。
従って `Correct accuracy - None accuracy <= 1-p` であり、完全正解でもこの上限を超えない。

| 閾値 | 既知長5セルの因果gap上限の範囲 | 0.50達成可能なセル数 |
|---|---:|---:|
| 0.50 | 0.584839～0.596191 | 5/5 |
| 0.65 | 0.452377～0.460938 | 0/5 |
| 0.80 | 0.338379～0.348145 | 0/5 |
| 0.95 | 0.246155～0.254211 | 0/5 |
| 1.10 | 0.175293～0.187012 | 0/5 |

25セル中20セルはどのモデルでも現行None条件を通過できない。追加学習、層の拡大、
seedの追加では解消しない。Wrong-argumentも、別qで参照ラベルが変わらない要素まで
含むため、正しく条件に従う出力が「元の正解と一致してしまう」としてgapを薄める。
そのoracle gap自体は保存集計から復元できず、本レビューでは測定していない。

Noneについてのみ、保存された通常accuracy a、balanced accuracy b、正例率pから
`TNR = (a - 2*p*b)/(1 - 2*p)` と復元できる。全125セルで整数のtrue-negative件数に
復元できることも確認した。None固有のeffectful要素（正解が非選択の要素）上の
Correct accuracyの最小値は、seed順に0.972181 / 0.966882 / 0.960306 / 0.968869 /
0.955787。そこでNone accuracy=0なのでgapも同じ値になる。
この部分的な復元は分母問題の証拠であり、Wrong-argument条件や集合数を含む
G2全体の修正再判定ではない。

## 4. 併せて補うべき証拠

1. **介入別の件数監査。** 現在の全体件数・共通セル件数では、各介入で必要な
   全体256集合／セル32集合を満たすと証明できない。gateは要素数を見ているが、
   計画§3.2の件数は集合単位。介入ごとに要素数・集合数・母集団比率を分けて残す。
2. **不確実性と副次出力。** 集合単位1,000回bootstrapのEM CIが保存reportにない。
   COUNT/SUM_FIRSTは二回SELECT後の集計で、計画が要求する単段SELECT後の
   パネルは別に保存されていない。既存の二段値を単段値として扱わない。
3. **復元のdevice。** fresh-loadは5seedともPASSだが、実装はCPUロード同士を
   sourceの一つの小例で比較する。学習・正式評価deviceであるCUDAからの保存と
   別プロセスCUDA復元の完全一致まで証明したとは言えない。

これらを理由に通常精度の保存値を消さず、証拠の有効範囲を限定する。
ADR-0190の「implementation errorではない」という解釈とeffectful/fresh-loadの
全面的PASS解釈には、本レビューを追記による訂正として適用する。

## 5. 改善の順序

**最優先は評価器の修正であり、モデル変更ではない。** 次の独立した修復契約を推奨する。
本レビューはその実装・再評価を実行せず、研究実行の承認にも代用しない。

1. 介入cごとに `E_c = valid & (reference_correct != reference_c)` を固定する。
   Correctと介入予測を同じE_c上で元正解と比較し、Correct≥0.95、gap≥0.50を
   介入別に適用する。Wrong-argument集合は必ず参照のwrong q出力から決める。
   学習器の誤り、seed、モデル方式で分母が変わらないようにする。
2. 手書きの不均衡ラベル例で、完全な参照モデルが各介入でgap=1になることを検証する。
   Wrong-argumentの一部だけがeffectfulな例、効果ゼロならINCONCLUSIVE、
   多数要素を含む少数集合では集合数要件を満たさない例、paddingを含める。
   分母の共有を再導入した場合に失敗する回帰テストにする。
3. 修正コード・全対象seed/checkpoint・分割・閾値・出力schemaを先に固定する。
   保存済みの全5seed最終checkpointと比較方式を使う、学習0の訂正再評価を別途定義する。
   元のパネルは新しい確認集合とは呼ばず、既開封の同一パネルの訂正解析と明記する。
   旧runは上書きせず別namespaceへ保存し、通常予測・通常指標のparityを確認する。
   介入別件数、集合単位CI、単段COUNT/SUM、CUDA fresh-loadもここで検証する。
4. 訂正評価後にのみ残る性能不足を判断する。Wrong-argumentのCorrect floorが
   不足するなら、境界付近の分類誤りを開発データで分析する。モデル変更が必要な場合の
   候補は、学習距離 `d_theta(x,q)` と `logit=s*(tau-d_theta)` を分けて閾値への単調性を
   保つ方式。ただし既存LEARNED_METRICが最小の比較対象であり、新方式の優位は未証明。
   v2の独立計画・未使用確認データで検証し、確認結果に合わせた閾値緩和をしない。

長い集合の完全一致を改善したい場合も、まず要素誤差と集合長の関係を分けて報告する。
F1が高いことだけで合成の完全性を主張せず、MLP対257重みMETRICの品質・容量・総費用を
同じ条件で判断する。現時点でadapter、追加family、モデル拡大を優先する根拠はない。

[計画§5](../exec-plans/active/CONDITIONAL_NEURAL_PRIMITIVES.md)の
「G0/G1/G2の失敗後は依存段階へ進まない」と
「保存済み確認結果を見た後の変更はv2という別計画」に従い、CNP-004は停止を保持する。
評価器訂正による過去パネルの再分析と、新モデルの独立確認は別の成果として扱う。

## 6. 訂正解析の実行結果

2026-09-21、ユーザーの明示指示により、上記1〜3を実施した。
評価器は介入 `c` ごとに `E_c = valid & (reference_correct != reference_c)` を使い、
Wrong-argumentの分母もreferenceのwrong q出力だけで固定する。手書き不均衡例の回帰テストは、
完全Correctで各介入gap=1、モデル予測を変えても分母が変わらないことを検証する。

新しい訂正runは
`runs/cnp_v1/correction/cnp003_correction2/`。元の
`runs/cnp_v1/confirm/cnp003_confirm1/` は一切上書きしていない。
元runの20 checkpoint、MLP bundle 5個、元report/data/run manifestの計33ファイルをSHA256で
入力固定し、final checkpointだけを読み込んだ。学習step/optimizer stepはともに0、
追加seed・checkpoint選択・閾値変更・旧sealedアクセスは0である。CNP確認パネルは
既開封の同一panelを決定論的に再生成してdata manifestの完全一致を確認した。

| 判定 | 結果 |
|---|---|
| 20 checkpointの通常指標parity | PASS（全20） |
| MLP 5 seedの訂正G2 | PASS（5/5） |
| MLP CUDA別プロセスfresh-load | PASS（5/5、logit/mask完全一致） |
| 各対照の最小effectful集合数 | Wrong-family 4,096、Wrong-argument 968、None 766 |
| MLP最小Correct accuracy | Wrong-family 0.969727、Wrong-argument 0.966832、None 0.955787 |
| MLP最小causal gap | Wrong-family 0.969727、Wrong-argument 0.930970、None 0.955787 |

通常品質の最小値は既存値と同じで、balanced accuracy=0.967740、集合F1=0.955169、
見本別F1 p10=0.927604。集合EMの最小値は0.645020であり、集合単位1,000回bootstrapの
セル別95% CIを保存した。単段SELECT後のCOUNT一致最小値は0.659180、SUM_FIRST MAE最大値は
0.193415である。これらは選択maskの因果G2とは別の副次結果であり、完全な記号実行の主張には
使わない。訂正runは4,309.30秒、peak CUDA 84,058,624 bytes、process RAM 1,562,157,056 bytes。

訂正した測定上、H-CNP1は `SUPPORTED_ON_ALREADY_OPENED_PANEL_ONLY` である。これは
元の`G2_FAIL_STOP`を遡及的に書き換えず、新しい未使用panelでの確認PASSとも扱わない。
CNP-004、candidate selection、promotionは`NOT_EXECUTED`のまま維持する。
Wrong-argumentのCorrect floorとgapは十分なので、モデル拡張や学習距離への置換を進める
根拠は得られなかった。長い集合のEM/terminalの限界は保存し、別のv2研究課題が明示されるまで
最適化対象にしない。

## 7. 検証

Python 3.12.13の標準ライブラリだけで保存JSONを再集計し、入力hash、14ソースhash、
全125セルの通常品質、因果gap失敗件数、None gap上限、整数混同行列の復元を確認した。
モデル・generator・optimizerはimportしていない。文書リンク・JSON整合性・diffを検査。
コード変更がないためpytest/ruff/mypyと研究実験は未実行。過去の全体pytestに関する
artifact欠損を今回のPASSとは扱わない。

この第6節の実装変更では、WSL Python 3.12.14でCNP対象21 tests、`ruff check`、
`mypy src/apc`をPASSした。`pytest -q --maxfail=1`は961件PASS後、CNP外の
`tests/test_mirror_attention_clamp_causal_replay.py::test_initial_parity_gate_and_fused_qkv_freeze`
で、既存`runs/phase_b_b2_model_bundle_recovery/staging/seed_10/rec004/core/shared_encoder.pt`
の欠損により停止した。全pytestはclean PASSではない。
