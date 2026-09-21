# CNP実装・研究実行計画

日付: 2026-09-21 / CNP-000 / ADR-0187

入口: [研究方針](../../research/CONDITIONAL_NEURAL_PRIMITIVES_CHARTER.md)
→ [詳細設計](../../design-docs/CONDITIONAL_NEURAL_PRIMITIVES.md) → 本計画。

## 1. 現在地と実行範囲

- CNP-000: 方針・設計・実装順・評価条件を記録する今回の計画タスク。
- CNP-001〜005: `NOT_STARTED`。ここに記載した実験は一つも実行していない。
- この依頼ではモデル初期化・訓練・新規評価を行わない。
- 実行指示が与えられた範囲を進める。前提を通過したことだけから未指定の段階を実行しない。
  逆に、指定範囲内の通常の実装・検証・ローカルcommitには再確認を挟まない。
- `legacy_sealed_access=0`、既存bundle promotionなし、Phase B/C再開なし。
  CNP確認データの一回の開封は確認段階の明示的な実行範囲に含める。

## 2. 段階と納品物

| ID | 作業・納品物 | 前提 | 完了条件 |
|---|---|---|---|
| CNP-001 | 詳細設計の型・生成器・独立参照・モデル・baseline・executor・評価器・保存・CLI・configを実装。seed/split監査 | 実装指示 | 以下のG0、標準検証3件、未実行dry-run計画書 |
| CNP-002 | 開発2seedで固定構成の学習、baseline、時間/メモリーを測定。開発報告・失敗例分類 | G0＋開発実行指示 | G1成立、全費用記録。未成立なら依存段階STOP |
| CNP-003 | hash固定後に5seedのH-CNP1確認。一回の全パネル測定、保存・別プロセス復元 | G1＋確認実行指示 | G2の判定を記録。FAILもタスク完了の成果 |
| CNP-004 | 凍結親から4block順次適応、同情報baseline、転移・保持・合成・総費用比較 | G2＋適応実行指示 | G3/G4を別々に判定。各blockの採否・棄却証拠 |
| CNP-005 | 保存証拠から最終採否、限界、次課題の必要性を記録 | CNP-004完了または先行STOP | 仮説別支持/不支持/未検証、比較表、ADR。自動的な拡張なし |

CNP-001は大きな一括変更にせず、次の順にローカルcommitを分ける。
①contracts/data/referenceと独立テスト、②primitive/baselines/execution、
③training/evaluation/artifacts/CLI、④全検証・config固定。
この分割は実装のチェックポイントであり、新たな訓練許可の単位ではない。

## 3. 事前固定する数値

### 3.1 初期学習

| 項目 | 固定値 |
|---|---|
| 実行環境 | Python 3.12、既存pyproject制約、単一RTX 5060 Ti 16GB、float32 |
| モデル | 65→64→64→1 MLP、8,449重み。設計書の固定phi |
| optimizer | AdamW、lr=0.001、weight_decay=0、既定betas/eps、schedulerなし |
| 勾配 | L2 norm clip=1.0、AMPなし、dropoutなし |
| 初期step/batch | 4,000 steps × 32集合、各方式・各model seedで同じ順序 |
| 初期教師 | 固定64見本×3閾値から均等抽出、xと長さは設計書の分布 |
| 学習方式 | CONDITIONAL_MLP、LEARNED_METRIC、UNCONDITIONED。RAW_DISTANCEは別の小規模適合 |
| seed | 開発2、確認5。設計書の値をCNP-001で静的監査後に登録 |
| checkpoint | 初期/1,000/2,000/4,000。主結果は最終4,000のみ |

確認でハイパーパラメータ探索、最良seed選択、最良checkpoint選択をしない。
開発構成も一つ。失敗時にその場で層数・予算を増やさず、次の変更が必要なら別ADRにする。
数値は性能の予測ではなく、固定された初期実験予算である。

### 3.2 確認パネル

各モデルにつき32未使用見本×5閾値×各長さで、1セル128集合を固定生成する。
既知長 `{1,2,4,8,16}` が主判定、未使用長 `{3,12,32}` は別の外挿/補間パネル。
同じ32見本を使うが、各長さ・閾値のcontent RNGは分離。
32長の不合格を既知長の平均で隠さず、v1対応長へ勝手に含めない。

因果対照は各seedのq/threshold/lengthごとに同じcontentを使用。
None/Wrong-family/Wrong-argumentが参照上効果を持つ集合が全体でそれぞれ256以上、
長さ×閾値の各集約セルで32以上なければ因果判定は`INCONCLUSIVE`。
件数不足を埋めるために同じ確認パネルを追加生成しない。

合成パネル: 固定見本リストの循環隣接32組、既知の3閾値同士の全9組、
長さ `{4,8,16}`、各組128集合で二回SELECTを測定。
COUNT/SUM_FIRSTは同じ単段SELECTパネルの予測出力を後段に渡す。
二回SELECTはこの段階では機構の確認・記述であり、H-CNP3の更新効果はCNP-004で判定。

### 3.3 順次適応

各確認モデルのMLP親とmetric親から開始。CNP-003の確認集合そのものは再利用せず、
独立したold retention用splitをCNP-004で使う。
LOCAL/FULL_REPLAY/FULL_NO_REPLAY/SCRATCH/METRIC_REPLAYを固定順の4blockで比較する。
LOCALがshadow不合格ならそのseedの後続blockを止め、対応する他方式も同じblockまでで
主比較を終える。残りは未実行と記録し、成功seedだけで全stream平均を作らない。

- 各block: 8見本×3閾値×64教師集合 = 1,536新規集合を一度だけ生成。
- 主予算は256更新、batch32（new16＋replay16）。FULL_NO_REPLAYはnew32。
- replayは旧条件から最大1,536集合を保持。sourceと既に採用したblockから層化均等、
  端数は古いdomain順。各domain内はrecord hashの小さい順で選ぶ。全方式で同一buffer。
- source replayのラベルは初期学習済みrecordから取得。追加のoracle問い合わせをしない。
- shadowは各既知domainの独立見本8×3閾値×64集合、長さは `{1,2,4,8,16}` を均等配分。
  固定256step候補の採否だけに使い、複数候補からの選択やearly stoppingに使わない。
- 転移評価は各blockの未使用見本16×3閾値×各既知長×128集合。
  親/候補/全baselineで同じ入力と見本を使う。学習・shadowへの流用禁止。
- `0,16,64,256` stepのスコアを記録するが主結果は256。
  中間スコアによる以後の変更やbest checkpoint採用はしない。
- 教師16集合/条件の小予算armを独立cloneで一回だけ併設し、同じ最大256step。
  64集合armとデータをprefix共有し、確認結果から採用armを切り替えない。
  16集合armはsample-efficiency曲線用の副次結果で、主ゲートを救済しない。

LOCALのoptimizer設定は初期学習と同じ、trainableはadapterのみ。
FULL/metric/scratchも同じ設定・step上限とし、計算量の違いは実測する。
各方式についてparent→candidateの保持も記録する。
baseline自身のshadow不合格も隠さず、LOCALと同じ採用・停止規則で運用結果を報告する。
block開始時、各方式の親が新domainのshadowで品質floorを満たす場合は更新を省略する。
確認用転移データを見てこの判断を変更しない。全方式で同じ再利用優先規則とし、
省略した学習は0step、必要なshadowの費用は計上する。
更新した場合は256step候補一つだけをshadow検証し、確認用転移評価はその後に一回行う。

## 4. 判定基準

### G0 — 実装・情報・データの妥当性

必要条件をすべて満たすこと。

- 型・形状・引数の契約、oracle情報遮断、未選択forward=0、凍結重み不変のテストPASS。
- Mを知る参照実行が手書き小例と独立実装で一致。通常実行に参照fallbackなし。
- 生成器の件数・unique・正例率・空集合率・分割重複を全て報告。
  開発用の長さ×閾値セルで正例率が[0.05,0.95]を外れる場合は
  `DEGENERATE_PANEL`としてSTOP。主分布をラベルで選別して救済しない。
- 既存の全seed登録、D-018までの予約、CNP各roleと旧sealedの静的境界監査PASS。
- 標準 `pytest -q`, `ruff check .`, `mypy src/apc` PASS。
- 学習前のCPU/CUDA forwardと1step gradient wiring、別プロセスloadの確認PASS。
  これは単体検証であり、研究訓練の実行とは区別してログ化する。

G0不成立は実装/測定前提の問題。科学仮説をFAILとしない。

### G1 — 開発上の技術成立性

2seedのそれぞれで、既知長の長さ×閾値セルごとに
balanced accuracy ≥0.95、平均集合F1 ≥0.90。
集合EMを併記するが開発ゲートには使わない。G0の因果検査も維持。
開発合格はH-CNP1確認の代用ではない。

### G2 — H-CNP1の確認

5seedすべてについて、既知長の各長さ×閾値セルで次を満たす。

- balanced accuracy ≥0.95、平均集合F1 ≥0.90。
- 見本別F1の10th percentile ≥0.80（方法は線形補間で固定）。
- 参照の選択ラベルが反転するeffectful要素上、元の正解へのCorrectの正解率 ≥0.95、
  Correctと各誤条件の正解率差 ≥0.50。集合F1の自然な多数派baselineとは混同しない。
- 64→32の見本分割・閾値補間・hash境界・fresh-loadに違反なし。

集合EMとそのCIは必須報告。完全な記号処理と同等とは主張しない。
長さ `{3,12,32}` は同じ閾値で個別に判定し、主判定とは別の対応範囲を記録する。
「5seed平均が良い」は全seed条件の代替にならない。
5seedは同じデータ・同じworldに対する初期モデルの反復であって、
五つの別の生成法を検証したことにはならない。

### G3 — H-CNP2/3の確認

LOCALの主64集合armについて、各seed・各blockで以下を評価。

1. 新domainの転移見本でG2と同じbalanced accuracy/F1のfloor。
2. 各旧domain・各長さ・各閾値のF1低下 ≤0.01、かつ旧domainのG2 floorを保持。
3. 基盤・他部品のhash不変、stableは一family＋一adapter、未選択実行0。
4. 新domainのqを一段含む二回SELECTの平均集合F1 ≥0.85。
   転移見本16を循環対応させたsourceの16見本と組み、3×3閾値、長さ4/8/16、各128集合。
   新qが先/後の両順序を別集計。合成を直接訓練していないことを記録。
5. frozen parentが新domainの単段floorを満たさない場合、単段・両順序合成のF1改善が
   各長さ集約で ≥0.05。初めから親がfloorを満たすblockは `ADAPTATION_NOT_NEEDED` とし、
   更新の必要性・改善の証拠に数えない。事前の親判定にはshadowのみを使う。

各seedで少なくとも一つのblockに適応必要性があり上記改善を満たした場合だけ、
そのseedのH-CNP2/3を支持する。5seedすべての支持が確認基準。
全て親で解ける場合は再利用成功、適応仮説は`NOT_TESTED_NO_DEFICIT`。
弱点を作るために親を弱く再訓練しない。
COUNTのEM、SUM_FIRST誤差は必須の副次指標。mask性能の代用品にしない。
shadow採否は1〜3のshadow版で行い、転移queryは採否に使用しない。

### G4 — H-CNP4の費用・容量判断

まずG3と同等の品質・保持条件を満たす方式だけを比較する。
主比較はLOCAL対FULL_REPLAYおよびMETRIC_REPLAY。全方式のPareto表も出す。

- **費用利益:** 初期学習、適応、リプレイ管理、shadow、評価、保存/復元を含む
  4block実測総時間が、品質適格な両比較方式のうち安い方の80%以下。
  5seedのpaired比率中央値 ≤0.80、かつどのseedも1.00を超えないこと。
- **容量利益:** 費用利益と別判定。全blockで品質適格なタスク別SCRATCH保存との比較で
  persistent weight bytesが50%以下、かつ最安の品質適格共有方式に対して
  総時間が110%以下。optimizer/replay/親候補のピークbytesも併記する。
- 比較方式が品質未達なら、その方式に対する「同等品質で安価」の比率を作らない。
  `QUALITY_ADVANTAGE_WITHIN_BUDGET`とし、効率仮説は未確定。

source事前学習の費用はLOCAL/FULLにそれぞれ全額を計上し、共有して実行したからといって
ゼロにしない。METRICは自身の学習費用を計上。
研究の全パネル測定費用と、実用運用で必要なshadow等の費用を別欄で出し、
主総時間には前者も含める。warmupを含む全研究費用も記録。
主比率は64集合armの費用を使い、16集合armの費用は全研究費用に別掲する。
両armで共通の事前学習費用は、各armの独立運用比較には全額、実際の研究総費用には一度だけ計上。
推論速度は同期CUDA計測、warmup20、反復100、batch32、長さ16でmedian/p95を報告。
その速度差を未知ルータや全ライフサイクルの優位性に一般化しない。

## 5. 上限とSTOP

| 段階 | 学習step上限 | wall-clock上限（GPU利用時間と別に記録） |
|---|---|---|
| CNP-002 | 2seed×3方式×4,000 = 24,000、RAW適合別枠 | 全学習・評価8時間 |
| CNP-003 | 5seed×3方式×4,000 = 60,000、RAW適合別枠 | 全学習・評価24時間 |
| CNP-004 | 5seed×5方式×4block×2教師予算×256 = 51,200 | 全学習・評価24時間 |

時間は実測済み予測ではなく停止上限。先にstepか時間のどちらかが上限に達したら停止する。
VRAMの本計画上限12GiB、process RAM32GiB、CNP artifact総量20GiB。
共有環境の他プロセスを止めない。資源不足は`RESOURCE_STOP`、科学的FAILではない。
step内の例数、unique教師集合数、有効要素ラベル数、重複提示回数を別々に数える。
RAW_DISTANCEの適合は各seed1,000反復まで、同一source教師だけ、stage時間に含める。

G0/G1/G2の失敗後は依存段階へ進まない。G3のblock失敗後はそのstreamを停止し、
未実行セルは未実行として保存。G4不合格は証拠集計を完了して拡張を止める。
修正可能な不具合と科学的な不合格を区別し、どちらも証拠・ADR・ローカルcommitを残す。
保存済み確認結果を見た後の変更はv2という別計画であり、v1の再試行として成功させない。

## 6. 完了時のチェックリスト

- task_resultと各仮説の結果を分ける。過去のrecovery/research gateは変更しない。
- 実行commit、config、seed、hash、情報境界、分割、件数、全比較方式を記録。
- resident/active/temporary params、optimizer/replay bytes、peak VRAM/RAM、実時間、
  データ/ラベル/step予算、推論median/p95を記録。
- `run_manifest.json`, `data_manifest.json`, `metrics.jsonl`, `report.json`,
  `report.md`, `bundle/` を新しいrun内に保存。欠けた証拠を推定値で埋めない。
- local commitにはコード・設定・文書・小さな集計だけを含め、runsはgitignoredのまま保持。
- 実装変更は標準3検証。文書だけの更新はリンク・整合性・diffを検査。
- 最終比較が単純方式を支持した場合も、その結論を正式な成果として残す。

## 7. 次の実装依頼で使えるスコープ

> CNP-001を実施する。研究方針・詳細設計・本計画に従い、条件付き選択の型、データ、
> 独立参照、NN、比較方式、連続実行、評価・保存、CLIを実装する。静的seed監査と
> 軽量な不変条件テスト、標準検証を完了し、ローカルcommitする。
> CNP-002以降の研究学習・確認評価は実行しない。

この文面は次タスクの具体的な引継ぎであり、現在の計画タスクに対する実装指示ではない。

## 8. CNP-000計画検証記録

2026-09-21: 方針・設計・段階計画・ADR-0187を作成。
新規3文書とADR/indexの追加部分についてローカルリンク19件（ADR anchorを含む）、
コードブロックの対応、パラメータ数8,449/1,024、学習step上限24,000/60,000/51,200を検査。
条件付きNNと一部品CNPの同一性、effectful要素での因果gap、replay/prefix共有の例外、
親で足りる場合の更新省略、開発/確認/適応の情報境界を整合性レビューした。
リンク・数値検査はPASS。git diffの空白検査も実施し、最終commit前に再確認する。
研究用seed監査、データ生成、モデル初期化、学習、確認評価は未実行。
文書のみのためpytest/ruff/mypyの全suiteは実行対象外。これは実装の検証PASSを意味しない。
既存の未commit変更 `config.json` は本タスクの変更・commit対象外。
