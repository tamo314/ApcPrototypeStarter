# 実装とデータの設計

## 実装済みと未実装を分ける

現在実装されているのは `RunConfig → ManiSkill → rollout → NPZ/JSONL → summary`。
単一環境、state観測、Box行動、random/zero方策、CPU/GPUの明示選択、任意動画に加え、
`APC-FetchReachGoal-v1` と手設計の診断方策 `fetch_goal/fetch_zero/fetch_random` を実装。
台車の小規模な模倣学習baselineを `bc.py` に実装した。
GPU並列auto-reset、APCのスキル銀行、selector、追加学習/蒸留は未実装。
本体コードは `src/apc_maniskill/`。旧 `src/apc/` をimportしない独立Pythonパッケージ。

到達課題は `fetch_reach.py` にあり、ManiSkill 3.0.1のFetchとbuild_groundを使用する。
Fetch/PickCubeの `pd_joint_delta_pos` と同じ13次元を維持し、arm 0:7、gripper 7:8、
body 8:11、base 11:13をmanifestにも保存する（実際の分割は上流controllerから取得）。
台車は前進m/sと旋回rad/s（正規化行動の3.14倍）。腕/胴体はrest姿勢への誤差を
0.1で割ったdelta指令、グリッパは0.015 mを指令する。学習・教師スキル列はない。
stateは既存qpos/qvel各15要素と、goal_xy 2、relative_goal 2、base_pose 3、
base_velocity 3の計40要素。相対目標だけ身体座標、他は世界座標でyawはrad。
報酬は制御step直前から直後への距離改善。評価関数の呼出し自体は履歴を更新しない。
到達と停止の閾値、開始/目標分布、行動対応はmanifestのtaskに保存する。
reset/final infoはepisode行、毎stepの距離・速度・姿勢誤差はsteps.jsonlに残す。
姿勢誤差は台車以外の一般化座標の最大絶対誤差であり、m/radの混在した診断値。

## 最初の学習baseline

`train-bc` はcompletedな `fetch_goal` の実軌跡のみを教師にする。
NPZの `observations[:-1]` と `actions[:, 11:13]` を対応づける。
入力は相対目標xyと身体座標の台車速度vx/vy/yaw rate（計5要素）。
state40内の位置はfeature schema v1で明示し、統合テストでinfoの幾何と照合する。
教師データ由来の平均/標準偏差（下限0.05）で標準化し、5→32→32→2のTanh MLPを
均等サンプリングの行動MSEで訓練する。CPU、Adam 0.001、batch 64。
`--stop-weight`（既定1、有限値かつ1以上）で、教師の台車2出力がともに
絶対値1e-7以下のサンプルの抽出重みを変更できる。1は元のrandint経路を維持し、
1より大きい場合は重み付き復元抽出を使う。損失自体はミニバッチ内の通常MSE。
元データの停止件数、期待抽出比率、実抽出件数を記録し、学習後の全体/停止/移動MSEは
重みなしの元データ上で計算する。checkpointとpolicy_detailsにもstop_weightを保存する。
旧checkpointはstop_weight=1として読み込める。物理的な停止判定は変更しない。
学習stepは勾配更新数であり、保存軌跡を使う学習中の環境step数は0。

checkpointはstate_dictと標準化統計、feature/task/control情報、教師seedを含む。
`weights_only=True` で読み込み、タスク/制御/入力schemaの互換性と数値を確認する。
学習runの `training.json` は状態、ハイパーパラメータ、データ由来とハッシュ、
学習MSE、パラメータ数、wall timeを保持し、失敗時はerror.txtを保存する。
学習MSEは全教師データで測った当てはまりで、汎化指標ではない。

`fetch_bc` のrolloutは1回ロードしたモデルを毎step使い、台車の手設計制御に戻らない。
非台車部分のみ `fetch_zero` と同じ姿勢補正を使い、送った全身13次元行動を保存する。
manifestの `policy_details` が実際の方策とcheckpointを識別する。
task内の `policy_source` は元の診断方策の説明で、学習方策の出自はpolicy_detailsを参照する。
チェックポイントのコピーとsha256、実行ソースsnapshotをrunに保存する。
この単一方策baselineは、銀行・temporary・圧縮・スキル再利用を実装していない。

## 次に実装する経路

```text
観測 + 最終目標
    ↓
selector（初期は明示的/手動でもよい）
    ↓ skill_id + 局所目標 + 実行時間上限
skill policy(observation, local_goal, memory) → continuous action
    ↓
ManiSkillの物理step → 次の観測 → 同じskillへ戻る
    ↓ 成功 / 失敗 / 時間上限 / 切替
selectorへ戻る

必要に応じ temporary learner → separate compact candidate → skill bank
```

skillは一回の内部表現変換でなく、時間を持つ閉ループ方策。
skill選択周期と環境制御周期を分け、途中の動作修正を可能にする。
開始可能範囲・終了条件・最大時間をスキルの契約に含める。
旧PrimitiveBaseの容量/凍結管理は参考にするが、その学習済み重みを無理に転用しない。

## 最初のスキルインターフェース案（未実装）

`reset(batch_size)`, `act(observation, local_goal)`, `should_terminate(observation)` を基本にする。
記録にはskill_id、引数、checkpoint、robot、control_mode、観測schemaを含める。
共有encoderを更新するなら旧スキルへの影響を測る。小さい初版では固定encoderまたは
スキル独立MLPを選び、潜在空間の変更が見えなくなる構成を避ける。

temporaryとcandidateは別のstate_dict/保存先にする。置換前は銀行を保持し、
圧縮したcandidateを実環境で試して比較する。容量解放の主張には参照/optimizerも含む。
追加スキルの発火は初期には手動でよく、自動判定の成果とは区別する。

## データschema v1

`manifest.json` にschema_version、設定、num_envs=1、obs_mode=state、環境周波数、
action shape/bounds、開始/終了日時、完了episode数、実行status、依存/git/ホスト情報を保存。
statusはrunning/completed/failed/interrupted。completedは試行完了でありタスク成功ではない。
CPU状態観測時は `render_backend=none` を指定して、不要なVulkan依存を避ける。

各NPZは次を含む。単一環境でもManiSkillのbatch軸を保存し、勝手にsqueezeしない。

| key | 先頭軸/型 | 意味 |
|---|---|---|
| observations | T+1、数値配列 | 初期観測と各行動後の観測 |
| actions | T、数値配列 | 環境に実際に渡した行動 |
| rewards | T、float32 | step報酬 |
| terminated | T、bool | 環境の終了 |
| truncated | T、bool | 環境wrapperの時間打ち切り |
| success | T、int8 | -1: 情報なし、0: 未成功、1: 成功 |

runner独自のmax_steps打ち切りはepisode行のrunner_truncatedに記録する。
最終NPZのtruncatedを勝手に書き換えない。強化学習に取り込む際には、両方の時間制限を
扱い、真のterminalとは区別してbootstrap方針を決める。
summaryの成功率は成功情報を得たepisodeだけを分母にし、その数と欠測数も表示する。

これは簡単な状態/行動データ形式であり、ManiSkill公式のHDF5実演形式ではない。
物理エンジンの全状態を保存していないため、NPZだけで厳密な物理リプレイはできない。
必要になった時点でRecordEpisodeのtrajectory保存やstate_dictの保存を足す。
seedを記録しても異なるGPU/依存/PhysX版でbitwise再現を保証しない。
git dirty状態を記録し、新しいrollout/学習runはパッケージ内Pythonソースを
`source_snapshot/` に保存する。リポジトリ全体や依存のsnapshotではない。
configはmanifest、解決済み依存は各runの記録を参照する。

## 終了・失敗・出力保護

環境のterminated/truncatedでそのepisodeを終了し、明示的reset後にのみ次へ進む。
最終episode後にcloseし、任意の動画をflushしてからcompletedにする。
捕捉できる例外はerror.txtとmanifestに残し非ゼロ終了する。失敗runを黙って再試行しない。
途中のstepログをflushする。SIGKILL/segfaultではfinallyは保証されず、runningのまま残り得る。
保存先の既存ディレクトリは拒否し、新しいrunで再試行する。

## 依存と拡張

ManiSkill 3.0.1を起点にし、PyTorch wheelは研究用GPUに合わせる。
rootのAPC依存は変更しない。バージョン解決/実測後に互換性メモを足す。
最初からベクトル環境のpartial resetを抽象化しない。速度が必要になった時点で
ManiSkillVectorEnv/上流PPOの方式を調べ、終了・最終観測・時間制限を扱う。
その時点で「並列収集」全体の統合テストを原則1本だけ足す。

上流APIの出典は [READMEの資料](../README.md) を参照。
