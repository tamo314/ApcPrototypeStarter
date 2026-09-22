# ManiSkill実験ワークスペース

**現在:** 環境の起動・試行・記録と、Fetchの最小目標到達課題を実装。
Windowsの専用venvでCPU実行を確認した。台車の小規模な模倣学習baselineも実装済み。
同一sceneで単一方策を2つの目標に順次使う手動連鎖も実装済み。
APCの自動スキル銀行・追加学習/圧縮は未実装。
ランダム方策の成功率を研究仮説の成否とみなさない。
方針は [AGENTS.md](AGENTS.md)、研究の順序は [計画](docs/RESEARCH_PLAN.md)、
データ形式と将来の接続は [設計](docs/ARCHITECTURE.md) を参照する。

## 1. セットアップ

ネイティブLinux + NVIDIA GPUを主対象とする。CPU物理・状態観測・描画なしから開始できる。
GPU物理とVulkan描画は別機能であり、CUDA検出だけでは動作保証にならない。
WSLは上流の対応表ではCPU物理のみ。GPU物理・動画はネイティブLinuxで試す。[S1]

リポジトリ直下から:

```bash
bash experiments/maniskill/scripts/setup.sh
# Python 3.11を使う場合
# PYTHON=python3.11 bash experiments/maniskill/scripts/setup.sh
```

スクリプトはこのフォルダの `.venv` にインストールし、旧APC環境には触れない。
Python本体、GPUドライバ、システムライブラリは自動インストールしない。
Ubuntuで動画を使う際の追加設定例は `sudo apt-get install libvulkan1 vulkan-tools`。
GPUドライバはホストの管理方針に従う。[S1]

ManiSkillは確認した安定版 `3.0.1` を直接固定する。[S2]
PyTorchは使用GPUに適したwheelを選ぶ。既定はPyPIのtorchだが、必要なら
`TORCH_SPEC` と `TORCH_INDEX_URL` を指定する。例のCUDAバージョンを無検証で固定しない。
解決された全依存はsetup/runの `requirements.freeze.txt` に保存する。
これは**そのマシンでの解決記録であって、検証済み共通lockではない**。
安定した実行構成が得られたら、その記録を基にconstraintsを残す。

アセットは `.assets/` に分離する。全データセットは取得しない。対象に必要なもののみ:

```bash
cd experiments/maniskill
source .venv/bin/activate
# Windows の Git Bash では代わりに: source .venv/Scripts/activate
export MS_ASSET_DIR="$PWD/.assets"
# Fetchなどで不足を指摘された場合は上流ダウンローダを使う
python -m mani_skill.utils.download_asset --help
# 指定タスクの追加アセットがある場合:
# python -m mani_skill.utils.download_asset OpenCabinetDrawer-v1
```

不足アセットの対話プロンプトを残している。無人実行する際は、先に必要な取得を済ませる。
非商用研究を前提に利用するが、出所・クレジットとアセット利用条件は記録する。

## 2. まず環境を動かす

リポジトリ直下から:

```bash
bash experiments/maniskill/scripts/run_iteration.sh
bash experiments/maniskill/scripts/run_iteration.sh experiments/maniskill/configs/fetch_pickcube.json
```

最初のPanda/PickCubeはインストールと記録の確認用で、移動能力の研究成果ではない。
Fetch/PickCubeは移動可能な身体を使う観察用設定。Fetchの同制御モードには腕・
グリッパ・胴体・台車の制御が含まれる。[S3, S5]
目的地点への移動には自作の `APC-FetchReachGoal-v1` を使う（次節）。
PickCubeを移動課題と同一視しない。

設定を変更して実行する例（以下はこのフォルダで）:

```bash
source .venv/bin/activate
# Windows の Git Bash では代わりに: source .venv/Scripts/activate
export MS_ASSET_DIR="$PWD/.assets"
python -m apc_maniskill doctor
python -m apc_maniskill rollout --config configs/fetch_pickcube.json --seed 1000 --episodes 5
# GPU物理を明示的に試す。初版ではGPUでも単一環境のみ。
python -m apc_maniskill rollout --config configs/fetch_pickcube.json --sim-backend physx_cuda
# 動画にはVulkanが必要。観測入力そのものはstateのまま。
python -m apc_maniskill rollout --config configs/pickcube_cpu.json --video
python -m apc_maniskill summarize runs/<run-id>
```

`run_iteration.sh` は試行とsummary保存を一度実行して終了する。常駐・自動再試行しない。
CLI直実行はsummaryを標準出力に表示する。保存する場合はリダイレクトする。
`--out` は存在しない新規ディレクトリのみ受け付ける。再実行は新しいrunを作る。

### Fetchの最小目標到達課題

```bash
python -m apc_maniskill rollout --config configs/fetch_reach_goal.json
python -m apc_maniskill rollout --config configs/fetch_reach_goal.json --policy fetch_zero
python -m apc_maniskill rollout --config configs/fetch_reach_goal.json --policy fetch_random
```

ManiSkillのBaseEnv・Fetch・床を流用した、障害物なし・単一環境の課題。
開始xyは±0.2 m、yawは±0.4 rad、目標は開始位置から世界座標で
x方向0.5〜1.0 m、y方向±0.4 m。state観測に目標、身体座標の相対目標、台車姿勢/速度を含む。
報酬は毎stepの距離改善（m）。距離≤0.08 m、並進速度≤0.05 m/s、
角速度≤0.10 rad/sが同時成立すると終了し、それ以外は200 stepで時間打ち切り。

全身行動はFetch/PickCubeと同じ13次元を保つ。`fetch_*` 方策は腕・胴体を
rest keyframeへ戻すdelta指令とグリッパ0.015 m指令を送り、台車の前進速度と旋回速度を選ぶ。
`fetch_goal` は手設計の閉ループ制御、`fetch_zero` は台車ゼロ指令、
`fetch_random` は台車のみランダム指令。既存の `random`/`zero` は全身への指令で別物。
姿勢保持は物理的な固定ではなく、ずれを `posture_max_error` に記録する。
これらの手設計方策を学習済み方策・スキル再利用の成果とは呼ばない。

### 台車の模倣学習baseline

このフォルダで、毎回新しい出力先を指定する:

```bash
python -m apc_maniskill rollout --config configs/fetch_reach_goal.json --episodes 10 --seed 10 --out runs/my-demo
python -m apc_maniskill train-bc --demo-run runs/my-demo --out runs/my-bc --updates 1000 --seed 0
python -m apc_maniskill rollout --config configs/fetch_reach_bc.json --checkpoint runs/my-bc/policy.pt --out runs/my-learned
python -m apc_maniskill rollout --config configs/fetch_reach_goal.json --seed 1000 --out runs/my-teacher-comparison
```

`train-bc` は保存済み教師軌跡の行動直前の観測から、正規化された台車の前進/旋回指令を学習する。
入力は身体座標の相対目標xyと台車速度vx/vy/yaw rateの5要素、ネットワークは
5→32→32→2（Tanh、1,314パラメータ）。CPU、Adam、学習率0.001、batch 64、
均等サンプリングのMSEを使う。入力の平均/標準偏差は教師データだけで求める。
停止指令を多く学習する比較は `train-bc ... --stop-weight 4` で実行できる。
台車2出力がともに絶対値1e-7以下の教師サンプルの抽出重みを4倍にし、復元抽出する。
既定値1は元の均等抽出を維持する。記録する全データ/停止/移動のMSEは重みなしで計算し、
実際に抽出した停止サンプル数も保存する。停止指令ラベルは物理的な停止状態とは区別する。
腕・胴体・グリッパは共通の手設計姿勢補正で、台車には学習済み出力のみを送る。
推論中のteacher切替や手設計の台車停止ルールはない。

`rollout ... --post-success-steps 20` は初回到達後も同じ目標で20 step観測する
Fetch専用診断。成功の幾何条件は変えず、wrapperがその観測期間の終了までterminatedを延期する。
元の終了信号はinfoの `task_terminated` に保存し、時間制限は維持する。
summaryの成功率は一度でも成功した割合。維持を見る際は `success_final_episodes` と
`hold_complete_episodes`、毎stepのsuccessを確認する。

`configs/fetch_reach_sequence.json` は同じ方策で2目標を順番に解く手動連鎖。
最初の目標で成功すると、物理状態をresetせず目標だけを変更する。
`--next-goal-offset X Y` は最初の目標から次の目標への世界座標offset（m）。
既定は0.7/0.2、時間制限は合計400 step、最後の成功後20 stepを観測する。
checkpointは `--checkpoint` で指定し、教師比較は `--policy fetch_goal` を使う。
これは単一方策の反復使用で、自動スキル選択や銀行の増設ではない。

学習runには `training.json`、`losses.jsonl`、`policy.pt` を保存する。
データのseed/ハッシュ、更新数、パラメータ数、学習データ上のMSE、実行時間を残す。
rolloutはチェックポイントのコピー/ハッシュを保存し、タスクと制御の互換性を検査する。
新しいrunには実行コードの `source_snapshot/` も保存する。
学習損失は独立評価ではない。比較seed 1000〜1007も今回確認した探索データとして扱う。

## 3. 保存されるもの

| ファイル | 内容 |
|---|---|
| `manifest.json` | config、seed、git状態、依存、物理/制御周波数、run状態とエラー |
| `episode_XXXX.npz` | 状態観測、行動、報酬、終了理由、成功判定の時系列 |
| `steps.jsonl` | step単位の報酬・終了・数値info。試行中にもflush |
| `episodes.jsonl` | episodeのreturn、成功、打ち切り、seed、所要時間、reset/finalの数値info |
| `dependencies.json` / `requirements.freeze.txt` | pip freeze実行結果 / 成功時の解決バージョン |
| `videos/` | `--video` 時の上流RecordEpisode動画 [S4] |
| `error.txt` | 捕捉できた例外。失敗したrunも削除しない |

`success_ever` と `success_final` を分ける。terminatedを成功とみなさず、
環境のtruncatedとrunnerの予算打ち切りを区別する。成功情報がない場合はnullにする。
強制killやネイティブライブラリのクラッシュではmanifestがrunningのまま残る可能性がある。
そのrunは完了扱いしない。途中episodeはstepログのみ残り、NPZは完了したepisodeのみ。

## 4. テスト

機能のまとまりを変更したときだけ、対象の機能全体のテストを行う。
「試行→保存→集計」「Fetch到達課題→閉ループ制御→成果物」
「教師軌跡→学習→重み再読込→学習方策の実行」に各1本。

```bash
APC_RUN_MANISKILL_TEST=1 python -m pytest -q tests/test_rollout_feature.py tests/test_fetch_reach_feature.py tests/test_bc_feature.py
```

通常実行では明示的にskipされる。skipは成功ではない。成功率の最低値は検査しない。
GPU・Fetch・動画は研究用マシンで短い実runを行い、結果を実験メモへ記す。
巨大な全件テストや旧APCの回帰スイートはこのトラックの前提にしない。

## 5. 次の作業

[実験メモ](docs/ITERATION_LOG.md)にPanda/Fetchと到達課題のCPU実runを記録した。
固定した重み1/4のモデルを追加seed 1003〜1007で各5 episode比較し、両方5/5到達。
重み4では合計276→234 step、目標範囲へ入ってから終了まで67→25 stepに短縮した。
ただしseed 1003の最終角速度は重み4でも0.098 rad/sで、停止閾値付近に残る。
到達後20 step（1秒）の診断では5/5が最終的に目標範囲から外れた。
教師10 episodeに停止中の200 stepを追加し再学習すると、同じ5例で1秒間の成功条件を維持した。
resetなしの手動2目標連鎖は前方・横・戻りの各3例で完了し、最終1秒間も条件を維持した。
次は1,314パラメータの方策を別の小型candidateへ蒸留し、容量と閉ループ挙動を比較する。
GPU物理・動画・APCの自動銀行/圧縮は未検証。

## 上流資料（2026-09-22確認）

- [S1: インストール・対応OS](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [S2: PyPI ManiSkill 3.0.1](https://pypi.org/project/mani-skill/3.0.1/)
- [S3: Quickstart・Fetchへの身体変更](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)
- [S4: RecordEpisode](https://maniskill.readthedocs.io/en/latest/user_guide/wrappers/record.html)
- [S5: v3.0.1 Fetchコントローラ](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/agents/robots/fetch/fetch.py)
- [S6: v3.0.1 BaseEnv・描画無効化](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/sapien_env.py)
- [S7: 上流RL導入](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html)

`latest`文書は変化する。実行条件はそのrunのインストール版と成果物を優先する。
