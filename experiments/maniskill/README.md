# ManiSkill実験ワークスペース

**現在:** 環境の起動・試行・記録と、Fetchの最小目標到達課題を実装。
Windowsの専用venvでCPU実行を確認した。APCの学習器やスキル銀行は未実装。
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
学習済み方策・スキル再利用の成果とは呼ばない。

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
「試行→保存→集計」と「Fetch到達課題→閉ループ制御→成果物」に各1本。

```bash
APC_RUN_MANISKILL_TEST=1 python -m pytest -q tests/test_rollout_feature.py tests/test_fetch_reach_feature.py
```

通常実行では明示的にskipされる。skipは成功ではない。成功率の最低値は検査しない。
GPU・Fetch・動画は研究用マシンで短い実runを行い、結果を実験メモへ記す。
巨大な全件テストや旧APCの回帰スイートはこのトラックの前提にしない。

## 5. 次の作業

[実験メモ](docs/ITERATION_LOG.md)にPanda/Fetchと到達課題のCPU実runを記録した。
次は同じ身体・行動対応のまま、小規模な模倣学習baselineを作り、
学習した方策自身の閉ループrolloutを手設計制御と比較する。
GPU物理・動画・学習は未検証。

## 上流資料（2026-09-22確認）

- [S1: インストール・対応OS](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [S2: PyPI ManiSkill 3.0.1](https://pypi.org/project/mani-skill/3.0.1/)
- [S3: Quickstart・Fetchへの身体変更](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)
- [S4: RecordEpisode](https://maniskill.readthedocs.io/en/latest/user_guide/wrappers/record.html)
- [S5: v3.0.1 Fetchコントローラ](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/agents/robots/fetch/fetch.py)
- [S6: v3.0.1 BaseEnv・描画無効化](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/sapien_env.py)
- [S7: 上流RL導入](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html)

`latest`文書は変化する。実行条件はそのrunのインストール版と成果物を優先する。
