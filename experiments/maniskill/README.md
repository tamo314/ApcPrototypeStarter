# ManiSkill実験ワークスペース

**現在:** 環境の起動・試行・記録と、Fetchの最小目標到達課題を実装。
WindowsとWSLの専用venvでCPU実行を確認した。台車の小規模な模倣学習baselineも実装済み。
同一sceneで単一方策を2つの目標に順次使う手動連鎖も実装済み。
別の小型ネットワークへの蒸留も実装済み。APCの自動スキル銀行・増設は未実装。
手設計の把持・運搬軌跡から操作BCを学習し、learner状態への教師再ラベルも実装した。
教師stageを物理状態へ再同期し、無効ラベルを除くと探索seedで初めて2/3目標到達。
ただし成功を1秒維持できず、未使用seedでは0/5のため操作の汎化は未達。
ランダム方策の成功率を研究仮説の成否とみなさない。
方針は [AGENTS.md](AGENTS.md)、研究の順序は [計画](docs/RESEARCH_PLAN.md)、
データ形式と将来の接続は [設計](docs/ARCHITECTURE.md) を参照する。

**2026-09-23の小操作実験:** [小さな身体操作と毎step判断：設計・実装計画](docs/PRIMITIVE_DECISION_PLAN.md)。
手先並進6・指2・管理2の10候補を実装し、固定姿勢の到達制約を受け回転6候補を追加した。
同一実行器上の手設計selectorは探索seed 1300〜1302で把持3/3、目標到達後20 step保持2/3。
8,336パラメータの小型MLPは同seedの単独rolloutで成功0/3。1-round DAggerとID抽出比率変更でも
成功0/3のため、現在は学習selectorの閉ループ判断が明確な障害。詳細なrun・費用・留保は
[実験メモ](docs/ITERATION_LOG.md) に記録した。未使用seedの独立評価と20候補の台車操作は未実施。

**同日の先行計画改訂:** [APC実現性の学習計画](docs/RESEARCH_PLAN.md) に、
操作BCの原因分析、最小スキル銀行、temporaryでの能力追加、小型candidateへの蒸留、
解放後の再利用・過去能力保持と対照実験をまとめた。優先1の操作診断と小比較を実施済み。

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

### WindowsのCPU逆運動学とPinocchio（2026-09-22調査）

ロボティクス用Pinocchioのpip配布名は **`pin`**、import名は `pinocchio`。
PyPIの同名 `pinocchio` は別のテスト用パッケージなので使用しない。[S8, S9]
現在のWindows/Python 3.12のvenvでは、PyPIの全リリースにWindows用wheelがなく、
`pip install --dry-run --only-binary=:all: pin` も該当配布なしで失敗した。
最新版4.1.0にもCPython 3.12のLinux/macOS wheelのみ。Python版の変更だけでは解消しない。

- **Windowsを維持:** 既存 `.venv` を残し、別のConda環境を作る方法を推奨。
  conda-forgeにはwin-64/py312の3.8.0と4.1.0の配布を確認した。[S10]
  まず3.8.0を互換性確認の候補にする。SAPIENとの動作確認は未実施。
  Condaは通常のvenvとは別方式で、DLL依存を含む別環境として扱う。
- **venvを維持:** Linux側にPython 3.12の新しいvenvを作り、`python -m pip install pin`
  で依存を解決するのが公式の配布経路。[S8] Windowsのvenvは流用しない。
- **現在のWindows venvへ直接導入:** ソースビルドの検討が必要。
  Boost.Python/EigenPy等のネイティブ依存・Python ABI・DLL探索を揃える作業になり、
  最初の実験再開手段としては優先しない。ビルド可能性は未検証。

Conda導入後、このフォルダでの解決確認案（未実行、環境を作成しないdry-run）:

```powershell
conda create --dry-run --prefix ./runs/pinocchio-conda-check -c conda-forge --strict-channel-priority python=3.12 pinocchio=3.8.0 pip
```

実導入時はCondaで依存を解決した後、このワークスペースのManiSkill/PyTorchを導入し、
`pip check` → `import pinocchio` → FetchのFK/IK probe → 短いrolloutの順に確認する。
既存NumPy 2.5.3を含む全依存の完全一致・Pinocchio/SAPIENの互換性は未検証。
Condaの解決結果とpipのdry-runを確認してからその環境用のfreezeを保存する。
このWindows調査ではパッケージの導入・更新は行っていない。調査成果物は `runs/pin-install-review-20260922-a/`。

### WSL2の専用venv（2026-09-22実行）

Ubuntu 26.04 / Python 3.12.14で、`/home/tamot/.venvs/apc-maniskill-wsl-py312`
を作成した。Windows側の `.venv` とは別環境で、前後のWindows依存freezeは一致した。
Pinocchio 3.8.0のimportと、既存Fetch URDFを使うSAPIENのFK/IK計算は成功した。
当初は描画デバイスエラーで停止したが、リポジトリ内の描画なし互換修正により、
**Fetch/PickCubeと最小目標到達課題のCPU rolloutが成功**。GPU描画は引き続き利用できない。

同じ構成の導入手順（WSLのbashで実行。Python 3.12とuvは既存のものを利用）:

```bash
cd /mnt/c/Work/ApcPrototypeStarter/experiments/maniskill
uv venv --python 3.12 --seed "$HOME/.venvs/apc-maniskill-wsl-py312"
source "$HOME/.venvs/apc-maniskill-wsl-py312/bin/activate"
python -m pip install 'torch==2.14.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-wsl-py312.txt -e '.[dev]' 'pin==3.8.0'
python -m pip check
python -c 'import pinocchio, sapien; print(pinocchio.__version__)'
export MS_ASSET_DIR="$PWD/.assets"
python -m apc_maniskill doctor
# 環境内FK/IKと5 stepの診断。毎回新しいrun名を指定する。
python scripts/probe_fetch_ik.py --out runs/my-wsl-fetch-ik-probe
python -m apc_maniskill rollout --config configs/fetch_reach_goal.json --episodes 3 --out runs/my-wsl-reach
```

作成済み環境を使うときはactivateから始める。Windowsの `.venv/Scripts/python.exe` を
WSLのPythonとして利用しない。既存シェルスクリプトを使う場合も
`VENV="$HOME/.venvs/apc-maniskill-wsl-py312"` を明示する。

`pin==3.8.0` だけでは新しいurdfdom/tinyxml2が選ばれ、`pip check` が通っても
`liburdfdom_sensor.so.4.0` / `libtinyxml2.so.10` が見つからずimportに失敗した。
上記constraintsは実測した組合せであり、あらゆるタスクの動作保証や完全lockではない。
全依存・ログは `runs/setup-wsl-20260922-a/`〜`setup-wsl-20260922-c/` に保存。

FK/IK単独確認ではFetchのrest姿勢から手先を上へ2 cm移す目標を解き、位置誤差は
約0.093 mm、腕以外の関節変化は0だった（`runs/wsl-pinocchio-fk-20260922-b/`）。
これは運動学のみで、物理step・接触操作の実績ではない。
`render_backend="none"` のFetch/PickCubeでも、SAPIEN 3.0.3のURDF読込が
`RenderMaterial()` を生成して `failed to find a rendering device` となった。
`runs/wsl-fetch-ik-20260922-a/` に0 episode / 0 stepの失敗を保存。
描画あり（`--video`）も `runs/wsl-fetch-render-probe-20260922-a/` で試したが、
RenderSystem生成時に `vk::createInstanceUnique: ErrorIncompatibleDriver` で停止した。
描画許可への切替だけでは解消せず、現在のWSLのVulkan描画経路が障害となっている。

修正は `src/apc_maniskill/headless.py`。runnerが動画なしで環境を生成する際に有効にする。
ManiSkill 3.0.1 / SAPIEN 3.0.3に限定したプロセス内の互換処理で、別バージョンでは明示的に停止する。
`scene.can_render()` がfalseの場合だけ、URDFの視覚要素と立方体・球の描画材質を省く。
URDFの慣性・衝突、actorの物理形状・body type・初期姿勢は上流処理を使用する。
環境のreset/reconfigure後にも有効。描画可能sceneは元の関数に委譲し、venv内のファイルは編集しない。
確認範囲はPanda/FetchのPickCubeとFetchReach。任意タスクの全描画処理を無効化する仕組みではない。

WSLの環境内FK/IKは成功（位置誤差約0.093 mm）し、5 stepを実行した。
FetchReachはseed 1000〜1002の3/3到達・129 step、Fetch/PickCubeのランダム方策は
3 episode / 150 stepを完了（把持・配置成功0/3）。保存先はそれぞれ
`runs/wsl-fetch-headless-ik-20260922-a/`、`wsl-headless-reach-20260922-a/`、
`wsl-headless-pickcube-20260922-a/`。物理の落下・接触・再生成も既存統合テストで確認した。
Windowsでの修正前後の同seed比較では、3 episode / 150 stepの全保存配列が完全一致した。

### Fetchの腕IKと接触診断（WSL）

同じWSL venvを有効化し、このフォルダで実行する:

```bash
# 世界座標で手先を上へ2 cm動かす。3 episode・各50 step。
python scripts/run_fetch_arm.py --out runs/my-arm-track
# 接近→把持→目標へ運搬→到達後1秒観測。毎回新しい出力先を指定。
python scripts/run_fetch_arm.py --protocol pick_place --pitch-deg 15 --grasp-height .012 --torso-ik --table-clearance --absolute-static --post-success-steps 20 --max-steps 350 --out runs/my-arm-pick
```

`arm_ik.py` はSAPIENのCPU Pinocchio IKを毎step呼び、実測関節角との差を
既存13次元 `pd_joint_delta_pos` へ変換する手設計方策。世界座標の目標はロボットroot座標へ変換する。
台車にはゼロ速度指令、頭には初期姿勢保持を送る。台車の物理固定ではなく、接触で移動し得る。
`--torso-ik` なしでは胴体高さも保持する。指定時は昇降関節もIKで動かし、
不収束/可動域外/机距離の不適合時には高さ0.1/0.2/0.3/0.386 mで腕IKを解き直す。
可動域内の解から現在の関節値との差のノルムが最小のものを選ぶ（m/radの単純な混合指標）。
全候補が失敗したstepでは腕・昇降へゼロ差分を送る。
`--table-clearance` は既存の衝突メッシュと上流FKを使い、9個の腕/指リンクと机の
軸平行境界箱の分離距離が2 mm以上ある候補を選ぶ。現在から目標関節値までの12個の
補間点を確認し、腕・胴体の差分全体を同じ比率で縮小して1 step最大0.03（rad/m）とする。
有限個の姿勢だけの保守的な確認で、連続衝突判定・自己衝突・指開閉中の全経路の保証ではない。
実行中の接触力も毎制御stepで記録する。

`track` は `--offset X Y Z`（既定0/0/0.02 m）の目標を保持する。
`pick` は物体の初期位置の12 cm上→初期位置+grasp-height→閉じる15 step→15 cm上→保持。
`pick_place` はさらに、測定した手先と物体の相対位置を使って目標位置へ運ぶ。
元の目標条件を満たせば途中段階からも保持へ移る。位置5 mm・姿勢0.05 rad以内で
移動段階を切り替える。姿勢pitchの既定90度/高さ0は旧診断を再現する設定で、
上の成功例は15度/12 mm。全て手設計の段階列であり、学習済み操作方策ではない。
スクリプトは指定したmax-stepsを環境の時間制限にも設定するので、既定PickCubeの50 stepとは区別する。

manifestに方策条件、steps.jsonlの `info.diagnostic` にIK解/誤差/可動域、物理手先姿勢、
段階番号、物体高さ、指と物体・各リンクと机の接触力を保存する。
`--absolute-static` は既存PickCubeを継承した `APC-FetchPickCube-v1` を選ぶ。
上流3.0.1 Fetchの `is_static` が符号付き速度を比較する問題を実験側で補正し、
指以外の身体関節の絶対速度≤0.2、台車関節≤0.05、物体の目標距離≤0.025 mを成功条件にする。
元の成功/停止判定は `upstream_success` / `upstream_is_robot_static` にも保存する。
reset直後に残る前episodeの接触フラグも補正し、`upstream_is_grasped` に元の値を残す。
既存のPickCubeやインストール済みパッケージは変更しない。
`--post-success-steps 20` は既存の継続診断を使い、最初の成功から20 step観測する。
観測期間にsuccessが再びfalseになる可能性も保存する。summaryのsuccessは物体目標条件で、
手先到達や手放し配置の成功とは区別する（今回は把持したまま目標へ運ぶ）。
runnerの `external` はPython APIの `policy_factory` 併用専用であり、通常のrollout CLIからは選ばない。

前回の机接触による停滞を修正し、補正した成功条件でseed 0〜2と1000〜1002の6例が到達。
追加の保持診断ではseed 1000〜1002の3例とも全20 stepで把持・成功を維持し、最終目標距離は
約0.7〜1.4 mm。修正後のこれらのrunでは全記録stepで机接触力0だった。広い配置や連続時間の
無衝突保証ではない。今回の改良は失敗条件を含め24 episode / 3,471 step、学習0。
詳細は `runs/wsl-arm-place-hold-20260922-b/audit.json` と実験メモを参照する。

### 操作の小規模な模倣学習baseline

WSLの同じ環境で、毎回新しい出力先を指定する:

```bash
# 手設計教師。成功例だけを選ぶかどうかは学習runに記録する。
python scripts/run_fetch_arm.py --protocol pick_place --pitch-deg 15 --grasp-height .012 --torso-ik --table-clearance --absolute-static --post-success-steps 20 --max-steps 350 --episodes 10 --seed 20 --out runs/my-operation-demo
python -m apc_maniskill train-operation-bc --demo-run runs/my-operation-demo --successful-only --updates 3000 --out runs/my-operation-bc
python -m apc_maniskill rollout --config configs/fetch_pick_bc.json --checkpoint runs/my-operation-bc/policy.pt --out runs/my-operation-rollout

# learnerが実際に訪れた状態で既存IK教師の行動を別ラベルとして保存する。
python -m apc_maniskill collect-operation-dagger --checkpoint runs/my-operation-bc/policy.pt --episodes 3 --seed 20 --out runs/my-operation-relabel
python -m apc_maniskill train-operation-bc --demo-run runs/my-operation-demo --successful-only --extra-demo-run runs/my-operation-relabel --exclude-grasp-open-conflicts --exclude-invalid-ik-labels --updates 3000 --out runs/my-operation-dagger

# 元runを変更せず、stage偏り、物理矛盾、近傍行動差、checkpoint誤差を解析する。
python -m apc_maniskill analyze-operation-data --demo-run runs/my-operation-demo --relabel-run runs/my-operation-relabel --checkpoint runs/my-operation-dagger/policy.pt --out runs/my-operation-analysis
python -m apc_maniskill analyze-operation-rollouts --run-dir runs/my-operation-rollout --out runs/my-operation-rollout-analysis
```

入力は上流Fetch/PickCubeのstate54全体、ネットワークは54→64→64→11のTanh MLP
（8,395パラメータ）。arm/gripper/bodyの正規化行動0:11を学び、教師で常にゼロだった
base 11:13はゼロ固定する。推論中にIKや手動stageへ切り替えない。
`--successful-only` は手設計runの成功episodeだけを選び、失敗run自体は削除しない。
学習記録では全入力episodeの実収集費 `source_environment_steps` と、成功選択・ラベル除外前の
`selected_source_samples_before_filter` を分ける。episodeログと選択NPZのhashを保存し、
`success_ever` / `success_final` がNPZのsuccess列と一致することを読込時に検査する。

`collect-operation-dagger` のNPZ `actions` はlearnerが物理環境へ送った行動のまま。
`steps.jsonl` の `info.diagnostic.teacher_action` に同じ状態での手設計教師ラベル、
`behavior_action` に送信行動を保存する。教師stageは各step開始時の物理状態へ再同期する。
再ラベルrunを独立評価とは呼ばず、学習データとして扱う。

最初の実測では手設計10 episodeの8/10が成功。成功8軌跡だけの通常BCは、同じ
seed 20〜22でも0/3成功・把持0/3だった。1回目の再ラベル1,050 stateを加えると
0/3成功のままだが3/3で把持・持上げまで進み、最短物体-目標距離は約4.7〜6.3 cm。
2回目を単純に追加すると把持1/3へ退行したため、再ラベル回数を増やし続けていない。
全て探索済みseedであり、学習済み配置・汎化・自動selectorの実証ではない。

退行診断では、2回目DAggerの1,050状態が全て接近stageの開放指令で、うち756状態は
既に把持済みだった。教師stageを把持・物体高さ・目標距離へ再同期し、無効IKラベルを
除いてもう1回だけ収集すると、seed 20〜22で把持・持上げ3/3、目標到達2/3へ改善した。
ただし到達2例は初回成功を含む21観測中、成功条件が5件/4件だけで、連続維持0/2、
最終状態も失敗条件へ戻った。固定checkpointを未使用seed
1200〜1204で評価すると目標到達0/5、把持・持上げ2/5。同じ手設計教師は2/5成功・最終成功を
維持した。探索条件での改善と未使用条件での汎化を区別し、追加再ラベルはここで止めた。

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

### 別の小型方策への蒸留

```bash
python -m apc_maniskill distill --teacher-checkpoint runs/my-bc/policy.pt --state-run runs/my-demo --hidden-width 8 --updates 1000 --out runs/my-compact
```

保存観測に対する固定ニューラルteacherの台車出力を教師にし、別の小型MLPを学習する。
`--state-run` は複数指定でき、`fetch_goal` / `fetch_bc` の完了runを利用できる。
元の送信行動は蒸留ラベルに使わず、同じ観測でteacherを再推論する。状態は均等抽出。
幅8の2隠れ層は138パラメータで、幅32の1,314パラメータとは別checkpointになる。
学習記録には全状態runのハッシュ、teacherのコピーとハッシュ、パラメータ数/byte数を保存する。
小型モデルのrolloutにはteacherファイルを必要としない。これはプロセス全体のメモリ削減や
temporaryを伴う自動銀行管理の測定ではない。

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
「教師軌跡→学習→重み再読込→学習方策の実行」
「resetなしの目標連鎖」「固定teacher→小型candidate→単独実行」に各1本。
腕IKには `tests/test_arm_ik_feature.py` の1本を追加。上流FKと物理link姿勢、
IK解から送信行動への対応、追従/接触診断と保存データを確認し、把持成功率は条件にしない。
操作BCと再ラベル収集には `tests/test_operation_bc_feature.py` の1本だけを追加した。
小操作の目標更新→実行→教師データ→MLP再読込→単独実行は
`tests/test_primitive_feature.py` の1本で確認する。

```bash
APC_RUN_MANISKILL_TEST=1 python -m pytest -q tests/test_rollout_feature.py tests/test_fetch_reach_feature.py tests/test_bc_feature.py
# 連鎖/蒸留を変更した場合は対象の機能を指定
APC_RUN_MANISKILL_TEST=1 python -m pytest -q tests/test_sequence_feature.py tests/test_distillation_feature.py
# 操作BC/再ラベル機能を変更した場合
APC_RUN_MANISKILL_TEST=1 python -m pytest -q tests/test_operation_bc_feature.py
# 小操作と判断器を変更した場合
APC_RUN_MANISKILL_TEST=1 python -m pytest -q tests/test_primitive_feature.py
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
138パラメータへの蒸留でも停止維持/戻り連鎖を完了したが、連鎖のstep数は増えた。
同じ拡張データで3,000更新へ増やすと戻り方向は737→524 stepに短縮した。
未使用だったseed 1008〜1012では元モデル・小型モデルとも5/5連鎖を完了し、
最後の1秒間も条件を維持した（合計1,106/1,088 step）。

**操作BC:** 2回目DAggerの退行は、把持済み756状態へ接近stageの開放指令を付けた
教師同期不良が主因候補だった。物理状態への再同期と無効IKラベル除外後、探索seedでは
初めて2/3目標到達したが、1秒維持0/2、未使用seedでは0/5だった。単純なラベル破損は
修正できた一方、次はstage抽出比率、履歴、局所目標化、新条件での再ラベルのどれを
変えるべきか一意でないため、操作学習を無制限に続けない。
上記の大きなスキルの銀行化順序は後続の [小操作計画](docs/PRIMITIVE_DECISION_PLAN.md)
で更新した。現在の次の一点は、教師が成功する探索条件で学習selectorが失敗する局面の
ID誤りと、保持目標を含む入力の識別性の診断である。
現在の学習済み操作を完成スキル・自動銀行・移動から把持への連鎖の実績とはしない。
GPU物理・動画・APCの自動銀行は未検証。

## 上流資料（2026-09-22確認）

- [S1: インストール・対応OS](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [S2: PyPI ManiSkill 3.0.1](https://pypi.org/project/mani-skill/3.0.1/)
- [S3: Quickstart・Fetchへの身体変更](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)
- [S4: RecordEpisode](https://maniskill.readthedocs.io/en/latest/user_guide/wrappers/record.html)
- [S5: v3.0.1 Fetchコントローラ](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/agents/robots/fetch/fetch.py)
- [S6: v3.0.1 BaseEnv・描画無効化](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/sapien_env.py)
- [S7: 上流RL導入](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html)
- [S8: Pinocchio公式導入方法](https://stack-of-tasks.github.io/pinocchio/download.html)
- [S9: PyPIの別パッケージpinocchio](https://pypi.org/project/pinocchio/)
- [S10: conda-forgeのPinocchio配布](https://anaconda.org/conda-forge/pinocchio/files)
- [S11: PyPI pin 4.1.0の配布ファイル](https://pypi.org/project/pin/4.1.0/#files)

`latest`文書は変化する。実行条件はそのrunのインストール版と成果物を優先する。
