# 実験メモと引き継ぎ

このファイルは短い判断メモ。承認・ゲート・全履歴の読み直しを要求する台帳ではない。
大きな軌跡や動画、チェックポイントはgitignoredなruns/に置く。

## 2026-09-22 — 環境試行基盤の作成

**基点:** main `319f3af8b16accf7347574247e4eae6ab30ad402`。
既存APC/CNPのコード、測定、依存は変更しない。旧AGENTSはAGENTS_LEGACY.mdへ
同一blobのまま保存し、新しいroot AGENTSからトラック別に参照する。

**変更:** 独立venvのsetup、CPU/GPUを明示した単一環境runner、Panda/Fetchの設定、
軌跡/終了/成功/依存の記録、集計、任意動画、計画・設計・AI指示を追加。
自動テストは「実環境rollout→保存→集計」の1本だけ。
目的地点への移動課題、学習、銀行、合成、圧縮は未実装。

**実行した確認:** Pythonの構文コンパイル、shell構文、setup/CLIのhelp、
両JSON設定の読み込みを確認した。ローカルの一時的なbatched interface fixtureで、
収集→保存→集計、観測のコピー、terminated/truncated/予算打ち切り、欠測成功判定、
既存runの非上書きを確認した。このfixtureは物理シミュレータではなく、研究結果は0件。
リポジトリにはこのfixtureやそのデータを追加しない。

**実環境の確認状況:** 作業用コンテナはPython 3.13.5、gymnasium/ManiSkill/SAPIENとGPUがない。
実runnerの起動は `ModuleNotFoundError: No module named 'gymnasium'` で非ゼロ終了。
manifestがfailed、episode数0、error.txtが保存されることを確認した。
実機用pytestはデフォルトの明示skipで **1 skipped**。実シミュレータのテスト成功ではない。
専用venvの実インストール、Python 3.11/3.12での実行、物理、Fetch、GPU、動画、学習は未検証。
依存lockや成功率を捏造して追加していない。

**次の作業:** 研究用マシンでsetup→Panda試行→Fetch試行を行い、runパスと症状を記録する。
そのデータから最小のFetch目標到達課題を作る。成功率による次段階ゲートは置かない。

### 次のAIコーディングへの指示例

> experiments/maniskill/AGENTS.md とREADME、RESEARCH_PLANの最初の作業を参照する。
> 研究用マシンで既存runnerを動かして軌跡を保存し、データとエラーを確認する。
> そのうえで既存Fetch/ManiSkill部品を使う最小の目標到達課題を実装し、少数episodeを試す。
> 大きな機能がまとまったときだけ、その機能全体の統合テストを原則1本追加する。
> フェーズゲートや文書承認を作らず、動く部分から実験を続け、変更と次の一点を記録する。

## 2026-09-22 — Windows Git Bash セットアップの修正と実行

- 問い/今回変えた点: Windowsの`core.autocrlf=true`でshell scriptがCRLF化し、Bashの`set -euo pipefail`が失敗した。`.gitattributes`で`.sh`をLF固定し、Windows venvの`Scripts/python.exe`とWindows形式のPython引数パスを扱えるようにした。
- commit / config / robot・controller / seed・split: 未コミット。config・robot・seedは未使用。
- 環境step・episode・学習step・wall time等の予算と実績: setupのみ。rollout、episode、学習stepは0。
- runパス / 実データの観測 / 失敗した条件: `runs/setup-20260922T120412Z-834/`に`doctor.json`と`requirements.freeze.txt`を保存。Python 3.12.13、ManiSkill 3.0.1、SAPIEN 3.0.3、PyTorch 2.14.0を導入。RTX 5060 TiとVulkanは検出した一方、`torch_cuda_available`はfalse。初回はCRLF、次にWindows venvの`bin/python`不在、さらにGit Bashの`/c/...`をWindows Pythonのeditable installに渡したことにより停止した。
- 解釈（測定と推測を分ける）: CPU環境の導入と診断は完了した。PyTorch CUDA不可はdoctorの測定結果であり、PhysX GPU実行可否は未測定。
- 次に変える一点: `run_iteration.sh`でPanda/PickCubeを少数episode実行し、CPU物理のrun成果物を確認する。
- 機能テストを実行した場合の結果 / 未実行と理由: setup scriptの実行とdoctorを確認。実シミュレータの機能テスト・rolloutは、この作業の範囲外のため未実行。

## 2026-09-22 — PickCube CPU rollout の実行

- 問い/今回変えた点: WSL等で`cygpath`がない場合にも、Windows Pythonへ渡すパスを`wslpath`または元のパスで扱えるようにした。
- commit / config / robot・controller / seed・split: 未コミット。`configs/pickcube_cpu.json`、Panda、`pd_joint_delta_pos`、seed 0〜2、explore。
- 環境step・episode・学習step・wall time等の予算と実績: 3 episode、150 environment step、学習step 0、runner wall time 13.875秒。
- runパス / 実データの観測 / 失敗した条件: `runs/trial-20260922T122449Z-79/`。3本のepisode NPZ、step/episode JSONL、manifest、依存記録、summaryを保存。ランダム方策の成功率は0/3、平均returnは3.840811556826035。各episodeは50 stepで環境のtruncatedとなり、runnerによる予算打ち切りは0。Pinocchioが未導入という警告は出たが実行は完了した。
- 解釈（測定と推測を分ける）: CPU物理でPanda/PickCubeの環境生成、軌跡保存、集計は測定済み。ランダム方策の0成功はAPC学習やスキル再利用の結果ではない。
- 次に変える一点: Fetch/PickCubeを少数episode実行し、身体と制御空間を観察する。
- 機能テストを実行した場合の結果 / 未実行と理由: `run_iteration.sh`の実環境trialが完走。pytest機能テストは今回未実行。

## 2026-09-22 — Fetch観察と最小目標到達課題の実行

- 問い/今回変えた点: 既存runnerでPanda/Fetchを各3 episode確認した後、上流BaseEnv・Fetch・build_groundで `APC-FetchReachGoal-v1` を追加。全身13次元を維持し、手設計の台車制御と腕/胴体のrest姿勢への補正を実装。距離改善報酬、到達時の速度、姿勢誤差、reset/final infoを保存する。
- 実行環境: このWindowsマシンの専用 `.venv`、Python 3.12.13 / ManiSkill 3.0.1 / SAPIEN 3.0.3 / torch 2.14.0、CPU物理、描画なし。別のリモート研究ホストではない。制御20 Hz、物理100 Hz。既存のsetup/script/文書の未コミット変更を保持した。
- config / seed・split / 予算: `pickcube_cpu.json` と `fetch_pickcube.json`（各3 episode、最大64 step）、`fetch_reach_goal.json`（各3 episode、最大200 step）。全てseed 0〜2、explore。学習step 0。以下のrunはdirty状態で、到達課題の成功runには実行コードの `source_snapshot/` を追加保存した。

| run（`runs/` 以下） | 方策 | episode / step | 成功 | 平均return | runner秒 |
|---|---|---|---|---|---|
| `9672e415d106490491409bb42b4a54c5` | Panda/PickCube random | 3 / 150 | 0/3 | 3.840812 | 3.547 |
| `253e6731512a46e282010a9c3d042287` | Fetch/PickCube random | 3 / 150 | 0/3 | 0.356938 | 4.469 |
| `reach-scripted-20260922-a` | fetch_goal（初回失敗） | 0 / 0 | 未測定 | — | 3.093 |
| `reach-scripted-20260922-b` | fetch_goal | 3 / 119 | 3/3 | 0.687587 | 3.031 |
| `reach-zero-20260922-a` | fetch_zero | 3 / 600 | 0/3 | 0.000000 | 3.657 |
| `reach-random-20260922-a` | fetch_random | 3 / 600 | 0/3 | -0.107683 | 3.641 |

- 実データ: 合計15 episode / 1,619 step（統合テストを除く）。全NPZの数値が有限、観測T+1と行動Tの整合、step数、到達課題のreturn＝初期距離−最終距離を確認。Pandaは行動8/観測42、Fetch/PickCubeは行動13/観測54、Fetch到達は行動13/観測40。PickCubeは環境の50 step制限、到達のzero/randomは200 step制限でtruncated。scriptedは41/39/39 stepでterminated。runner予算打ち切りはなし。
- 到達データ: 同じseedの初期距離は0.774852/0.709843/0.720483 m。scriptedの最終距離0.047772/0.045533/0.049114 m、並進速度0.049885/0.033827/0.035683 m/s、角速度0.000661/0.000365/0.000645 rad/s。zeroは位置を維持。randomの最終距離0.406608/1.144347/0.977274 m。
- 姿勢保持: scriptedの台車以外の最大座標誤差は約0.00570、randomは最大約0.22559。m/rad混在の診断値であり、剛体固定を意味しない。急なランダム速度指令では姿勢誤差が増えたという観測を残す。
- エラー: 初回到達runはNumPy float64からtorch float32への代入で失敗（`manifest=failed`、`error.txt`保存、0 episode）。明示float32化して新規runで再試行。Pinocchio未導入の警告は全runで出たが、使用した関節制御の実行は完了した。
- テスト: 新規機能に統合テスト1本だけ追加。実環境CLI→2 episode→NPZ/JSONL→summaryを通し、距離・報酬・速度から成功/終了判定を照合する。性能閾値はなし。既存1本と合わせ **2 passed in 10.55s**、成果物は `runs/feature-tests-20260922-b/`。最初のsandbox実行はpytest一時ディレクトリのアクセス拒否で環境起動前に失敗し、権限付き実行で別ディレクトリへ保存した。これはシミュレータ失敗とは区別する。
- 解釈/未実行: 狭い開始/目標範囲での手設計制御の動作確認。学習成功・汎化・APC銀行/再利用の証拠ではない。GPU物理、動画、障害物、学習は未実行。成功率を次作業のゲートにしない。
- 次に変える一点: 同じ課題・全身行動対応で、台車2出力の小規模な模倣学習baselineを作り、学習した方策自身の閉ループrolloutを比較する（姿勢補正は共通）。

## 2026-09-22 — 台車の小規模な模倣学習baseline

- 問い/今回変えた点: 既存の手設計 `fetch_goal` 軌跡から台車2出力を学習し、学習済み方策自身を閉ループで動かした。相対目標xy・身体座標速度vx/vy/yaw rateの5入力、5→32→32→2 Tanh MLP（1,314パラメータ）。全身行動13次元は維持し、腕/胴体/グリッパ補正は共通。台車の手設計停止ルールやteacher切替は推論中に使わない。
- 基点/環境: `fab33f9` + この変更のdirtyソース。Windows専用venv、Python 3.12.13、ManiSkill 3.0.1、SAPIEN 3.0.3、torch 2.14.0、CPU物理・描画なし。学習/比較runには実行時の `source_snapshot/`、学習に使ったデータのハッシュ、学習方策runにはcheckpointコピーとハッシュを保存。
- 予算: 教師10 episode（seed 10〜19、各最大200 step）、学習seed 0、Adam lr 0.001 / batch 64 / 1,000勾配更新。比較は教師・学習方策各3 episode（seed 1000〜1002、各最大200 step）。この比較seedは学習には使わないが、結果を確認した探索用として扱う。
- config: `fetch_reach_goal.json` を使用し、教師はepisodes/seed、学習方策はさらにpolicy=`fetch_bc`とcheckpointをCLIで上書きした。初回比較runのtask_labelは元configの手設計ラベルを継承しているが、実際の方策はconfig.policyとpolicy_detailsで識別できる。今後用の `fetch_reach_bc.json` は教師ありbaselineと明示する。

| run（`runs/` 以下） | 用途 | episode / 環境step | 成功 | 平均return | wall秒 |
|---|---|---|---|---|---|
| `bc-demo-20260922-a` | 教師データ | 10 / 406 | 10/10 | 0.675954 | 3.609 |
| `bc-train-20260922-a` | 模倣学習1,000更新 | 0 / 0（保存データ406件使用） | 対象外 | 対象外 | 1.453 |
| `bc-policy-20260922-a` | 学習方策自身の閉ループ | 3 / 152 | 3/3 | 0.786092 | 3.453 |
| `bc-teacher-compare-20260922-a` | 同じ開始条件の教師 | 3 / 129 | 3/3 | 0.768854 | 3.329 |

- 学習データ上のMSE: 0.1180903 → 0.000407562。全教師データ上の当てはまりであり、独立検証損失ではない。`training.json`、`losses.jsonl`、`policy.pt`（9,727 byte）、`comparison.json`を保存。新規環境相互作用は計16 episode / 687 step（統合テストを除く）。
- 閉ループ観測: seed順に学習方策は43/54/55 step、最終距離0.038546/0.026470/0.026628 m、並進速度0.046736/0.000358/0.000864 m/s、角速度0.032052/0.098916/0.097651 rad/s。教師は42/43/44 step、最終角速度0.000324/0.001119/0.000944 rad/s。全episodeはsuccessによるterminatedで、時間打ち切りなし。開始位置/目標に対応する初期距離は両者で一致。
- 解釈: 学習方策の3/3到達は狭い分布での初回試行。教師より23 step多く、2 episodeは最終角速度が0.10 rad/sの停止閾値付近に残る。return増加は最終距離が小さいことによるもので、到達速度改善を意味しない。教師の台車ゼロ指令は25/406件。この少なさが原因かは未測定。銀行・temporary・容量削減・再利用の成果とはしない。
- 確認/エラー: 全保存NPZの数値有限・T/T+1整合、距離差とreturn、checkpoint再読込による各時点の出力と実送信行動の一致を確認。今回の実験runに例外はなく、Pinocchio警告のみ。`bc.py` の入力に同じstate40の相対座標・速度を使うこともinfoの幾何から照合した。
- 統合テスト: 新規機能に1本追加（実教師2 episode→学習40更新→重み再読込→学習方策2 episode→成果物/集計）。実際にパラメータが更新され、記録行動が学習済み出力と一致することを検査。損失改善/成功率の閾値なし。共有runnerの既存2本と合わせ **3 passed in 23.21s**。保存先 `runs/bc-feature-tests-20260922-a/`。前回のpytest一時フォルダのアクセス問題に合わせ権限付きで実行した。
- 次に変える一点: 同じ教師データ・ネットワーク・1,000更新で停止指令サンプルの学習比率だけを上げ、同じseed 1000〜1002で角速度と到達stepを比較する。GPU、動画、障害物、分布外目標、APC銀行/合成/圧縮は未実行。

## 2026-09-22 — 停止指令の抽出重みだけを変更

- 問い/変更: 停止指令の少なさが到達後の旋回残りに関係するかを確認するため、`train-bc --stop-weight 4` を追加。教師台車2出力の絶対値がともに1e-7以下なら抽出重み4、それ以外は1とし復元抽出。既定の重み1は従来のrandint経路を保つ。ネットワーク・姿勢補正・成功条件は変更しない。
- 基点/条件: `6d5fa97` + 今回のdirtyソース（各runにsnapshot保存）。既存 `bc-demo-20260922-a` の406件、seed 10〜19を再利用し、manifestと全軌跡ハッシュの一致を確認。CPU、学習seed 0、1,314パラメータ、Adam lr 0.001、batch 64、各1,000更新。新しい教師環境stepは0。
- 学習run: `runs/bc-stop4-train-20260922-a/`（1.968秒）と既定条件の再現確認 `runs/bc-uniform-check-20260922-a/`（1.797秒）。旧モデルと再学習した重み1のstate_dict・正規化統計は全要素で完全一致。checkpoint/学習記録/比較表 `comparison.json` を保存。
- サンプリング: 停止指令は25/406件。期待停止比率は重み1で6.158%、重み4で20.790%。実際の抽出は各64,000件中3,833件（5.989%）→13,431件（20.986%）。更新予算は両方同じ。
- 元データ上の重みなしMSE（重み1→4）: 全体0.000407562→0.000511113、停止0.00383062→0.00122773、移動0.000182952→0.000464091。停止への当てはまりは改善し、移動側は悪化した。独立評価損失ではない。
- 実環境run: `runs/bc-stop4-policy-20260922-a/`、`configs/fetch_reach_bc.json`、seed 1000〜1002、最大200 step×3 episode、explore。既存の重み1・教師runと開始姿勢/目標/seedを照合。成功3/3、126 step、平均return 0.763022、wall 8.687秒。全てsuccessによるterminatedで打ち切りなし。wallは統合テストとの同時実行時間を含むため、処理速度の比較には使わない。

| 条件 | seed 1000/1001/1002の終了step | 初めて目標範囲に入るstep | 最終角速度 rad/s |
|---|---|---|---|
| 重み1（前回保存run） | 43 / 54 / 55 | 36 / 37 / 38 | 0.032052 / 0.098916 / 0.097651 |
| 重み4 | 41 / 42 / 43 | 36 / 37 / 38 | 0.026888 / 0.019740 / 0.019708 |
| 教師（前回保存run） | 42 / 43 / 44 | 37 / 38 / 39 | 0.000324 / 0.001119 / 0.000944 |

- 解釈: この3例では目標範囲へ入るstepは変わらず、その後の停止成立が7/17/17 step後から各5 step後に短縮した。合計152→126 step（26 step減）。重み4の最終距離は0.051674/0.054411/0.054770 m、並進速度0.038472/0.042365/0.046184 m/s。目標からやや遠くで停止するため、return減少だけで悪化とは判断しない。少数の既知探索seedと学習seed 0での結果であり、一般的な優位性は主張しない。
- 確認/テスト: 新規環境相互作用は3 episode / 126 step、学習は比較用を含め2,000更新。全NPZの有限値・T/T+1・距離報酬・保存モデルの出力と実送信行動を照合。例外なし、Pinocchio警告のみ。小変更なので新規テストを増やさず既存BC統合テストに重み付き経路を反映し、**1 passed in 16.38s**。`runs/bc-stop-feature-tests-20260922-a/` に保存。無関係なテストは再実行しない。
- 次に変える一点: 重み1/4の学習済みモデルを固定したまま、未使用のseed 1003〜1007で各5 episodeを対で比較する。追加学習や重み探索をせず、停止改善と移動側悪化が別開始条件でどう出るかを見る。

## 2026-09-22 — 固定モデルを追加5 seedで対比較

- 問い/今回変えた点: 重み1/4の既存checkpointを固定し、未使用だったseed 1003〜1007だけを追加。学習・方策・環境コードは変更せず、同じ初期条件の各5 episodeを順番に実行した。今回確認したseedもexploreとして記録する。
- 基点/条件: `599722d`、実行時git clean。Windows専用venv、Python 3.12.13 / ManiSkill 3.0.1 / SAPIEN 3.0.3 / torch 2.14.0、CPU物理、描画なし、Fetch、`pd_joint_delta_pos`、制御20 Hz / 物理100 Hz。`configs/fetch_reach_bc.json` に `--seed 1003 --episodes 5`、各最大200 step。モデルは `bc-train-20260922-a/policy.pt` と `bc-stop4-train-20260922-a/policy.pt`。学習seed 0、教師seed 10〜19のまま。
- 予算/実績: 最大2,000環境stepの予算に対し、10 episode / 510環境step。追加教師データ0、学習更新0。両runにmanifest・依存・実行ソース・checkpointコピー・全軌跡・summaryを保存。例外なし、既知のPinocchio未導入警告のみ。

| run（`runs/` 以下） | 条件 | 成功 | 環境step | 平均return | runner秒 |
|---|---|---|---|---|---|
| `bc-uniform-newseeds-20260922-a` | 重み1 | 5/5 | 276 | 0.879925 | 5.000 |
| `bc-stop4-newseeds-20260922-a` | 重み4 | 5/5 | 234 | 0.861213 | 4.500 |

| seed | 終了step（1→4） | 初めて目標範囲に入るstep（1→4） | 進入後の終了までのstep（1→4） | 最終角速度 rad/s（1→4） |
|---|---|---|---|---|
| 1003 | 61→47 | 42→42 | 19→5 | 0.098287→0.098045 |
| 1004 | 55→47 | 42→42 | 13→5 | 0.097887→0.001589 |
| 1005 | 62→48 | 44→43 | 18→5 | 0.099468→0.021462 |
| 1006 | 50→48 | 42→43 | 8→5 | 0.011236→0.036530 |
| 1007 | 48→44 | 39→39 | 9→5 | 0.098334→0.073822 |

- 観測/解釈: 全episodeがsuccessによるterminatedで、時間/runner打ち切りなし。全5例で終了が早まり、合計42 step（15.2%）減。進入stepの合計は両方209で、進入後の待ち時間が67→25 stepに減った。移動中の教師データMSE悪化は、今回の5例では進入時間の一貫した悪化としては現れない。重み4の最終距離は0.050778〜0.055048 m、速度0.033010〜0.048144 m/s。距離改善returnは停止位置が遠くなる分低下した。
- 留保: 最終角速度が全seedで改善するわけではなく、seed 1003は重み4でも0.098045 rad/sと閾値0.10付近。最終値の最大も0.099468→0.098045にとどまり、前回3例の最大角速度低下はそのまま再現しなかった。成功時点で終了する現在のrunから、到達後に停止を維持できるとは分からない。狭い同一分布・学習seed 0・追加5例での結果であり、APC銀行や連鎖、分布外汎化の実証とはしない。
- データ確認: 対応episodeの初期state40が完全一致。NPZの有限値・T/T+1形状・距離差と各step報酬・幾何/速度からの成功と終了を照合し、全送信台車行動が保存モデルの出力に一致。コピーしたcheckpointのSHA256が元ファイルと一致（重み1 `6ce906e5c2c29d68a5f913d3ab7142c155d7191e1c5305c2250088fe7e307c99`、重み4 `a64121c6779a1b141a4bb04d85ec684b121b54fc8c04eda9ab3c852a753dddf0`）。重み4のrun内に集計/照合スクリプト `analyze_pair.py` と軌跡ハッシュを含む `comparison.json` を保存した。
- テスト/未実行: 機能コードの変更がないためpytest追加・再実行なし。runnerの実10 episodeと保存データの照合を実行した。GPU、動画、障害物、分布外目標、到達後の停止維持、銀行/合成/圧縮は未実行。
- 次に変える一点: 重み4を固定し、成功後も同じ目標で20 step（1秒）制御を継続する診断を追加する。seed 1003〜1007で到達後の距離/並進速度/角速度と条件逸脱を記録し、次の目標への連鎖前に、停止が瞬間的か維持されるかを見る。既存課題の成功条件は保持し、継続診断の終了方法をrunに明示する。

## 2026-09-22 — 初回到達後の停止維持診断

- 変更: `--post-success-steps` とFetch専用wrapperを追加。成功条件・目標・制御出力を保ち、初回成功から指定step後に診断を終了する。元のtask_terminated、first_success_step、post_success_steps、hold_completeを記録し、summaryに最終成功件数を追加した。時間制限は保持。
- 条件/予算: `26a0bae` + dirty source、CPU、固定した重み4 checkpoint、seed 1003〜1007、最大200 step×5 episode、到達後20 step。`runs/bc-stop4-hold-20260922-a/`、5 episode / 334 step、学習更新0、wall 4.625秒。全5例が初回成功し、20 stepの診断を完了したが、最終成功は0/5。
- 観測: seed順の到達後20 step中の成功状態は7/18/19/10/8 step。最終距離0.090042/0.082853/0.081245/0.085883/0.088028 mで、全て許容距離0.08 mを超えた。最終速度0.031474〜0.035000 m/s、角速度0.017287〜0.077848 rad/s。停止は安定な維持状態ではなかった。低成功率は研究停止理由にしない。
- 確認: 元の到達runとの初回成功までの全観測が完全一致。有限値・T/T+1・モデル出力と行動・距離報酬・wrapper終了と20 stepの継続を照合し、`analysis.json` と `analyze.py` を保存。例外なし、Pinocchio警告のみ。
- テスト: 小変更として既存到達統合テストを継続診断に拡張。**1 passed in 6.40s**、`runs/hold-feature-tests-20260922-a/`。成功率をテスト条件にしない。
- 次の一点: 同じ教師seed 10〜19について教師も成功後20 step動かし、停止中の軌跡を増やす。ネットワーク・重み4・更新数1,000・学習seed 0を維持して学習し、同じ5 seedの停止維持を比較する。推論時の手設計停止への切替は追加しない。

## 2026-09-22 — 教師データに停止中の状態を追加

- 変更: 学習コード・ネットワーク・更新数1,000・seed 0・抽出重み4はそのまま、教師seed 10〜19を成功後20 stepまで収集。旧406 stepに停止中の200 stepを加えた606件を使った。単なる同数データ比較ではなく追加環境相互作用を伴う比較。
- 実行: 基点 `fcb0709`、CPU、教師 `runs/bc-hold-demo-20260922-a/` は10 episode / 606 step、wall 4.875秒、全て初回成功後20 stepも成功を維持。学習 `runs/bc-hold-train-20260922-a/` は1,000更新、wall 1.828秒。停止ラベル225/606件、実抽出44,940/64,000件（70.219%）。元データMSEは全体0.000371330、停止0.000221033、移動0.000460088。
- 閉ループ: `runs/bc-hold-policy-20260922-a/`、seed 1003〜1007、20 step継続診断、5 episode / 335 step、wall 4.641秒。初回成功stepは46/48/49/48/44。全5例がその後の20 stepすべてで成功条件を維持し、最終成功は旧モデル0/5→5/5。最終距離0.044894〜0.047342 m、速度0.000284〜0.001730 m/s、角速度0.002308〜0.021322 rad/s。旧モデルより初回成功の合計は1 step多い。
- 確認: 教師の旧406 stepまでの観測は旧データと完全一致。両runの有限値・T/T+1・20 step継続・終了、学習方策の行動とcheckpoint出力を照合し、比較runに `analysis.json` / `analyze.py` を保存。今回計15 episode / 941環境step、1,000更新。例外なし、Pinocchio警告のみ。コード変更なしのためpytest再実行なし。
- 解釈/次の一点: 停止状態を含む教師分布がこの5例の短期維持に役立った。長期安定性・新しい目標方向・複数目標の連鎖は未測定。次はこのモデルを固定して、同一scene内でresetなしに2つの局所目標を実行する。順序は手動指定で、単一方策の反復使用とAPCの自動スキル銀行を区別する。

## 2026-09-22 — 同一sceneでの手動2目標連鎖

- 変更: `TwoGoalSequence`、`next_goal_offset`、連鎖configを追加。第1目標の成功時にgoalだけを切り替え、qpos/qvel・controller・時計をresetしない。第2目標成功をsequence successとし、今回もその後20 stepを観測。全体上限400 step。切替stepの旧目標/報酬はreward_info、新しい目標は次のpolicy観測に保存。自動selector/銀行ではなく単一方策の手動反復使用。
- 条件: 基点 `22e1064` + dirtyソース、CPU、`fetch_reach_sequence.json`、seed 1003〜1005。固定checkpoint `bc-hold-train-20260922-a/policy.pt`。まず前方offset [0.7,0.2]を両方策各3 episode、次に横 [0,0.7]、戻り [-0.7,0]をそれぞれ各3 episode。方向を変える以外の追加学習なし。各小比較は最大2,400 step。

| run（`runs/` 以下） | 方策 | 完了 | step | wall秒 |
|---|---|---|---|---|
| `sequence-forward-bc-20260922-a` | 学習 | 3/3 | 334 | 3.953 |
| `sequence-forward-teacher-20260922-a` | 手設計教師 | 3/3 | 339 | 3.703 |
| `sequence-side-bc-20260922-a` | 学習 | 3/3 | 386 | 3.500 |
| `sequence-side-teacher-20260922-a` | 手設計教師 | 3/3 | 368 | 3.532 |
| `sequence-return-bc-20260922-a` | 学習 | 3/3 | 473 | 4.015 |
| `sequence-return-teacher-20260922-a` | 手設計教師 | 3/3 | 413 | 3.594 |

- 観測: 合計18 episode / 2,313環境step、学習更新0。全episodeで2目標達成後の20 step全てで成功条件を維持。前方の学習方策は第1目標46/48/49 step、第2目標90/89/95 stepで成功。戻り方向のseed 1004は学習184 step・教師122 stepで第2目標に到達し、学習側が62 step遅い。成功だけで効率が同じとはしない。
- 確認/テスト: 全runの有限値・T/T+1・学習出力と全送信行動・旧目標での各step報酬を照合。切替前後の全qpos/qvelと姿勢/速度が完全一致し、elapsed_stepsは連続。戻り学習run内に全6条件のanalysis.json/analyze.py保存。新規機能全体に統合テスト1本を追加し、変更した継続診断と合わせ **2 passed in 10.19s**、`runs/sequence-feature-tests-20260922-a/`。例外なし、Pinocchio警告のみ。最初の前方学習runのsnapshot後、hold wrapperが元task_terminatedを上書きしないよう記録を修正した（物理/方策は同一、hold_input_terminatedを別保存）。
- 解釈/次の一点: この範囲では単一goal-conditioned方策の反復使用で連鎖できたため、能力を無理に別スキルへ分割しない。次は固定1,314パラメータモデルの出力から別の小型candidateを蒸留し、実際の閉ループ維持/連鎖と容量を比較する。自動銀行増設・temporary解放の成果とはまだ呼ばない。

## 2026-09-22 — 固定方策から別の小型candidateへ蒸留

- 変更: `distill` を追加。保存状態に対する固定ニューラルteacherの出力をラベルとして、別MLPを学習。複数state-runの連結、各出典ハッシュ、teacherコピー、candidate幅/容量を記録。推論はcandidateだけをロードする。既存BCの既定幅32と旧checkpoint読込は保持。
- 条件: 基点 `4ee9cf0` + dirty source、CPU、teacher=`bc-hold-train-20260922-a/policy.pt`、幅8、seed 0、Adam 0.001、batch 64、均等抽出、1,000更新。小型candidateは138パラメータ/552 byte（重みtensorのみ）、元モデル1,314/5,256 byteから89.5%減。candidate checkpointは5,311 byte。プロセス全体のメモリや自動temporary解放の成果とはしない。
- 初回学習: `runs/distill8-20260922-a/`、元教師runの606状態、MSE 0.0827351→0.000265207、wall 1.531秒。`distill8-hold-20260922-a` はseed 1003〜1007、5/5到達・その後20 step全て成功維持、350 step、wall 3.844秒。`distill8-return-20260922-a` はseed 1003〜1005、戻りoffset [-0.7,0]、3/3連鎖・20 step維持、611 step、wall 4.500秒。元モデルの335/473 stepより遅い。
- 状態分布追加: candidate自身の戻り連鎖をseed 20〜22で3 episode収集した `distill8-return-states-20260922-a` は548 step、3/3完了、wall 4.234秒。この548状態と元606状態を連結し、同じ固定teacherを再推論して新規candidate `distill8-expanded-20260922-a` を同じ1,000更新で学習。1,154状態、MSE 0.0761131→0.000527423、wall 1.484秒。異なるデータ上の損失を直接の改善指標にはしない。
- 再試行: `distill8-expanded-return-20260922-a` は同じ比較seedで3/3連鎖・20 step維持、737 step、wall 4.328秒。`distill8-expanded-hold-20260922-a` は5/5到達・20 step維持、341 step、wall 3.735秒。戻り方向はデータ追加だけでは改善せず611→737 stepと遅くなった。今回計19 episode / 2,587環境step、2,000更新（テストを除く）。
- 確認/テスト: 全5実runの有限値・T/T+1・モデル出力と行動・切替を考慮した距離報酬・終了を照合。teacherの元ファイルと両学習runコピーのSHA256が一致。expanded-return runに全条件のcomparison.json/analyze.pyを保存。蒸留機能に統合テスト1本を追加し、既存BCと合わせ **2 passed in 28.98s**、`runs/distill-feature-tests-20260922-a/`。テストでは教師ファイルを一時退避後、別プロセスでcandidateを実行し、teacher出力から計算し直したMSEとの一致も検査した。例外なし、Pinocchio警告のみ。
- 次の一点: 状態1,154件と幅8を固定して更新数だけを3,000へ増やし、戻り方向の遅さが学習不足で変わるかを見る。低性能を障害とみなして停止せず、単一条件ずつ比較する。

## 2026-09-22 — 小型candidateの更新数と未使用seedでの比較

- 変更/条件: 基点 `68bd35e`、ソース変更なし。状態1,154件・幅8/138パラメータ・seed 0を固定し、更新数だけ1,000→3,000。`runs/distill8-expanded-3000-20260922-a/`、wall 2.515秒、同じデータ上のMSE 0.000527423→0.000197943。全入力runハッシュと最初の1,000更新までの記録損失が一致することを確認。
- 既知探索条件: `distill8-3000-return-20260922-a` はseed 1003〜1005、3/3連鎖・到達後20 step全て成功維持、524 step（1,000更新の737から短縮、元モデル473よりは遅い）、wall 4.437秒。`distill8-3000-hold-20260922-a` はseed 1003〜1007、5/5到達・20 step全て成功維持、336 step、wall 3.922秒。
- 追加比較: 両モデルを固定し、未使用だったseed 1008〜1012で戻り連鎖を各5 episode実行。`sequence-return-newseeds-teacher-model-20260922-a`（元の学習モデル、手設計teacherではない）は5/5、1,106 step、wall 5.000秒。`distill8-3000-return-newseeds-20260922-a`（小型）は5/5、1,088 step、wall 4.844秒。両方とも全例で最後の20 stepを維持。初期state40は対ごとに完全一致。追加seedも観測後はexplore扱い。
- 実績/確認: 合計18 episode / 3,054環境step、3,000更新。全保存軌跡の有限値・T/T+1・checkpointと送信行動・目標切替を考慮した距離報酬・終了を照合。最後のrunにcomparison.json、analyze.py、additional_checks.json、check_sources.pyを保存。コード変更なしでpytest再実行なし。例外なし、Pinocchio警告のみ。合計step数は近いが、seed 1011では元モデル136→小型304 stepと遅くなり、seed 1010では292→216 stepと速くなった。少数条件での比較であり、一貫した効率改善・広い汎化・APC銀行の優位性を主張しない。
- 次の一点: 移動と小型化が動く範囲を残し、研究計画Cの接触操作へ向け、既存Fetch/PickCubeの腕IK部品の利用可否を確認する。関節制御13次元を保ち、異なる身体/行動対応の転移とはしない。

## 2026-09-22 — 接触操作準備のCPU逆運動学で依存障害

- 問い/試行: 上流Fetch/PickCubeと13次元 `pd_joint_delta_pos` を生成し、腕の目標関節角を得るための前提として `agent.robot.create_pinocchio_model()` を実行。独自IKや接触方策の実装前に既存部品を確認した。runnerのenv_factoryでprobeを行い、失敗時もmanifest/error/依存/source_snapshotを保存する。
- 条件/実績: 基点 `68bd35e`、Windows専用venv、CPU、seed 0、計画1 episode/最大1 step。`runs/fetch-pickcube-ik-probe-20260922-a/` は **failed、0 episode / 0環境step**、wall 3.641秒。環境/Fetchの生成とaction shape [13]までは確認し、kinematics生成で `TypeError: 'NoneType' object is not callable`。`probe.py`、`ik_probe.json`、`error.txt`、summaryを保存した。
- 原因の確認: `importlib.util.find_spec('pinocchio')` はNone。インストール済みSAPIEN 3.0.3の `wrapper/pinocchio_model.py` は、非LinuxでPinocchio importに失敗すると `PinocchioModel=None` を設定し、生成関数はそれを呼んで失敗する。ManiSkillのCPU Kinematicsもこのクラスを使う。これまでの関節制御では警告のみだったが、今回の上流CPU IK経路には実際の障害となった。
- 停止理由/次の一点: ユーザーの「明らかな障害が発生するまで自律的に進める」という停止条件に当たるため、新規実験はここで止める。移動・連鎖・蒸留の成果や既存runは保持し、環境依存は変更していない。次は専用venvで使えるPinocchio導入経路を確認してこの生成probeを再実行する。導入できない場合の別IK経路は未調査。接触操作の成功・学習は未実行であり、この失敗を研究全体の科学的ゲートにしない。

## 2026-09-22 — Pinocchio導入経路の非変更調査

- 依頼/範囲: venvへの導入方法を検討。基点 `9e3657b`。パッケージ導入・更新、長いソースビルド、Conda環境作成、シミュレータ実験は行わず、配布メタデータとpip dry-runを確認した。環境step/学習更新は0。
- PyPI確認: 正しいpip配布名は `pin`、import名は `pinocchio`。PyPI `pinocchio` はnoseテスト用の別パッケージ。PyPI JSONを取得し、全リリースのWindows wheelは0件、最新4.1.0もLinux/macOSのみと確認。現在のPython 3.12専用venvで `pip install --dry-run --only-binary=:all: --index-url https://pypi.org/simple pin` を実行するとNo matching distribution。初回sandbox内の通信失敗と一時フォルダ削除警告を区別し、通信可能な権限付き実行でも同じ配布不一致を確認した。導入はしていない。
- Windows代替: conda-forge APIにwin-64/py312のPinocchio 3.8.0と4.1.0が存在。3.8.0の直接依存はNumPy >=2.1かつ<3、EigenPy 3.12、Boost 1.88等で、現在のNumPy 2.5.3は直接のバージョン範囲内。ただし推移依存の解決/ABI/ManiSkillとの動作は未検証。4.1.0はlibpinocchio/pinocchio-python分割パッケージに依存。Conda/micromamba/CMake/clは今回のPATHでは見つからず、WSLの実行ファイルのみ検出（ディストリビューション利用可否は未確認）。
- 判断: Windowsを維持するなら既存venvを保存して別Conda環境を作るのが第一候補。venv方式を必須にするならLinux側に新設し公式pip経路を使う。Windows venvへのソースビルドはネイティブ依存整備を要するため後順位。Condaパッケージを現在のvenvへ直接コピーする構成は採用しない。READMEに実行前の解決確認案と、導入後のFK/IK検証手順を記載した。
- 成果物/状態: `runs/pin-install-review-20260922-a/` にpip-dry-run-network.txt、pypi-metadata.json、conda-metadata.json、local-checks.json等を保存。既存venvの `pip check` はNo broken requirements found、Pinocchio importは依然不在。調査のみのためpytest追加/再実行なし。次の一点は別検証環境で依存を解決し、Pinocchio 3.8.0を候補として上流Fetch CPU IK probeを再実行すること。版の候補選定は互換性実証とは区別する。
- 一次資料: [公式導入方法](https://stack-of-tasks.github.io/pinocchio/download.html)、[PyPI pin](https://pypi.org/project/pin/4.1.0/#files)、[conda-forge配布メタデータ](https://api.anaconda.org/package/conda-forge/pinocchio)、[上流Windows導入回答](https://github.com/stack-of-tasks/pinocchio/discussions/2470)。上流のWindows回答は2024年のため、今回の配布メタデータでも照合した。

## 2026-09-22 — WSL venvにPinocchio導入、Fetch FK/IK確認と描画障害

- 依頼/変更: WSLのvenvで続行。基点 `88c28f9`、Ubuntu 26.04 / WSL2、Python 3.12.14。既存uvで `/home/tamot/.venvs/apc-maniskill-wsl-py312` を新設し、CPU版torch 2.14.0+cpu、ManiSkill 3.0.1、SAPIEN 3.0.3、Pinocchio 3.8.0を導入した。Windows側venvの前後freezeは一致。システムドライバは変更していない。
- 導入時の失敗: `runs/setup-wsl-20260922-a/` はpip解決/pip check成功後、`liburdfdom_sensor.so.4.0` 不在でimport失敗。`setup-wsl-20260922-b/` でurdfdomを4.0.1にすると、次は `libtinyxml2.so.10` 不在。`setup-wsl-20260922-c/` でtinyxml2を10.0.0へ固定するとimportとpip checkが成功した。初期解決はurdfdom 6.0.0 / tinyxml2 11系。NumPyは2.3.5。各試行のscript/log/report/freezeを保持し、実測した主な依存を `constraints-wsl-py312.txt` に保存した。完全lockや物理実行の検証済み構成とはしない。
- 環境生成probe: `scripts/probe_fetch_ik.py` を追加し、runnerを通じて既存Fetch/PickCube・13次元関節制御・CPU・描画なしで試行。予定seed 0 / 1 episode / 最大5 step。`runs/wsl-fetch-ik-20260922-a/` は **failed、0 episode / 0環境step**。URDF読込中のSAPIEN `RenderMaterial()` が `RuntimeError: failed to find a rendering device` で停止し、環境内IKまでは到達していない。保存済みprobe.pyが試行時のソース。現行スクリプトにはIK前のseed明示resetを追加したが、環境生成を通過できないためその経路は未実行。
- 動く部分の確認: シミュレータを作らず、同梱Fetch URDFと実際のSAPIEN `PinocchioModel` でFK/IKを計算。腕7関節のみを有効にし、向きを保って手先を上へ2 cm動かす目標、最大500反復。`runs/wsl-pinocchio-fk-20260922-a/` の全関節0姿勢では有限値だがsuccess=false、位置誤差約0.300 mm。`wsl-pinocchio-fk-20260922-b/` では初期値だけをFetchのrest keyframeへ変更し、success=true、SE(3)誤差9.2874e-5、位置誤差9.2865e-5 m、腕以外の最大関節変化0。両runにcheck.py/result.jsonを保存した。nq=19 / nv=15は連続関節のPinocchio表現を含み、制御行動次元の変更ではない。
- 実績/限界: 今回の物理episode/環境step/学習更新は全て0。URDFの運動学計算成功を接触操作やrolloutの成功と呼ばない。実環境を生成できないため統合テストは未実行、細切れテストは追加していない。Vulkan/glvndのICD警告と非推奨Frame.parent警告あり。WSLにはMesa ICDとWindows連携のCUDA/D3D12ライブラリがあるが、今回のSAPIEN描画経路は動かなかった。
- 次の一点/停止理由: Pinocchio依存不足は解消した。次は描画無効でもvisual materialを作るURDF読込経路を対象に、描画デバイスなしで環境生成する最小修正を検討する。PickCubeの物体生成にも描画材質生成があるため、URDFだけの修正で解決すると断定しない。ユーザーの「明らかな障害まで」の指示に従い、今回の実験はこの実行障害を記録して止める。

## 2026-09-22 — WSLで描画を有効にした場合の切り分け

- 問い/条件: ユーザーの「描画有りを許容するなら進められるか」に対し、既存WSL venvと依存を維持し、基点 `2cd4b5d`、`fetch_pickcube.json` に `--video --episodes 1 --max-steps 1` を指定して確認した。CPU物理のまま、描画backendをgpuへ変更。システム/ドライバの変更なし。
- 結果: `runs/wsl-fetch-render-probe-20260922-a/` はfailed、0 episode / 0環境step。描画なし時のURDF読込より前、`_setup_scene` の `sapien.render.RenderSystem` 生成時に `vk::createInstanceUnique: ErrorIncompatibleDriver`。コンソールにはVulkan外部memory/semaphore拡張の不在と描画ドライバ非対応のメッセージが出た。別プロセスの `sapien.render.get_device_summary()` も `failed to find a rendering device` で失敗。
- 解釈/次の一点: 描画禁止という実験条件を緩めるだけでは解消しない。現在のWSLからSAPIENのVulkan描画経路を利用できない。上流の対応表もWSLのRenderingは非対応、Windows/NVIDIAおよびLinux/NVIDIAは対応と記載している（[公式資料](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#system-support)、今回再確認）。WSL継続時の次の一点は引き続き描画なしの環境生成経路の修正。描画を使う別案はネイティブLinux、またはWindows側の別環境でPinocchio導入を検証すること。今回の確認からWSL全般での絶対的な不可能性は主張しない。機能変更がないためpytestは追加/再実行していない。

## 2026-09-22 — WSLの描画なし経路を修正しCPU実験を再開

- 変更/範囲: 基点 `0003c89` + dirty source。`headless.py` にManiSkill 3.0.1 / SAPIEN 3.0.3限定の互換処理を追加し、runnerの動画なし環境生成で有効化。URDF loaderはlinkの浅いコピーからvisualsだけを除き、慣性・衝突は元の構築処理へ渡す。立方体/球は元と同じ衝突形状、body type、scene indices、初期姿勢で生成し、描画材質だけを省く。sceneが描画可能なら元の関数を呼ぶ。reset/reconfigureのため処理はプロセス内に保持。パッケージ本体・ドライバ・既存runは変更していない。任意タスクや他のパッケージ版までの一般対応ではない。
- 環境内IK: `runs/wsl-fetch-headless-ik-20260922-a/`、Fetch/PickCube、CPU、seed 0、13次元関節制御。環境生成と既存SAPIEN Pinocchioモデル生成が成功。腕7関節のみ有効にして手先を上へ2 cm動かす解はsuccess=true、位置誤差9.2746e-5 m、腕以外の関節変化0。その後のゼロ行動rolloutは1 episode / 5 step、wall 5.303秒。IK解そのものはまだ制御に送っていない。ik_probe.jsonに実際の関節順・解・目標を保存した。
- 到達/物理実行: `runs/wsl-headless-reach-20260922-a/` は既存手設計fetch_goal、seed 1000〜1002、3/3到達、129 step、wall 5.255秒。`runs/wsl-headless-pickcube-20260922-a/` は既存ランダム方策、seed 0〜2、3 episode / 150 step、wall 5.430秒、全例50 stepの環境時間制限で終了、把持・配置成功0/3。ゼロ成功を起動失敗とはしない。今回WSL実験は計7 episode / 284 step、学習更新0。保存NPZの有限値、T/T+1、13次元行動と範囲、報酬・集計・最終infoを照合し、後者runにaudit.py/audit.jsonを保存。
- 物理保持の比較: Windowsの同じ既存venv内で、上流の描画なし生成と修正後の生成をseed 0〜2のFetch/PickCubeで各3 episode / 150 step実行。`runs/headless-parity-20260922-a/` のupstream/patched全NPZ配列（観測・行動・報酬・終了・成功）が完全一致。check.py/comparison.jsonを保存。Windows比較分は計6 episode / 300 step、WSLとの数値一致は要求していない。
- テスト: 新規の細切れテストは増やさず、既存 `test_rollout_feature.py` の1本を物理確認まで拡張した。再生成後に描画コンポーネントがなく、ロボット/机/立方体の衝突があり、立方体の質量0.064 kgが維持されることを確認。高さ0.15 mから20 step落下させ、最終z=0.01999991 m、机との接触力z=0.62786049 Nを測定した後、通常の収集→保存→集計を実行。`wsl-headless-tests-20260922-a/` の初回はテスト側がEntityを物理componentと誤認して **1 failed / 1 passed（16.89秒）**。参照だけを修正し、失敗した収集テストのみ `wsl-headless-tests-20260922-b/` で再実行、**1 passed（9.96秒）**。到達統合テストは初回に通過済み。最終的に対象の既存2本が通過。テスト内診断stepは上記実験step数から除外。
- 留保/次の一点: Vulkan/glvndのimport警告は残るが、このCPU状態観測経路を阻害しない。GPU描画・画像観測は未対応で、修正後の描画あり経路は未実測。依存不足と描画なし起動障害が解消したので、次はIK解を既存13次元関節制御へ渡す短い閉ループ診断で、手先追従誤差と姿勢保持を測る。現在の結果を把持能力・接触操作の学習成果とは呼ばない。

## 2026-09-22 — 腕IKを物理制御へ接続、把持接近の机接触を特定

- 変更/条件: 基点 `c63d4c7` + 各runのdirty source。既存WSL/Python 3.12/CPU venv、Fetch/PickCube、状態観測、13次元 `pd_joint_delta_pos`。`arm_ik.py` と `scripts/run_fetch_arm.py` を追加。上流Pinocchio IKを毎step解き、世界→root座標変換と関節差分の正規化を行う。runnerへ `external` policy_factoryと明示的な環境時間制限を追加し、通常の保存形式/元タスク成功判定を維持。各runへスクリプトとsource_snapshotを保存した。外部依存や旧研究コードは変更していない。
- 最初の追従: `runs/wsl-arm-track-20260922-a/`、seed 0〜2、各50 step、計150 step、wall 5.549秒。世界z方向+2 cm、向きは保持。全例4 step目で位置5 mm/姿勢0.05 radの範囲へ入り、最終位置誤差1.9977e-6 m、最後10 stepの最大誤差約5.004e-5 m。3例の腕初期姿勢は同一であり、多様な姿勢での汎化ではない。PickCube成功は0/3で、手先到達と区別する。
- 最初の把持診断: `wsl-arm-pick-20260922-a/`、同seed、各200 step、計600、wall 14.269秒。物体初期位置+12 cmへpitch 90度で接近→下降→閉じる15 step→+15 cm持上げ→保持、という手動段階列を実装したが、全600 stepでIK不収束、接近から進まなかった。有限でも不収束/可動域外の解は腕へ送らない。
- 運動学の切り分け: `wsl-arm-reachability-20260922-a/`、3 seed × 胴体固定/可動 × pitch 0/30/60/90度 × 初期値4種類、計96 solver呼出し、物理step 0、wall 6.842秒。実物理linkとFKの世界位置一致を確認。胴体固定のpitch 0度では12/12候補に可動域内の解、pitch 60/90度は胴体可動を含め今回候補では0件。有限探索での結果であり、全空間で到達不可能という証明ではない。
- 水平姿勢比較: `wsl-arm-pick-horizontal-20260922-a/`、pitchだけ0度、同seed、600 step、wall 11.725秒。全例34〜38 stepで接近目標へ到達したが下降時はIK不収束、最終位置誤差約0.126 m。次に昇降関節もIKへ含めた `wsl-arm-pick-torso-20260922-a/` は600 step、wall 7.123秒。上流IKが可動域を制約しないためseed 0では昇降範囲外の解で停止、seed 2も下降時に範囲外。seed 1はIK有効でも物理位置誤差約0.012 mで接近中に停滞した。
- 有効解の選択: IK不収束/可動域外の場合だけ昇降高さ0.1/0.2/0.3/0.386 mを仮置きし、その高さを固定して腕IKを再計算する有限候補を追加した。可動域内の解から現在関節値との差のノルムが最小のものを選ぶ。m/rad混合の単純指標であり、衝突制約付き最適化ではない。`wsl-arm-pick-bounded-20260922-a/` は同seed、600 step、wall 7.864秒。全stepで有効IK解を制御へ送れたが、物理実行は段階1/0/1（下降/接近/下降）、最終誤差0.03753/0.01182/0.04669 m、机接触力のリンク別ノルム総和19.35/9.64/35.42 N。把持0/3。台車はゼロ速度指令でも反力で移動し、最大のbase関節変化は約0.064（xy[m]/yaw[rad]混合指標）。
- 接触の特定: 各リンクの力ベクトルを追加した `wsl-arm-pick-contact-20260922-a/` は同seed、各120 step、計360、wall 6.710秒。最後の接触は全例 `upperarm_roll_link`、seed 0は `wrist_flex_link` も接触。力ノルム総和27.86/12.87/11.45 N、位置誤差0.04231/0.01585/0.01500 m。有限で可動域内のIK解を得ても上腕/手首が机に接触して追従が停滞する、という物理的な障害を確認した。単なる成功率0を停止理由にしていない。WSL/描画なし起動・収集は正常で全runはcompleted、時間制限で終了した。
- 実績/監査: 実験合計18 episode / 2,910環境step、学習更新0。最後のrunに全6実験の `audit.py/audit.json` を保存。全NPZの有限値、T/T+1、13次元行動/範囲、報酬と集計、物理位置誤差、最終infoの一致を確認。同seedの全実験の初期観測は完全一致。各runは変更時点の実装を保存しており、最初のrunへ後から挙動修正を適用していない。
- 統合テスト: 新機能全体に `test_arm_ik_feature.py` 1本を追加。実シミュレータでtrack/pickを各2 episode実行し、上流FKと物理link姿勢、IK解→実際の送信行動、物理位置誤差、元の把持判定、軌跡/報酬/終了の対応を検査。成功率の閾値なし。変更したrunnerの既存テストと合わせ **2 passed in 15.99s**、成果物 `runs/wsl-arm-tests-20260922-a/`。テストstepは実験予算から除外。既知Vulkan/glvnd/Pinocchio/NumPy非推奨警告のみで実行例外なし。
- 停止理由/次の一点: ユーザー指定の「明らかな障害まで」に従い、衝突を考慮しないIK解が机への接触で実行できない点を切り分けたところで停止する。次は机との接触を避ける腕姿勢・中間目標を導入し、同seedで力と追従を比較する。把持/持上げ段階自体にはまだ到達しておらず、その実行や学習済み操作・自動スキル銀行の成果は主張しない。機能を動かすための文書承認や新しいゲートは作っていない。

## 2026-09-22 — 机接触を修正し、把持・運搬・到達後保持へ続行

- 依頼/条件: 修正方針が明確な障害は直して自律的に続ける、という指示に更新。基点 `4e1bf87` + 各runのsnapshot、WSL既存venv/CPU/Fetch/13次元関節制御、学習0。最初は各3 episode・250 step以内、途中姿勢の確認で動作を遅くした後は各350 step以内とした。既存run・パッケージ本体・旧研究は変更していない。
- 幾何診断: `runs/wsl-arm-clearance-probe-20260922-a/` と `-b/`、各72回、計144回の上流IK、物理step 0。上流のcollision mesh取得とFKから机/腕リンクの軸平行境界箱を計算。pitch 15度・物体中心より12 mm上なら、少数の胴体/肘姿勢候補に可動域内で机から離れた解が存在した。机・物体の衝突や配置を変更せず、手先目標を変更した。
- 実装: `--grasp-height` と `--table-clearance` を追加。後者は9個の腕/指リンクと机の境界箱を使い、分離距離2 mm以上のIK候補を選ぶ。最初は終点だけだったが途中の接触が残ったため、現在関節値から目標までの12補間点へ拡張し、腕/胴体の差分を共同比率で縮小して最大0.03（rad/m）にした。有限個の候補/姿勢の確認であり、連続衝突判定・自己衝突・指の開閉経路を含む一般plannerではない。実際の接触力は毎制御stepで別に測定する。
- 運搬: `pick_place` を追加。接近→下降→閉じる15 step→持上げ→目標へ運搬→保持の手動段階列。運搬目標は実測した手先と物体の相対位置から計算し、resetしない。目標と正しい停止条件を途中で満たした場合も保持へ移る。PickCubeの目的は把持した物体を目標へ運ぶことであり、手放して配置した成果ではない。

| run（全て `runs/` 下） | seed | step / wall秒 | 観測 |
|---|---|---|---|
| `wsl-arm-pick-clearance-20260922-a` | 0〜2 | 545 / 9.722 | pitch15度/高さ12 mm、胴体固定。seed0は14.75 cm持上げ保持、seed1は下降IK不収束、seed2は把持後に上流成功で早期終了。途中の机接触は残った |
| `wsl-arm-pick-clearance-20260922-b` | 0〜2 | 750 / 8.899 | 胴体IKと終点距離確認を追加。全3例把持・持上げ保持、最終高さ増分14.26/14.78/13.59 cm。途中接触最大0/47.66/757.50 N |
| `wsl-arm-pick-place-20260922-a` | 0〜2 | 207 / 6.520 | 運搬追加で上流成功3/3。ただし全3例で終了時の身体絶対速度が0.2を超えていた |
| `wsl-arm-pick-place-newseeds-20260922-a` | 1000〜1002 | 215 / 6.488 | 固定方策で上流成功3/3。seed1002も絶対速度条件を超過。途中接触あり |
| `wsl-arm-path-static-20260922-a` | 0〜2 | 338 / 10.289 | 途中姿勢確認・指令縮小・停止判定補正後、正しい成功3/3、全記録stepで机接触力0 |
| `wsl-arm-path-static-newseeds-20260922-a` | 1000〜1002 | 432 / 13.385 | 同条件固定で成功3/3、全記録stepで机接触力0 |
| `wsl-arm-place-hold-20260922-a` | 1000〜1002 | 492 / 13.727 | 到達後20 stepを観測。3/3が全期間の把持・成功を維持、机接触力0 |
| `wsl-arm-place-hold-20260922-b` | 1000〜1002 | 492 / 13.809 | reset把持フラグ補正後も同じ物理軌跡・保持結果。最終目標誤差0.994/1.433/0.699 mm |

- 停止判定の補正: インストール済みManiSkill 3.0.1 Fetchの `is_static` は負の速度に絶対値を取らず、`qvel <= threshold` を比較していた。保存観測のqvelと照合すると、最初の運搬6例中4例は上流successでも身体絶対速度0.2を超過していた（0.2995/0.6079/0.3529/0.2733）。これらを正しい停止成功とは呼ばない。`fetch_pick.py` に既存PickCubeを継承した `APC-FetchPickCube-v1` を登録し、`--absolute-static` で選ぶ。指以外の身体関節絶対速度≤0.2、台車関節≤0.05と元の物体目標距離≤0.025 mでsuccessを計算。上流success/staticは別infoへ保存し、旧タスクを変更しない。
- 保持とreset: 既存 `HoldAfterSuccess` を補正PickCubeでも利用できるようにした。最後の3例は全20 stepでsuccess=trueかつgrasped=true。初期観測の全一致監査は一度失敗し、54要素のうちindex30の把持フラグだけ異なることを特定した。CPU接触問い合わせがreset直後に前episodeの接触を返すためで、このタスクの開いた指と離れた物体の初期配置ではfalseへ補正する。元の値は `upstream_is_grasped` に残す。`-b/compare_reset.py` で補正前後の全NPZがreset時の把持フラグを除き完全一致と確認。監査失敗時の `-a/audit.py` も保持した。
- 保存と集計: 今回は24 episode / 3,471環境step、学習更新0。`wsl-arm-place-hold-20260922-b/audit.py` と `audit.json` に全8実験の有限値、T/T+1、13次元行動と範囲、報酬・終了・物理手先誤差・補正後の成功判定・保持期間を照合した結果を保存。全同seedの初期関節/物体/目標は一致するが、補正前のreset把持フラグまで一致したとはしない。成功例も含め使用したseedは探索扱い。補正後の全記録stepで机接触0だが、制御step間の接触まで保証した値ではない。
- テスト: 新規の細切れテストは追加せず、既存腕機能テスト1本を運搬/保持/負の速度/連続resetまで拡張。実物理FK、IKと送信行動、段階をまたぐqpos連続性、目標距離・停止・終了を確認し、成功率は条件にしない。機能変更の節目で `wsl-arm-clearance-tests-20260922-a`（1 passed/12.95秒）、`wsl-arm-path-static-tests-20260922-a`（1 passed/17.20秒）、`wsl-arm-hold-tests-20260922-a`（1 passed/19.83秒）。最後のreset補正後は `wsl-arm-hold-tests-20260922-b` で **1 passed/18.21秒**。最後の最初のコマンドはvenvパスの `wsL` 大文字誤記でPython起動前に失敗したため、訂正して実行。テスト失敗や物理run失敗ではない。既知の非推奨/import警告のみ。
- 解釈/次の一点: 机接触、停止判定、reset観測の障害を修正して把持・運搬・保持まで続けた。手設計制御の動作確認であり、学習済み操作やスキル銀行の実証ではない。次の一点は、この方策の状態/行動軌跡を教師データとして小規模な操作模倣学習を試すこと。移動方策と操作を同一sceneで連鎖する実験は未実施。明確な修正案がある障害は以後も停止理由だけにせず修正を続ける。

## 2026-09-23 — 操作BCとlearner状態への2回の教師再ラベル

- 問い/変更: 手設計のFetch把持・運搬軌跡から、推論時にIKやstageへ戻らない小型操作方策を学習できるか試した。state54→64→64→11（8,395パラメータ）のBC、`fetch_pick_bc` rollout、成功教師だけの明示選択、learnerが実際に訪れた状態へ教師行動を別保存する `collect-operation-dagger` を実装。base 2出力は教師どおりゼロ固定。実行行動はNPZ、教師ラベルはstep診断に分離した。
- 条件/予算: 基点 `ef511ad` + 各runのdirty snapshot。WSL/Python 3.12/CPU、Fetch、`APC-FetchPickCube-v1`、13次元 `pd_joint_delta_pos`、制御20 Hz。手設計seed 20〜29、閉ループ比較と再ラベルはseed 20〜22、未見の最初の比較だけseed 1100〜1102。全て結果を見た探索seed。学習seed 0、幅64、主比較3,000更新。
- 既存runner再確認: `runs/op-bc-demo-20260923-a/` はseed 20〜22の3/3が配置後20 step保持、462 step、wall 9.995秒、例外なし。全軌跡が有限、state54のT+1と13次元行動Tが整合、base行動は厳密にゼロ。段階sampleは接近150/下降97/閉じ45/持上げ48/運搬62/保持60。
- 実装: `operation_bc.py`、`operation_dagger.py`、`configs/fetch_pick_bc.json` とCLIを追加。教師runのtask/controller/action shape、NPZ hash、数値、送信行動を検査し、checkpointのtask/control/schemaを再読込時に照合する。学習runはsourceごとの選択seed/hash、MSE、更新数、パラメータ数を保存する。

| run（全て `runs/` 下） | episode / step | 観測 |
|---|---:|---|
| `op-bc-demo-20260923-a` | 3 / 462 | 手設計3/3成功・保持。最初の教師データ |
| `op-bc-policy-20260923-a` | 3 / 1,050 | 1,000更新、未見seed 1100〜1102。成功/把持0/3、最短TCP距離0.154〜0.243 m |
| `op-bc-policy-trainseeds-20260923-a` | 3 / 1,050 | 同じ教師seedでも成功0/3。seed22のみ一時把持 |
| `op-bc-train3000-trainseeds-20260923-a` | 3 / 1,050 | 更新数だけ3,000へ増加。成功/把持0/3で改善せず |
| `op-bc-demo10-20260923-a` | 10 / 2,020 | 手設計8/10成功。seed27/28は350 step時間制限 |
| `op-bc-demo10-policy-trainseeds-20260923-a` | 3 / 1,050 | 全10軌跡で学習、成功0/3 |
| `op-bc-demo10-success-policy-trainseeds-20260923-a` | 3 / 1,050 | 成功8軌跡で学習、把持0/3、最短TCP距離0.041〜0.101 m |
| `op-bc-dagger1-20260923-a` | 3 / 1,050 | 上記learner状態を再ラベル。実行成功0/3 |
| `op-bc-dagger1-policy-trainseeds-20260923-a` | 3 / 1,050 | 成功教師+再ラベル、成功0/3だが把持・持上げ3/3。最短目標距離0.047〜0.063 m |
| `op-bc-dagger2-20260923-a` | 3 / 1,050 | 1回目learner状態を再ラベル。実行成功0/3 |
| `op-bc-dagger2-policy-trainseeds-20260923-a` | 3 / 1,050 | 2回分を単純集約。成功0/3、把持1/3へ退行 |

- 学習: 3軌跡1,000更新の教師MSEは0.123668→0.001232、3,000更新は0.000542だが閉ループ改善なし。成功8軌跡は1,320 sample、1回目集約は計2,370、2回目は3,420 sample。各3,000更新。1回目の最終MSE 0.001048、2回目0.001831。低い教師MSEを閉ループ成功とみなさない。今回のrunner実験は計40 episode / 11,932環境step、学習16,000更新（テストを除く）。
- データ/エラー: 各runはcompletedで `error.txt` なし。既知のVulkan/glvnd/Pinocchio/NumPy警告のみ。`op-bc-policy-20260923-a/analysis.json` と各比較runの `analysis.json` に有限値、距離、持上げ、把持、base行動を保存。失敗軌跡・2回目の退行も削除していない。
- 解釈: 1回のon-policy状態再ラベルは接近失敗を把持・持上げまで改善したが、配置条件には未到達。2回目の単純追加は改善せず、成功教師と各roundの教師stage/行動がstate54上で競合する可能性、均等混合比率の問題、memoryless入力不足をまだ分離できない。手動stageを推論入力に追加して成功させた結果ではない。操作スキル完成、自動銀行、汎化の証拠とはしない。
- 統合テスト: 操作機能全体の `tests/test_operation_bc_feature.py` 1本を追加・拡張し、実環境の手設計軌跡→初期BC→learner状態/教師ラベル分離保存→結合再学習→checkpoint単独rolloutを確認。最終実行は `runs/op-bc-dagger-feature-tests-20260923-b/`、**1 passed in 38.55s**。成功率や損失低下は合格条件にしていない。
- 次の一点/停止理由: 集約したsource間で、近傍のstate54に異なるstage・教師行動が割り当てられている量をまず測る。その結果なしに、再ラベル比率調整、履歴入力、明示stage selectorのどれを直すべきかが一意でないため、新しい学習runはここで止める。GPU、動画、未見seedでの再ラベル後評価、移動→操作の同一scene連鎖は未実行。

## 追記テンプレート

### YYYY-MM-DD — 変更点の短い名前

- 問い/今回変えた点:
- commit / config / robot・controller / seed・split:
- 環境step・episode・学習step・wall time等の予算と実績:
- runパス / 実データの観測 / 失敗した条件:
- 解釈（測定と推測を分ける）:
- 次に変える一点:
- 機能テストを実行した場合の結果 / 未実行と理由:

見込みと違ってもそのまま記録し、設定変更や小さい試行に戻ってよい。
