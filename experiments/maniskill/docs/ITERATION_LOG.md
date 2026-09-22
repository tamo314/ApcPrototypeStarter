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
