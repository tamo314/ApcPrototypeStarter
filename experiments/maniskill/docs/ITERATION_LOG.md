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
