# Source notes — REC-004E

## 資料に基づく内容

[S1: ADR-0099](evidence/REC004D_ADR0099_SUPPLIED.md)は今回の根拠である。Evidence 6–9のpaired gain、長さ別値、0/5達成、bias-zero介入、Evidence 1の文章訂正、Evidence 11–13のsmoke／テスト結果、Consequencesのblock状態を継承した。

[S2: 旧REC-004D指示](evidence/REC004D_TASK_SUPPLIED.md)から、元の位置式、I01〜I05、原artifact保持、scope、旧採用条件とRG3の区別を継承した。

[S3: position bias v1設計](evidence/REC004D_POSITION_BIAS_V1_SUPPLIED.md)は、4特徴、ReLU32、192 parameters、head共有、mask／layout、初期化、保存契約の配布時仕様である。現repoの実装確認に代わるものではない。

原snapshotは提供ファイルのbyte copyであり、解釈の訂正を原文へ混ぜていない。S1本文の「length 10-specific」を黙って書き換えず、新taskでは長さ7〜9の不足も同じ表に存在すると別記した。

## 数式からの整理

330位置対は長さ6〜10のn²の和、21450は330×13時点×5モデルである。

softmaxの一行に共通な定数を加えても分布は変わらないため、有効key集合上のrow-centered scoreを併記する。これは学習済みモデルの新測定結果ではない。

固定n,iでは旧phiはjの一次式になる。ReLU活性集合が一定の区間ではbiasも一次式となる。ただし既存scoreとの合算により動作は変わるので、これだけで全operatorの表現限界とは言えない。

## 今回新設した設計

REC-004Eを学習なしの残差診断＋次修正契約の作成に限定すること、score観測32例／長さ、介入用256例／長さ、J0〜J6の固定条件、出力仕様は今回の提案である。S1が測定済み・承認済みだった設定として扱わない。

alphaや長さfeatureのforward変更は反実仮想の診断だけで、本番feature変更・L_ref変更・追加訓練を許可しない。最大1件の次修正案は実装担当者が実artifactを読んで作成するが、実装・実験には改めてユーザーの明示指示を必要とする。

## 未確認事項

このパックの作成時点では、ユーザーの実repo・raw checkpoint・position grid・学習曲線のJSONは独立検査していない。とくにL_ref、score range、ReLU活性、終盤収束は不明である。それらの原因を仮定して修正機構を決めない。

外部研究・最新APIへの依存を新設していない。実装はrepoの固定ライブラリversionと現APIを確認し、upgradeを行わない。
