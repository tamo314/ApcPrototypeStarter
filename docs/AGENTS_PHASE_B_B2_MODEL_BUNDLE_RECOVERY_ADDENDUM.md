# Agent Addendum — Model Bundle Recovery

## 一タスク・一目的

指定された`B-C005REC-xxx`だけを実施する。重い学習を伴うREC-004／005はunit testから呼ばない。`--all`、暗黙next-task、自動sealed評価は禁止。

## 不変条件

`h_content=f(content)`、task/content分離、同一shared representation上のheterogeneous primitives、選択した候補だけの実forward、compact-first、shadow後のcommit、workspace解放、fresh-runtime recurrenceを維持する。

Coreを構築するときはREC-003で固定した既存shared/queryable recipeの範囲だけ更新可能。Core確定後は、依存bank構築中に勝手にCoreを変えない。Coreを変える必要が生じたらdependent bundleを無効化してscope reviewにする。

## 共有キャッシュは修正しない

既存`runs/`の履歴を削除・上書き・移動しない。既存scriptのdefault cacheを復旧先へ付け替えない。旧共有cacheは証拠として保存し、新しいrun／bundle namespaceを使う。既存ファイルの権限を一律変更することも不要。

キャッシュ欠落時の自動train、random initialization、他seedへのfallbackは禁止。新経路のloaderは不足をtyped errorで返す。互換性不明のlegacy artifactは、監査用限定モード以外で使わない。

## BuildとEvaluateを分離する

buildは明示commandで行う。calibration、router/scorerのjoint学習、variant選択はbuildまたはrepair preparationであり、evaluationではない。評価開始後はtrain API／optimizer／builderを呼べないようにする。

ただし実K/C/N/R試験でのNによるplastic learningは、**評価対象のruntime機能**であり禁止しない。許可されたtemporary parameters、consolidation candidate、bounded router updateだけをepisode単位で記録する。初期Core／bankを秘密裏に再構築する行為とは別。

## Sourceを越えて断定しない

現コード・checkpointは最初に確認する。レポートにない8+2+6の学習step数、optimizer、Core幅、decoder仕様を発明しない。既存レシピがなければ`RECIPE_UNAVAILABLE`として停止し、候補を報告する。

資料に記された関数名がrepoで移動していれば実pathを記録する。`run_unified_oracle_causal_benchmark`等の名称を根拠に、未確認の関数signatureを作らない。

## 診断用oracleの範囲

正しいcall・引数を与えるdirect executionは評価上限として使用できる。generatorのoracleをruntime primitiveに置換しない。K/C/N/R labelはevaluatorのみ。explicit TaskSpecは従来どおり可視で、Task Inferenceを新設しない。

## データ・モデルの独立性

development model cohortは10～14を原則とし、pilotは10。既存artifactがなければ同じseedを魔法の復元IDとして扱わない。data／model／repair seedを別欄にし、`seed % 5`によるparent切替を禁止する。

validation15～19、sealed30～34という既存名はデータ区分として実契約を監査する。validation seedで勝手に新モデルを学習しない。過去sealed0～4／20～24は調整に使わず、新sealed30～34の出力を先に見ない。

## 完了報告

task completion、復旧Gate、研究Gateを別欄で示す。`COMPLETE`を性能PASSの代わりにしない。欠損／未実行／reference未確定はnullや専用statusにし、0失敗として埋めない。

全タスクでfocused test、`python -m pytest -q`、`python -m ruff check .`、`python -m mypy src/apc`を記録する。repoが新canonical commandを定めていれば従う。共有環境の他processを停止しない。未実行・resource不足は明示する。

一つの失敗が次タスクの前提を壊すなら停止する。報告のみのREC-008は失敗地点から作成可能だが、未完taskを通過扱いにしない。
