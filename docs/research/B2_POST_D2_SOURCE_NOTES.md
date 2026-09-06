# Source Notes — B2 Post-D2 Repair

## A. 添付から得た事実

**[S1]** ユーザー提供 `貼り付けたマークダウン（1）(9).md`。内容：ADR-0080/0081、2026-09-06。

- [原文コピー](evidence/PHASE_B_D2_ADR0080_0081_SUPPLIED.md)
- SHA256: `8dd92c336900683e96d5a907e4f921e0dcf252bd318a7b5310d091113cedc5b6`
- 実リポジトリでは `docs/DECISIONS_PHASE_B.md` と当該ADRを照合する。

| 計画に用いた事実 | 出典箇所 | この計画での扱い |
|---|---|---|
| hash(operation)由来の非決定性、2つのgenerator経路 | ADR-0080 evidence 5 / consequences | R3-001で独立修正 |
| SHIFT reference EM 0.9033–0.9248、model_seed=4 | ADR-0080 evidence 1 | 機能不足の歴史的証拠。新referenceとは別 |
| 再構成4条件でunsafe reuseが2件 | ADR-0080 evidence 2–4 | verifier仕様の修正対象。元run同一episodeとは主張しない |
| wrong functional acceptance指標の盲点 | ADR-0080 consequences | metrics v2を別versionで設計 |
| z/q probeは成功、COUNT↔BINDのkey/scoringに局所化 | ADR-0081 findings | R3-006のみ許可。z/q全面変更は除外 |
| SELECT→BINDはUNRESOLVED | ADR-0081 findings | 測定妥当性を確認し、成功を捏造しない |
| relation holdout再設計を第一優先 | ADR-0081 recommendation | 再現性修正後、全trainingの前にR3-002 |
| SELECT encoding / BIND scorer / SHIFT functionの問題 | ADR-0081 findings | 別タスクとして修正 |
| D2はdiagnostic-only、B-C006はblocked | ADR-0081 consequences | 新Gateまで維持 |

ADR-0080は元seeds 0–4に関するADR-0076の帰属を保持している。本パックはその歴史的記述を上書きしない。同時に、今回提供された資料だけで元15件のreference adequacyを再確認したとも主張しない。

## B. 今回の新規提案

以下は添付の実測結果ではなく、依頼に対して作成した計画である。

- タスクID B-C005R3-001～012、実装順、追加module/CLI案。
- stable SHA256 seed導出、dataset/checkpoint/exposure manifestの詳細契約。
- targeted regressionとrepair-time relation transferの分離、逆pair grouping、loss exposure監査。
- REF_UNRESOLVED、runtime UNCERTAIN、scope限定のlibrary不足の区別。
- finite-lookで補正したexact bounds、受理/棄却の別alpha予算。
- support 32–512、reference 4096、query 256、16 episodes/cell、Gate v2の閾値。
- 同familyのversioned SHIFT置換とcache invalidation。
- 元B2指標とのbridge reportingとB2_PROTOCOL_V2判定。

値は検証前の事前設計として明示する。実装で実現不能/不適切と判明した場合は実験を通ったことにせず、seal前に別ADRと明示scope変更で扱う。

## C. 技術契約のために確認した一次資料

参照日：2026-09-06。外部資料はAPCの実験結果を裏づけるものではなく、seedと統計手法の技術仕様だけに用いた。新しいモデル・API・frameworkへの変更は要求していない。

**[S2] Python 3.12 Data model / hash randomization**  
https://docs.python.org/3.12/reference/datamodel.html#object.__hash__  
文字列等のhashがprocessをまたいで同一とは限らないことの確認。

**[S3] Python 3.12 hashlib**  
https://docs.python.org/3.12/library/hashlib.html  
SHA256などのdigestを使用するための標準API確認。

**[S4] NIST Dataplot — Proportion Confidence Interval**  
https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/propconf.htm  
二項比率のexact confidence limitsの根拠。

**[S5] SciPy — BinomTestResult.proportion_ci**  
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html  
exact methodの参照実装を用いた照合。実repoのinstalled versionとAPIを確認し、文書参照のために依存versionを勝手に更新しない。

**[S6] Howard, Ramdas, McAuliffe & Sekhon — Time-uniform, nonparametric, nonasymptotic confidence sequences**  
https://arxiv.org/abs/1810.08240  
逐次的な観測で時点横断の有効性を考慮するための背景。今回のprimaryは同論文のアルゴリズムの実装ではなく、事前固定finite looksのexact boundsとunion boundである。

## D. 計算例

Adequacy設計内のall-success lower boundや必要success countは、明記した式・alphaからの計算例である。APCを実行した結果ではない。作成時の数値sanity checkはパック末尾のvalidation JSONに記録する。
