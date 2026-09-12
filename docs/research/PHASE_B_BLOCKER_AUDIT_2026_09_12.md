# Phase B ブロック監査と再開方策

日付: 2026-09-12。調査対象 commit: `6f54fdfe1639d44a466f05eab7061d813034749e`。
状態: **調査・方策設計完了。復旧 RG3 は FAIL のまま。研究ゲートは未解除。**

## 1. 結論

最初に行うべき作業は、最新 REC-004AC の **attention 計測の訂正と保存済み証拠の再集計**である。
その前に温度変更・大容量化・追加学習を選ぶと、誤った原因指標に基づいて次の修復を設計することになる。
本監査では実装と操作定義を照合し、正解参照位置の取り違えを確認した。
ただし通常実行の sequence EM はこのバグとは別に算出されており、訂正だけで性能 FAIL が PASS になるわけではない。

長期的な停止理由は三層に分かれる。

1. **復旧の停止:** 完全に読み込めるモデルは構築できたが、全15非SHIFT操作の性能条件を満たした child bundle がない。
2. **研究性能の停止:** 元の B2 検索・引数解決・SHIFT・実 K/C/N/R 統合の合格証拠がない。
3. **評価設計の停止:** 独立した未見 relation が不足し、既存学習の全候補 CE が holdout key にも更新を与える。

MIRROR_HALVES を直せば直ちに Task Inference へ進める、という依存関係ではない。
一方で、診断を重ねた成果は残っている。以下の出口を明示すれば、同じ原因探索を繰り返す必要はない。

## 2. 調査範囲と証拠の強さ

確認したものは、Phase B・Post-D2・Model Bundle Recovery の実行計画、適用する契約・addendum、
ADR-0074〜0125、復旧ディレクトリの28件の `summary.json`、主要判定 JSON、最新実装と隣接テストである。
REC-001〜004 など summary 以外に状態を記録するタスクは、ADR と個別の qualification 等を照合した。
すべての過去タスクを再実行したわけではない。古い封印評価の値は既存 ADR の記録として扱い、封印データを開いていない。

- 新しい確認: 実 operation の位置写像、実装定数の不一致、理想 attention の反例、top-k 集計の反例、保存 tensor のパラメータ数。
- 既存測定の確認: JSON にある PASS/FAIL、EM、候補未採用、RG3 未実行。
- 未実施: checkpoint の forward 再評価、学習、候補選択、bundle 作成、RG3/G4/G5、封印評価。
- 本文の「提案」は未実行・未承認。現在の個別タスク境界を変更しない。

再現用の [audit.py](../../runs/phase_b_blocker_audit/20260912/run_001/audit.py) と
[audit.json](../../runs/phase_b_blocker_audit/20260912/run_001/audit.json) に、実行環境・source commit・証拠 hash・反例・状態一覧を保存した。

## 3. これまで何が解決し、何が残ったか

| タスク群 | 実施内容・結果 | 今回の解釈 |
|---|---|---|
| B-C001〜003 / B1 | プロトコル整備後、明示 TaskSpec の未見 family lifecycle が5 seedsでPASS。novel EM 97.66%、recurrence EM 97.19%（ADR-0074） | APC lifecycle の成果は保存。Task Inference の成功ではない |
| B-C004〜005 / B2 | hard negative を導入。N=128 のL3 top-1=0.662、L4=0.504、known false plastic=3.75%でFAIL（ADR-0075） | 検索・引数解決・機能検証を先に切り分ける必要があった |
| B-C005D / R1 / R2 / G | development で ranking と verifier 修正がPASS。しかし新 sealed re-gate はL3=0.6555、L4 full-call=0.8883でFAIL（ADR-0076〜0079） | development の局所成功は新しい関係への移転成功を保証しなかった |
| D2-001〜006 | SHIFT 自体の機能不足、正しいIDでも不十分な候補を早期受理する穴、生成の非決定性、relationごとの原因を分離（ADR-0080/0081） | すべてをrouterやsupport分散で説明する解釈は維持できない |
| R3-001〜005 | 再現性G0・指標G2・厳密な有限look verifier G3は成立。relation G1は不足（ADR-0082〜0086） | 安全性の基盤は前進。availability・未見relationは別問題 |
| R3-006〜008 | COUNT↔BINDの局所scoring、SELECTのBCE/sigmoid不一致、BINDのvalue coverageに対処（ADR-0087〜0089） | 過去の親モデル上の成果。新Coreへの適用を自動的にPASSとはできない |
| R3-009〜010 | SHIFTは3/5 commitでGate FAIL。Core更新後に旧bankとの不整合、再build経路の10操作未学習、scorer/router組合せ問題が顕在化。G4 FAIL（ADR-0090/0091） | 復旧分岐が必要になった直接の理由 |
| REC-001〜003 | 来歴調査、fail-closed loader、全16操作build経路・予算を固定。RG0〜RG2成立（ADR-0092〜0094） | 「キャッシュを削除して再実行」では解けなかった基盤問題に対処 |
| REC-004 / A〜D | seed10のfresh-load成功。4操作が性能不足。Aで3操作のvalidation floorを確保、MIRRORが残る。位置biasで5初期化の平均EMが0.387→0.719に改善するも5/5未達（ADR-0095〜0099） | MIRROR中心の診断は妥当。ただし3操作も最終childでRG3を通ったわけではない |
| REC-004E〜J | oracle attentionで初期化ごとの異なる原因を確認。12k/18k延長でも全初期化合格せず。clean-v2軌跡でI03だけ一度もfloorを超えず、I05は終端で崩れる（ADR-0100〜0105） | 全員一律延長・最良init選びでは解決しない |
| REC-004K〜S / ENV1 | downstream rollbackとFFN/value相互作用、score-only・content prep解放、勾配・一更新の動力学を調査。SciPy依存問題はENV1で対処（ADR-0106〜0115） | 環境停止と研究性能停止は別。単純な勾配干渉・一更新overshootを主因にする証拠はない |
| REC-004T〜Y | 遷移を7500〜8000に局在。attention固定でもdownstreamのO1互換性は崩れる。CVOF共同凍結は互換性を保つが学習不足。trust region・機能drift指標も規定条件を満たさず（ADR-0116〜0121） | 時間的先行だけからscore劣化を崩壊原因と断定できない。共同凍結で防げるが最小必要集合を完全列挙した証拠ではない |
| REC-004Z / AA / AB / AC | Key/Value分離、pre-V residual、post-attention residual、score residual。互換性は保つがJ0改善条件を満たさない（ADR-0122〜0125） | 安定性の制御には成功。通常実行を回復させる証拠は不足。ACの一部原因指標には今回確認したバグがある |

主要出典: [Phase B ADR](../DECISIONS_PHASE_B.md)、[Post-D2 ADR](../DECISIONS_PHASE_B_B2_POST_D2_REPAIR.md)、
[REC Part 1](../DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_PART1.md)、
[REC Part 2](../DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_PART2.md)、
[REC active ADR](../DECISIONS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md)。

REC-004での非SHIFT未達は CYCLE_FOUR=0.890625、ROTATE_TRIPLETS=0.921875、
SWAP_ENDS=0.473633、MIRROR_HALVES=0.009766。
REC-004Aの規定selectionでは前3操作がそれぞれ1000/2000/2000更新でvalidation floorを通った。
MIRRORは6000更新でも0.6875で未達だった。
[REC-004実行表](../../runs/phase_b_b2_model_bundle_recovery/rec004/run_001/all_primitive_execution.json)、
[REC-004A selection](../../runs/phase_b_b2_model_bundle_recovery/rec004a/run_001/selected_recipe.json)。

I01〜I05は **同じCore/model seed10上のprimitive初期化5本**であり、REC-005が要求する独立Core/bankモデル5個ではない。
REC-004Iで4初期化が「軌跡のどこか」でclean-v2 floorを超えたことも、共通終端での5/5合格とは異なる。
既存の全5本条件・I01固定を今の結果を見て外すことは、本監査の提案に含めない。
[採用契約](../CODEX_TASKS_PHASE_B_B2_MIRROR_LENGTH_POSITION_BIAS.md)、
[clean-v2判定](../../runs/phase_b_b2_model_bundle_recovery/rec004i/run_001/pattern_classification.json)。

## 4. 今回確認した不具合と解釈上の問題

### 4.1 REC-004ACの正解key位置が誤っている — 修正優先度P1

位置番号は0始まり。実 operation は、長さ10について次の写像を持つ。

```text
出力位置: 0 1 2 3 4 | 5 6 7 8 9
入力位置: 4 3 2 1 0 | 9 8 7 6 5
```

[実装](../../src/apc/evaluation/mirror_parallel_score_residual_pilot.py) の114〜115行は
`REC004AC_CORRECT_KEY_P4 = 4`、`REC004AC_CORRECT_KEY_P5 = 5`。
正しくは0、9であり、隣接Z/AA/ABはその正しい値を使っている。
operationの `apply` に一意な入力 `0..9` を渡して独立に確認した。

影響するのは同ファイル1326〜1375行の、p4正解keyのmargin・probability・rank・recall、
base/total scoreのmarginとresidualによるmargin差である。
P5定数も誤っているが、現ACでは宣言以外に参照されていない。
理想的に正解key=0だけにscore=10を与えた反例でも、正しいmarginは+10なのに現実装では−10と報告する。

**影響しない経路:** J0/O1のsequence EM・出力位置token accuracyは `target_tokens` と比較している。
O1のattention生成も既存 `mirror_halves_position_map` を使っている。
compatibility/plasticity/strong floor/location advantageの判定はこれらのEM/accuracyを使う。
したがって今回のバグだけを理由に既存FAILを取り消さない。
訂正した実checkpointのmargin等は **未再測定**であり、ここでは数値を発明しない。

### 4.2 top-k recallとscore規模の定義にも補修が必要

ACは各headの正解順位を平均し、`<= 1.5/3.5/5.5` をtop-1/3/5 recallとしている。
例えばhead順位 `[1,2,1,2]` は現方式ではtop-1=100%だが、head単位のtop-1率は50%。
head単位、平均attention分布単位など、どの母集団のrecallを測るか事前に定義し、その定義に沿って集計する必要がある。

`norm_ratio_mean` はheadなしのresidualとheadありのbaseのノルムを割っている。
4headへ同じresidualを展開すると分子は2倍になる。またsoftmaxに影響しない行ごとの定数offsetも現ノルムに含む。
「residualの値が小さい」という観測は残るが、機能的なscore支配を示すには、同shape・valid mask・行中心化・正解marginの尺度を併記する。

### 4.3 「凍結base score」「構造的に不可能」は証拠を超える

ACのCVOFは凍結されているが、Q/K、KEY_CONTENT_PREP、POSITION_BIASは訓練対象である。
`S_base = QK/sqrt(d) + position_bias` 全体が凍結されているわけではない。
したがってADR-0125の「frozen base attention score attractor」は正確な機構説明ではない。

rank4・256追加parameter・I03・7500→8000の500更新という限定条件での不成功から、
低rank residual一般の表現不可能性や、score scaleこそが原因だとは結論できない。
さらにACのmarginは上記の誤座標を測っていた。
維持できる結論は「この設定では互換性は保持したが、規定のJ0改善を達成しなかった」までである。

### 4.4 パラメータ会計の訂正が必要

AC保存checkpointのtarget primitiveは **24,746 scalar**。
コードの凍結・復元maskに対応するCVOFは13,568、更新可能な残りは **11,178**。
現会計は凍結後に復元するparameterも `requires_grad=True` なので、24,746を全てactive trainableと数えている。
これは実行時のactive capacityとは別の指標である。

追加residentは比較基準により、Z比256、legacy比7,456（Key分離7,200＋residual256）。
7,456はモデル総数ではない。Coreと他15操作を含むシステム総数も別欄にする必要がある。
既存 `cost_accounting.json` のVRAM=45.40625 MiBとADR-0125の50.7 MBも一致していないため、元runのscopeを確認して追記訂正する。
本監査の再集計は保存tensorの数え直しであり、新しいGPUメモリ測定ではない。

## 5. 最初の一件: 計測訂正と修復判断の再確定

仮の次タスク名: **REC-004AC 計測訂正・保存checkpoint再分析**。
状態は `PROPOSED_NOT_AUTHORIZED`。新しい訓練ではなく、誤った診断からの分岐を防ぐための有限な訂正作業とする。

1. 正解参照先をoperationの位置写像から導出する共通関数へ統一する。長さ6/8/10、全出力位置、padding、複数headで独立のoperation対照を入れる。
2. 理想one-hot attentionが全位置で正解、誤写像attentionが不正解となる意味的テストを追加する。単に「定数が0」をassertするだけにしない。
3. ACの既存step7500/8000と保存済みprobeだけを使い、manifest/digestを再照合して訂正前後の指標を対にする。必要な中間state/tensorが保存されていなければ `UNAVAILABLE` とし、学習replayで作り直さない。
4. AC・Z・AA・AB・W・historicalを同じ入力上で比較する。各タスク独自のfresh validation値を横並びにして改善幅を計算しない。
5. EM/予測の不変を確認し、変更されるmargin/rank/recall・会計だけを新namespaceへ保存する。旧JSON、checkpoint、ADR本文は保存する。
6. 訂正後の証拠で修復仮説を一つ選ぶか、原因未確定として止める。full pytest / ruff / mypyはこの実装修正時に実行する。

完了条件: 正解写像・集計定義・source再現がPASS、EM差=0（同一実行条件の規定数値許容差内）、
新optimizer update=0、旧artifact hash不変、影響範囲と原因結論の更新が記録されること。
**この完了はRG3 PASSを意味しない。**

## 6. 次の修復方策: 条件付きの優先順位

### 第一候補: scoreの尺度・初期化・学習速度の不均衡を切り分ける

訂正後も「正解score marginが負で、residualの正解方向への作用が極小」が再現する場合に限り検討する。
追加parameterを増やす前に、既存score経路と新residualの相対scaleを変える一つの仮説を検証する。

例として、同じfrozen value/FFN経路・同じrank4のまま `S_total = a(t) S_base + b(t) DeltaS` の
事前固定scheduleを用いる小規模な対照実験を提案できる。
開始時は `a=1, b=1` とzero residualで既存forwardとのparityを確保する。
係数の決め方は学習用calibrationだけから定め、confirmation成績で係数を探索しない。
単なる温度変更はscoreの順位を直接反転させないため、entropy増加だけを成功としない。

提案時点の上限例は、基準・base scaleのみ・residual scaleのみ・両方の2×2、各500更新、合計2000更新。
これは既存承認budgetではなく、**新タスク契約で確定する提案値**である。
masked/中心化したhead別score、勾配、通常J0、O1、短いlengthの回帰を追跡し、
成功しなくてもrank/LR/更新数の探索を自動追加しない。

### 第二候補: 早期からの位置routingとvalue変換の役割分離

尺度を訂正しても有効な方向に改善できないなら、I03@7500への後付け修正に固執せず、
位置依存の操作に対してcontent scoreと位置routingが不要に競合していないかを問う。
既存の位置feature `i,j,n` だけを使う学習score経路とtask-blind value経路を、開始から分ける案がある。
既存position-biasの改善が、この仮説を検討する根拠になる。

これは新しいprimitive recipeの研究であり、本復旧契約の「既存recipeのみ」からの明示的なscope変更が必要。
oracleの位置写像をruntimeへ埋め込んでMIRRORを解くことは採用しない。
oracleは評価対照、必要なら別途宣言したtrain-only supervisionに限定する。
この段階へ進むかは第一候補の結果を見て別に判断し、同時に複数architectureを探索しない。

### 採用しない近道

- I04だけ、またはinitごとの最良checkpointだけを採用して既存5/5条件を通過扱いにする。
- 同じrecipeを理由なく全initでさらに延長する。
- MIRRORをregistryから外す、SHIFTの `COHERENT_LIMITED` 例外を広げる、0.95 floorを下げる。
- oracle O1=1.0を通常J0の性能として扱う。
- legacy共有cacheを書き換える、Coreだけ差し替えて依存bankを保持する。

新recipeを採用する場合は、固定endpoint・選択規則・独立confirmation・全init安定性・
Core別build/serializationを測定前に一体で登録する。
「I03の診断が改善したら終了」だけの契約では復旧へ接続できない。
pilotと全init再検証・候補採用は別段階とし、各段階を実行する許可範囲も明示する。
新recipeの条件を満たしても、過去REC-004D/G/HのFAILは書き換えない。

## 7. ブロック解除の依存関係

| 順序 | 解除する条件 | 必要な証拠 | 現在 |
|---|---|---|---|
| 1 | 原因指標の信頼性 | §5の計測訂正、既存EMの維持、同入力比較 | 計測バグ確認済み・訂正未実施 |
| 2 | MIRROR修正版の採用条件 | 固定recipeで必要な全init安定性、通常実行の絶対floor、独立confirmation | 未達 |
| 3 | RG3 | seed10の全16coverage、非SHIFT15操作各EM≥0.95、固定3修正との組立、protected12不変、fresh-process load | FAIL / 最新recheck未実施 |
| 4 | REC-005 / RG4 | seeds10〜14の独立Core/bank、全16×5の表、全非SHIFTfloor | 未実施 |
| 5 | REC-006 / RG5 | runtime bundle注入、明示repair preparation、router/scorer/formula/lambda整合、immutable lineage | 未実施 |
| 6 | REC-007 / RG6と研究G4 | 同じ親・入力のC0〜C5比較、実K/C/N/R、安全性・coverage・SHIFT等の独立判定 | 未実施 / 旧G4 FAIL |
| 別経路 | R3-002 / G1 | 十分な独立relationと露出管理。復旧成功だけでは解除不能 | PROTOCOL_INSUFFICIENT_RELATIONS |
| 7 | R3-011〜012 / G5 | 必須G1/G4等の解決、事前封印、その後一度のB2_PROTOCOL_V2評価 | blocked |
| 8 | B-C006〜007、続いてTask Inference | G5後にB3の計算量gate、さらにTask Inferenceの各gate | blocked |

性能条件の正は [復旧受入計画](../EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md) と
[親Post-D2計画](../EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md)。
RG6は正しく比較を実施できたという判定であり、研究G4がFAILでも成立し得る。

G1の解決設計はMIRROR修復と独立して進められるが、研究実行は別の明示タスクとする。
現状はL3共有keyによる結合を考慮すると、L3独立2＋L4独立4の計6単位。
既知失敗3単位をdevelopmentへ置くとclean単位は3で、validation≥2＋sealed≥2に一つ足りない。
同じrelationの逆方向・別seed・言い換えで数を増やすことはできない。
新しい独立relationの設計に加え、heldout keyを全クラスCEの勾配経路から外す修正と露出台帳が必要である。
候補数だけ増やす、または学習コードだけ直す、の片方だけでは両方の障害を解消しない。
[ADR-0083](../DECISIONS_PHASE_B_B2_POST_D2_REPAIR.md)、
[relation契約](../design-docs/B2_REPRODUCIBILITY_AND_RELATION_SPLITS.md)。

もしrelation-transfer要件そのものを研究目的から外すなら、限定された研究課題への変更として
別ADR・新protocolを定める必要がある。それは元のPhase BをPASSにして進む経路ではない。

## 8. 検証と変更範囲

- PASS: Python 3.12.13で監査scriptの全assertion。実operation写像、誤座標反例、recall反例、保存tensor会計、対象歴史ファイルの前後hash不変。
- PASS: 文書内ローカルリンク、更新diff、`git diff --check`。
- NOT EXECUTED: 全pytest / ruff / mypy。今回のtracked変更は文書のみで、製品実装・研究実験は変更していないため。
- NOT EXECUTED: 新学習、checkpoint forward、候補選択、RG3、REC-005以降、封印評価。
- 変更: 本報告、active recovery planへの状態追記、ADR-0126とdecision index。`src/`、`tests/`、`configs/`、旧runは変更しない。

この監査は「解除方法の提案」を完了した。性能の回復や研究ゲートの解除を実証したものではない。
