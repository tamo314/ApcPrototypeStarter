> **Archive status — 2026-09-13 / ADR-0149:** `CLOSED_ARCHIVED`; `NEGATIVE_CONCLUSION_TERMINATED_CURRENT_ARCHITECTURE`.
> Current authority: [final evidence ledger](results/PHASE_B_CLOSEOUT_EVIDENCE_LEDGER.md#terminal-state). The entire original text below is historical, including proposed/active statuses and permission clauses.
> No Phase-B experiment is queued or authorized by this document. RG3=`NOT_EXECUTED`, REC-005=`BLOCKED`, G1=`STOP`, G4/G5=`BLOCKED`; candidate_selected=`null`, child_bundle=`null`, bundle_write=`false`; closeout sealed-data/model-output access=0.
> REC-006--008, R3-011/012 and B-C006 onward are archived non-executions due to upstream STOP, not backlog.

# Experiment Plan — B2 Post-D2 Repair / Protocol v2

**区別：本書のGate v2・数値・予算は今回の事前提案。旧Gateや過去実績ではない。**

## 1. Primary research questions

1. 新generatorは別processでも同じ評価例を生成するか。
2. 修正学習のrelation exposureを制御した条件で、COUNT↔BINDの修正が成立するか。
3. SELECTのencodingとBINDのargument scorerを分離して修正できるか。
4. 正family/argsでも機能不足のcandidateを受理せず、十分なcandidateのreuseを維持できるか。
5. SHIFT実装をコンパクトなまま改善し、version置換後も旧機能とfresh-runtime recurrenceを守れるか。

Task Encoder / query projectionの全面改造と、一般自然言語理解は研究対象に含めない。[S1: ADR-0081]

## 2. 実験の共通単位

- 5個以上の固有model checkpoint。parent/training seed/repair seedを追跡する。
- data seedはその内側のreplicate。同じcheckpointをN・levelごとに独立モデルと数えない。
- primary queryは1 episodeあたり256独立例、referenceはcall/task distributionあたり4096独立例を初期提案とする。
- 元B2の矩形：5 model seeds × N={16,32,64,128} × L0–L4 × {SHIFT,SELECT,COUNT,BIND} = 400 cells。各cellに16独立support episodesを初期提案する。400 cellsを400独立モデルと呼ばない。
- relation-transferの追加セル数はR3-002のcatalogから確定し、元400セルに混ぜず別記する。
- nominal suiteのcell/episodeを、reference EMや結果を見て削除しない。referenceで不足と判明した場合もnominalのunconditional指標に残す。
- 完成済みrunに追加seedを都合よく足すのではなく、全seed・quotaをseal前に固定する。

これらの予算が16 GB GPU上で実行可能か、R3-003/004で事前確認する。referenceは同一call/version/distributionでcacheし、candidateとbatchを逐次処理する。長時間sweepはmilestone実験でありunit testには入れない。

## 3. G0 — 再現性

R3-001のPASS条件：

- 4種類のPYTHONHASHSEED設定による独立processで同じ入力hash。
- generator呼び出し順・resume・worker変更でsample IDごとの入力一致。
- primary pathにbuiltin hashによるseed導出が残っていない。
- metadataにgenerator versionとデータ・モデルhashが存在。
- 旧artifact非破壊。復元不能な過去入力が明示される。
- モデル重み/threshold変更=0。

旧runの完全再現不能は正直に保存する。新入力の再現性検証が未完ならPASS不可。

## 4. G1 — Relation分割とbenchmark妥当性

R3-002のPASS条件：

- DEV / VALIDATION / SEALEDがalias/逆relationを含めて非重複。
- 各partitionに原則2個以上のreal learned relation group。実bankで不足する場合はprotocol不足として停止。
- 新repair lossのCE/replay/miningに関するexposure auditが通る。
- parent checkpointの既往露出が明示され、repair-time holdout以上の主張をしない。
- COUNT↔BIND等のtargeted regressionと、未使用relationへのtransferが別suite。
- L3がreal learned competitorか、synthetic score attackかを区別する。
- sealed出力の先行閲覧=0。

sample数を増やしただけのseed holdoutへ代替しない。未見group不足時にsilently「mining holdout」でPASSしない。

## 5. G2 — Metrics/Statistics Contract

R3-003のPASS条件：

- 正ID/argsだが不十分なcandidateを、fixtureでunsafe acceptanceとして検出。
- candidate inadequacy、H内の不足、予算切れ、未確定が別state。
- referenceのUNRESOLVEDをadequate/inadequateへ強制二分しない。
- 新verifierのalpha配分・look数・candidate上限が検証可能。
- legacy metricとv2 metricを同じ名前で混在させない。
- 空分母・abstention・unknown referenceを含むmetric testsが通る。
- reference/support/queryのアクセス分離が通る。

Runtime verifierはまだ旧版のまま。G2は仕様・offline計算の妥当性であり、性能の実証ではない。

## 6. G3 — Verifier safety / calibration

### 6.1 理論・数値契約テスト

primary：tau=0.95、最大5 candidates、looks=[32,64,128,256,512]、受理誤り予算0.01、棄却誤り予算0.01。詳細は[統計設計](design-docs/B2_FUNCTIONAL_ADEQUACY_V2.md)。

固定pのBernoulli streamに対し、finite-lookのexact dynamic programmingまたは独立の十分なMonte Carlo検証を用いる。数値的に期待と合うかを調べるp gridは同設計の§8に従う。

- `p<0.95`で誤受理確率のunion-bound契約を満たす。
- `p>=0.95`で誤棄却確率の契約を満たす。
- p=0.95近傍でUNCERTAINが多いことを隠さない。
- availabilityの契約テストはp={0.99,0.995,1.0}で `P(ACCEPT)>=0.97`。
- p=0.98等についても曲線を報告するが、有限budgetで常にACCEPTするよう修正しない。

確率計算自体とAPC実測を別表にする。Monte Carloの場合は推定誤差を示し、結果が閾値近傍ならPASSを断言しない。

### 6.2 Neural candidate validation stress

既存のwrong-callに加え、正family/argsでreference不足の候補を含める。旧不十分SHIFTは必須だが、同じ保存traceの繰返しを独立試行に数えない。

- `inadequate_call_accept_rate <= 0.01`。
- `same_identity_unsafe_accept_rate <= 0.01`。
- `wrong_call_accept_rate <= 0.01`。
- query/reference leakage、未許可更新、未選択forward=0。
- 同じidentityの不十分candidateについて、独立support drawを合計300以上確保する。per-model、per-candidateの分母を表示する。

上記は経験的Gateであり、0件だから母集団率が0だとは言わない。95% interval、対象数、reference確定率も報告する。referenceで判定可能な不十分candidateがない場合は `NOT_ESTIMABLE` としてPASSしない。

R3-005では旧不十分SHIFTに対する低coverage自体はFAILではない。正しい拒否/未確定化を測るタスクである。通常運用のavailabilityはG4/G5で別に測る。

## 7. 局所repair Gates

| 対象 | Validation primary | 回帰/不変条件 |
|---|---|---|
| COUNT→BIND / BIND→COUNT | 各方向top-1>=0.95、top-5>=0.99 | 正常routing低下<=1 pp |
| SELECT encoding | full argument accuracy>=0.95、full call top-1>=0.90 | 他args低下<=1 pp |
| BIND scorer | full argument accuracy>=0.95、full call top-1>=0.90、family>=0.98、top-5>=0.99 | 他args低下<=1 pp |
| SHIFT execution | mean query EM>=0.99、全modelがREF_ADEQUATE | 旧条件/他task低下<=1 pp、fresh reuseでadaptation/temp=0 |

すべて5 model checkpoint以上。meanだけで失敗seedを隠さず、worst seed・全内訳を報告する。L4の引数やrelationの主張は該当operationに限定する。

R3-009で容量増加が必要だと分かった場合は、その結果を保存し、別ADR/明示taskで評価する。既定のcompact予算を黙って越えない。

## 8. G4 — Development統合・回帰

### Nominal B2

- N=128のL0–L2 full-call top-1>=0.98、L3>=0.95、L4>=0.90。
- 全levelのtop-5>=0.99。
- L4 family top-1>=0.98、各parameterized operationのargument accuracy>=0.95。
- `unconditional_query_EM >= 0.95`。
- `execution_coverage >= 0.97`。
- `avoidable_plastic_rate <= 0.02`。
- `adequate_solution_nonreuse_rate <= 0.03`。
- `reference_unresolved_rate <= 0.05`。これを超えた場合は評価力不足として記録し、未確定例を削除しない。

L3は全体macroに加えてCOUNT↔BINDの各方向を必須層とする。SELECT→BINDが有効なevaluation stratumとして解消されていない場合は、必要なrelation評価はUNRESOLVEDのままであり、全L3成功とは言わない。

### Safety

G3の3つのfalse-acceptance指標を満たす。stress suiteのquery EM/coverageにnominalと同じ下限は要求しないが、stressをnominal集計に混ぜて意味を変えない。

### Legacy K/C/N/R

既存の合法な短縮streamを同じv2入力でbefore/after比較する。構成と凍結checkpointをR3-004で登録する。

- 旧task EMとroutingのmean低下<=1 pp、worst-operation低下<=2 pp。
- Cで根拠のない新primitive追加=0。
- Rでadaptation steps=0、temporary params=0、再commit=0。
- 成功Nのpromotionはそのcapabilityにつき1回。失敗Nを除いた成功例だけのEMと、全予定Nの結果を区別。
- workspace leak、query/reference leakage、無許可パラメータ更新=0。

新verifierのUNCERTAINがlegacy availabilityを大きく下げた場合もFAILとして保存する。安全性を下げて旧coverageへ戻さない。有限budgetの選択問題として次の局所taskで扱う。

## 9. G5 — New sealed B2 Protocol v2

R3-011で封印した5個以上の完成model、未使用relation/入力、same-identity stressをR3-012で評価する。G4のnominal性能・safety・整合性基準をそのまま用い、加えて以下を要求する。

- repair-relation transfer suiteのL3 macro top-1>=0.95、各held-out relation group>=0.90、top-5>=0.99。
- targeted regressionとtransferの両方を報告し、修正学習したpairを未見成功として数えない。
- same-identity stressの十分な分母とreference確定がある。
- 全cell/seedの実行完了。planned/executed/missing countsが一致する。
- model/primitiveはmatrix中に更新しない。SHIFT新versionのfresh-runtime再学習=0。
- protocol・dataset・component hashesが封印と一致する。

**全必須項目PASSでのみG5 PASS。** empty metric、UNRESOLVED、未実行項目があるときはPASSしない。旧B2のknown-task false plasticも報告するが、正しい候補が不足していることを無視して新指標へ代入しない。

## 10. 計算量の表示

旧「mean support<64」は廃止した新primary設計と明記する。次を別々に表示する。

```text
support_unique_examples_per_episode
support_evaluations_per_candidate
support_evaluations_per_accepted_candidate
support_evaluations_per_rejected_candidate
support_evaluations_per_uncertain_candidate
sum_candidate_example_evaluations_per_episode
primitive_forward_calls_by_stage
recipe_evaluations / composition_depth
median/p95 decision latency / final execution latency
resident/active/temporary parameters
peak VRAM / analytical FLOPs assumptions
```

primary direct verifierの上限は5×512=2560 candidate-example evaluations/episode。compositionを加えるときは別予算を封印する。top-5を全て実行したことを「top-1 sparse compute」と説明しない。

CPU計算の統計判定時間だけをend-to-end latencyと呼ばず、GPU同期、support生成/転送、candidate実行の計測範囲を明記する。効率改善の体系的探索はB-C006の課題であり、本系列では過剰コストを隠さないための最低限の計測に限定する。

## 11. 比較・判定の原則

- 新generator値を旧run再現と呼ばない。
- 旧checkpoint欠落時に「同じseedだから同じ重み」と仮定しない。
- REF_INADEQUATEの割合が下がっただけでverifierの安全性が上がったと主張しない。
- candidateをrepairした効果とverifierを変えた効果を同一差分に混ぜない。
- すべての候補がUNCERTAINで出力しないsystemを、安全だから全体成功としない。
- 非同定・invalid argument・同値call・source不足はperformance failと別に計数する。ただし該当例を削除してgateを有利に変更しない。

## 12. 新しい結果ラベル

```text
G0/G1/G2: INFRASTRUCTURE_OR_PROTOCOL_PASS / FAIL
Local repair: VALIDATION_PASS / FAIL / NEEDS_SCOPE_REVIEW
G4: DEVELOPMENT_INTEGRATION_PASS / FAIL
G5: B2_PROTOCOL_V2_PASS / FAIL / UNRESOLVED / RESOURCE_BLOCKED
```

G5 PASSは「B2修正サブフェーズの新protocolで通過」であり、Phase B全体の完了、128 learned semantic skills、Task Inference成功を意味しない。B-C006が次に実行可能となるだけである。
