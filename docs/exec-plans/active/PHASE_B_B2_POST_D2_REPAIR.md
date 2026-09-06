# Phase B / B2 — Post-D2 Evidence-Aligned Repair

**Status: proposed / not executed.** タスク：`B-C005R3-001`～`B-C005R3-012`。

## 1. 研究上の位置づけ

第二診断は完了し、ADR-0081はOption C（semantic-relation holdout再設計）を次の研究上の第一優先に指定した。ADR-0080は、それと独立したbenchmark生成の非決定性修正を推奨している。本サブフェーズでは、再現性を前提としてOption Cを実施し、その後、証拠で支持された局所修正を行う。[S1: ADR-0080 evidence 5 / consequences; ADR-0081 recommendation]

詳細：[タスク文書](../../CODEX_TASKS_PHASE_B_B2_POST_D2_REPAIR.md)、[実験計画](../../EXPERIMENT_PLAN_PHASE_B_B2_POST_D2_REPAIR.md)、[出典](../../research/B2_POST_D2_SOURCE_NOTES.md)。

## 2. 維持すること

- `h_content = f(content)`。content encoder/decoderをtask-conditionedなsolverへ戻さない。
- explicit TaskSpec条件を維持。Task Inference、pretrained LM、RL、新しいcontent representationは対象外。
- `PrimitiveCall = physical primitive + arguments`。引数ごとのpersistent primitive複製は禁止。
- 候補のidentity、候補実装のadequacy、ライブラリ範囲の不足を別概念として扱う。
- functional verification → compositionの確認 → 必要時compact-first plasticity → shadow validation → commit/releaseを維持。
- 評価時はAPC重みを凍結。修正学習は各タスクで許可した機構だけ。
- 過去のFAIL、ADR-0075～0081、runsは保存。新データversionの結果を過去結果の上書きとして扱わない。

## 3. タスク系列

| ID | 主目的 | 主に変更できる範囲 | 境界 |
|---|---|---|---|
| R3-001 | benchmarkの再現性修正 | seed導出・生成version・manifest | G0 |
| R3-002 | relation単位の分割 | 評価プロトコル・露出台帳 | G1 |
| R3-003 | adequacy指標・統計契約 | offline指標・schema | G2 |
| R3-004 | 同一入力上の凍結対照 | 評価runnerのみ | 実験整合性 |
| R3-005 | unsafe reuseを防ぐverifier | verifier・薄いsafety adapter | G3 |
| R3-006 | COUNT↔BIND key/scoring | 対象key/scoringのみ | 局所Gate |
| R3-007 | SELECT argument encoding | SELECTのencodingと必要最小限のhead | 局所Gate |
| R3-008 | BIND argument scoring | BIND head/compatibilityのみ | 局所Gate |
| R3-009 | SHIFT functional generalization | 別candidateの学習・安全な置換 | 局所Gate |
| R3-010 | 修正の統合・旧機能回帰 | integrationのみ | G4 |
| R3-011 | 新Gateの事前封印 | manifest・preflight | 封印 |
| R3-012 | 新しいB2 Gate v2 | 評価・最終ADR | G5 / STOP |

上表の省略IDにはすべて `B-C005` を付ける。依存は表の順。各タスク完了は次タスクの自動実行を許可しない。

## 4. 何を修正しないか

`z_task`/`query_proj`の全面改造は行わない。報告上はCOUNT↔BINDのkey/scoringが局所化されており、SELECT→BINDは未解決である。後者を前者と同じ原因だと仮定しない。SHIFT/COUNTの引数pathは回帰対照として保持する。[S1: ADR-0081 findings]

SELECT→BINDの未解決controlはR3-004で最小限の妥当性確認だけを行う。未解決のまま残った場合でも隠さず記載し、そのrelationの科学的成功主張はしない。最終Gateに必要な層が測定不能ならG5はPASSにしない。

## 5. 新しい判定境界

```text
候補の参照adequacy：ADEQUATE / INADEQUATE / UNRESOLVED
runtime verification：ACCEPT / REJECT / UNCERTAIN
runtime envelope：EXECUTED / NEEDS_MORE_EVIDENCE / NO_VERIFIED_SOLUTION
```

UNCERTAINは第4の学習済みcontroller actionではなく、既存3-action controllerへ不正な確定証拠を流さないための外側の状態である。UNCERTAINを0点やREJECTへ潰して新規性とみなさない。plasticityへ進む前には既存composition経路も確認する。

## 6. Gateの位置づけ

G0/G1/G2は実験基盤・契約の妥当性でありモデル性能を主張しない。G3は受理policyの安全性。G4はdevelopment統合。G5は新しい封印条件での正式評価。

G5は旧Gateを置換して過去FAILをPASSにするものではない。旧指標を併記し、新しい `B2_PROTOCOL_V2` の結果として記録する。G5がPASSして初めて、B-C006を「実行可能だが未着手」に変更する。

## 7. 計算資源と中止

単一16 GB GPUを前提に、候補・reference評価をchunk化し、中間テンソルを全bank分保持しない。unit testsはCPUで軽量にする。milestone sweepは明示されたタスクでのみ実行する。

budgetを超えたとき、seed・relation・失敗candidateを削って合格させない。未実行部分は `NOT_RUN`、資源制約は `RESOURCE_BLOCKED` として停止する。全Phase A/A.1/A.2の再学習・再実行は要求しない。
