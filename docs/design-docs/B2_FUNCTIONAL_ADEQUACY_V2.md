# Functional Adequacy v2 — Safety, Uncertainty, and Scoped Novelty

**新規設計。旧verifierの実装結果ではない。** 目標adequacyは `tau=0.95` のまま維持する。

## 1. ADRから確定した問題

ADR-0080では、正しいSHIFT callであっても独立reference EMが90.33～92.48%で、その候補が少数supportの点推定により受理された。旧wrong-candidate acceptance指標はこの失敗を捕捉できなかった。仕様どおりの実装だったことも確認されている。[S1]

本仕様はidentityと機能adequacyを分け、早期受理も早期棄却も証拠に基づける。旧Wilson上限ルールは `legacy_asymmetric` として対照に残す。

## 2. 三つの階層

### Candidate

`call_key = primitive_id + primitive_version + canonical_arguments + task_distribution_id`。
候補のreference adequacyは、そのcallがそのtask distributionで十分な性能を持つかで決める。候補名から決めない。

### Search space / library scope

`H` はmanifestに列挙した既存direct callsと合法なcomposition recipesの有限集合。

- 一つでも十分な解が確認できれば `FOUND_ADEQUATE_SOLUTION`。
- Hの全要素を評価し、全てが不足と確認できた場合だけ `INADEQUATE_WITHIN_DECLARED_SCOPE`。
- 未評価、未確定、探索打切りがあれば `UNRESOLVED_WITHIN_SCOPE`。

一つのSHIFT候補が不足したことを、bank全体の新規性とみなさない。全bank・無限深度compositionへの不可能性を主張しない。

### Action

`candidate REJECT` はcontrollerの `PLASTIC_SEARCH` そのものではない。直接候補に失敗したら、既存composition経路を確認する。探索予算が尽きた状態は `NO_VERIFIED_SOLUTION_WITHIN_BUDGET` と記録し、証明済みのnovel operationと混同しない。

## 3. Data rolesと確率的前提

学習、validation、runtime support、final query、offline referenceは別stream・別observation IDにする。reference/query targetをruntime候補選択・停止へ渡さない。

primaryの区間計算は、凍結候補を同じtask distributionから得た独立な例で評価するBernoulli EMを前提とする。task argument/length分布を途中で変えない。token accuracyを試行数として使わず、一つのsequence exact matchを一試行とする。

IID samplingで偶然同じcontentが現れることと、同じ保存例を複数roleへ流用することは異なる。自然衝突の率、role ID、内容fingerprintを記録する。重複の有無で恣意的に成功/失敗例を再抽選しない。without-replacement、相関したbundle、固定層化を採用する場合は、このbinomial契約がそのまま使えると仮定せず、別の妥当性確認が必要。

候補はsupportの正誤を観測する前に固定し、top-k=5の順序も固定する。同一supportを見て追加候補を最適化することはprimaryでは禁止。必要なら新しい独立検証streamと追加の誤り予算を宣言する。

## 4. Primary bounded verifier：有限lookで補正したexact binomial bounds

### 4.1 採用理由と出典

Clopper–Pearson型exact boundsは二項モデルに基づく区間であり、固定標本の区間は[S4/S5]に従う。逐次的な観測では通常の固定時点区間をそのまま繰り返すだけでは不十分であることから、ここでは**事前固定した有限lookへのunion bound**を採用する。confidence sequenceへの全面変更は、この修正の必須条件にしない。[S6]

以下の配分と運用は本パックの設計であり、文献の推奨値という意味ではない。

### 4.2 初期proposal

```text
tau = 0.95
max_candidates = 5
looks = [32, 64, 128, 256, 512]
alpha_accept_episode = 0.01
alpha_reject_episode = 0.01
```

各candidate×lookに等配分する。

```text
a_A = alpha_accept_episode / (max_candidates * len(looks)) = 0.0004
a_R = alpha_reject_episode / (max_candidates * len(looks)) = 0.0004
```

使わなかったcandidateやlookの予算を、結果を見て再配分しない。directとcompositionを同一のepisode保証へ含める場合は、総候補数を事前に増やして再配分する。direct5件の保証を無制限recipe searchへ流用しない。

### 4.3 区間と停止

`k` successes / `n` trialsに対し、片側のexact limitsを使う。

```text
L = 0                                          if k == 0
L = BetaQuantile(a_A; k, n-k+1)                 otherwise
U = 1                                          if k == n
U = BetaQuantile(1-a_R; k+1, n-k)               otherwise
```

```text
L >= tau  -> ACCEPT
U < tau   -> REJECT
otherwise -> UNCERTAIN
```

最大supportまでUNCERTAINなら**UNCERTAINのまま返す**。点推定による強制ACCEPT、信頼区間未確認の早期ACCEPTは禁止。実装済み依存でexact boundsを求められるならそれを使い、なければ必要最小限の依存追加をADR化する。独自の近似式で「exact」と名乗らない。reference testは独立した信頼できる実装と照合する。[S5]

### 4.4 保証の範囲

凍結候補・固定look・binomialの前提の下で、各look/candidateの誤受理の確率を `a_A` 以下に制御し、その和を取ることで、一episode内で不足candidateを一つでも受理する確率を `alpha_accept_episode` 以下に抑える。同様に誤棄却側は別の `alpha_reject_episode` を持つ。両側を合わせて1%と呼ばない。

候補間やlook間の独立性はunion boundには不要だが、各candidateのboundsを正当化する標本モデルは必要。未知の分布シフト、生成器の相関、バグまで含めた安全保証とは主張しない。real neural tasksで経験的なstress評価も行う。

### 4.5 Support予算の含意

上の `a_A=0.0004` では、全問正解の下限は `a_A**(1/n)`。95%を超えるための最小整数nは153で、宣言look上の最初の受理可能点は256である。

| n | 全問正解時の下限 | 受理に必要なsuccesses |
|---:|---:|---:|
| 32 | 0.783095 | 不可能 |
| 64 | 0.884926 | 不可能 |
| 128 | 0.940705 | 不可能 |
| 256 | 0.969900 | 254以上 |
| 512 | 0.984835 | 502以上 |

これは式からの参考計算で、APC実測ではない。R3-003/005のCPU contract testで再計算する。

旧R2の「平均support<64」を新primaryの目標として維持しない。少数のwrong candidateを早期棄却した平均だけで、正しい候補の検証コストを隠さない。見合う安全性/コストかをdevelopmentで測り、実行可能性が満たせなければseal前にFAILとして報告する。失敗結果に合わせてtauを下げない。

## 5. Reference adequacy（offlineのみ）

primary referenceはtask/callごとに独立した固定 `n_ref=4096`。必要なcall集合はreference正誤を見る前にmanifest化する。同一call・同一weights・同一task distributionのreferenceはN/levelをまたいで再利用できるが、別独立測定として数えない。

一episodeの評価call集合Mに対し `alpha_ref_episode=0.01` を上下とMへ事前配分したexact boundsを用いる。

```text
REF_ADEQUATE       iff L_ref >= 0.95
REF_INADEQUATE     iff U_ref < 0.95
REF_UNRESOLVED     otherwise
```

各callのconfidence allocationとMを保存する。4096例で未確定なら、結果を見て都合よく追加しない。primaryはUNRESOLVEDを残し、追加測定は別versionのdiagnosticとして区別する。

tau近傍で必ず二分できるとは仮定しない。過去のWilson 1024例監査は歴史的証拠のまま残し、新しいprimary reference判定と混ぜない。

## 6. Runtimeへの接続

薄い `VerifiedDecisionEnvelope` 相当のwrapperを使う。名前は既存APIに合わせてよい。

```text
candidate_verdict: ACCEPT | REJECT | UNCERTAIN
search_status: FOUND_SOLUTION | COMPLETE_WITHIN_SCOPE | BUDGET_EXHAUSTED
controller_action: DIRECT_REUSE | COMPOSE | PLASTIC_SEARCH | null
execution_status: EXECUTED | NEEDS_MORE_EVIDENCE | NO_VERIFIED_SOLUTION
```

controllerの3-class MLPは再学習しない。UNCERTAINなfeatureを0などの確定値にしてcontrollerへ入力しない。controllerがunsafe reuseを選ぼうとしても、ACCEPTされたcandidateがなければ実行を許可しない。

不足の証拠が揃い、既存compositionも十分でなく、必要なdata accessがある場合にのみ現行compact-first pathを使用する。最大予算で未確定なら本タスクではtemporary workspaceを確保せず、明示的にdeferする。将来、探索打切り後のrisk-aware adaptationを導入する場合は別の研究課題とする。

## 7. 新指標と分母

すべて分子/分母、per-operation、per-relation、per-model-seedを保存する。

| Metric | 分子 / 分母 |
|---|---|
| `legacy_known_task_plastic_rate` | known-labelでPLASTIC / known-label episodes |
| `wrong_call_accept_rate` | taskと機能的に非同値なwrong-call受理 / その候補の評価数 |
| `inadequate_call_accept_rate` | REF_INADEQUATE受理 / REF_INADEQUATE評価数 |
| `unsafe_reuse_episode_rate` | 最終実行callがREF_INADEQUATE / referenceで判定可能な最終reuse数 |
| `same_identity_unsafe_accept_rate` | 正family/argsかつREF_INADEQUATEの受理 / その候補の評価数 |
| `avoidable_plastic_rate` | adequate既存解のwitnessがあるのにPLASTIC / witnessのあるepisodes |
| `adequate_solution_nonreuse_rate` | witnessがあるのに正しいreuse/compose出力がない / witnessのあるepisodes |
| `uncertain_candidate_rate` | UNCERTAIN / 全評価candidate数 |
| `reference_unresolved_rate` | REF_UNRESOLVED / reference対象call数 |
| `unconditional_query_EM` | 正解query数 / 全予定query数（未出力は不正解） |
| `selective_query_EM` | 正解query数 / 実際に出力したquery数 |
| `execution_coverage` | 出力query数 / 全予定query数 |

REF_UNRESOLVEDの受理も `accepted_ref_unresolved_count` として別途表示する。未確定をadequate扱いせず、unsafeと断定もしない。空分母はnull/NOT_ESTIMABLE、0%としてPASSしない。

## 8. 機能不足の候補を含むstress suite

低EMのwrong-familyだけでなく、正family/argsで不十分な凍結candidateを含める。旧SHIFT checkpointは必須の履歴再現対照だが、旧sealedの元データは学習に使わない。fresh development samplingで再評価し、最終stressには新しい独立例と、利用可能なら別の不十分candidateも含める。

統計unit testでは既知pのBernoulli列を使い、`p={0.10,0.90,0.92,0.94,0.949,0.95,0.951,0.97,0.98,0.99,0.995,1.0}` でoperating curveを出す。これはprimitiveの代用品としてruntimeへ入れるものではなく、検定器のテストである。

referenceが不足を示す候補に対し、より多くのsupportで不足と確認してreuseを見送ることは正しい。新しいverifierで旧SHIFTが拒否されても、SHIFTの機能修正を省略してよいわけではない。正常運用suiteのavailabilityは別Gateで測る。
