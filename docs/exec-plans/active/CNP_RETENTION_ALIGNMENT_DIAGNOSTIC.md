# R-CNP-001D 保持勾配整合診断

日付: 2026-09-22 / ADR-0207 / `AUTHORIZED_ZERO_UPDATE_DIAGNOSTIC`

## 1. 目的と範囲

R-CNP-001Rで観測した「lambda=1はoldを保持するがnew品質を悪化させる」現象について、
係数や目的関数を変更する前に、new BCE、replay BCE、replay logit MSEがadapterへ要求する
初期更新方向を測る。これはR-CNP-002の確認、候補選択、係数探索ではない。

ユーザーの「その順序で進める」指示により、この**名指しの更新数0診断**を認可する。
本書にない学習、候補採用、bundle promotion、sealed access、R-CNP-002は認可しない。

## 2. 固定条件

| 項目 | 固定値 |
|---|---|
| 名称 | `R-CNP-001D` / `cnp_repair_alignment_v1` |
| 親 | 保存済みCNP-003 MLP seed 610200–610204 |
| new panel | fresh root 620210、`alignment` role、8 query×3 threshold×64 sets=1,536 |
| replay | 保存済みdigest選択R-CNP-001 corrected replay、1,536 sets |
| candidate | first-hidden rank-8 adapter、base 8,449 weights凍結、更新eligible 1,024 |
| optimizer update / candidate write / selection / promotion / sealed | 0 / 0 / 0 / `NOT_AUTHORIZED` / 0 |
| gradient batch | 32 sets、固定順、dropoutなし、float32 |
| virtual probe | initial AdamWの1更新をメモリ内だけで再現し、必ずadapterを復元。optimizer stateとcheckpointを作らない |
| resource上限 | wall 1,800秒、CUDA 12GiB、process RAM 32GiB |

new/replayは生成前にinput/queryの非重複、件数、panel全体の正負supportを監査する。
shadow panelを使う将来の診断では各cellの正負supportも要求する。学習用replay bufferの
偶発的な細分stratumへshadow用のclass-support条件を適用しない。
new root、config、source hash、parent hash、panel digestを保存する。旧R panelを再評価しない。

## 3. 測定

各親で、candidate=parentのゼロadapter状態に以下を全panel平均で測る。

```text
g_new    = ∇ L_BCE(new)
g_replay = ∇ L_BCE(replay)
g_task   = 0.5 * (g_new + g_replay)
g_keep   = ∇ L_MSE(candidate_logits, stop_gradient(parent_logits))
```

ゼロadapterでは`g_keep=0`が必然なので、これをMSEとtaskが整合する証拠とは扱わない。
そこでoptimizerを作らず、登録済みAdamW初回方向
`delta = -0.001 * g_task / (abs(g_task)+1e-8)` をadapterへ一時適用し、同じfull panelで
`g_task'`、`g_keep'`を測る。計測後は全adapter tensorをbitwise復元し、base hashも前後一致を要求する。

主指標は`cos(g_task', g_keep')`。補助としてnew/replay間cosine、各loss、各gradient norm、
virtual MSEを保存する。virtual probeはモデル学習、candidate、または実際のfirst optimizer stepではない。

## 4. 判定と停止

事前登録した機序判定は、全5親で`cos(g_task', g_keep') <= -0.10`なら
`ALIGNMENT_CONFLICT_SUPPORTED`。それ以外（ゼロnorm/NaNを含む）は
`ALIGNMENT_CONFLICT_NOT_ESTABLISHED_STOP`。この判定は性能gateでもレシピ選択でもない。

いずれの場合もR-CNP-001S/RのFAIL、v1 G3 STOP、R-CNP-002未認可を維持する。
`NOT_ESTABLISHED`なら第3段階の目的関数変更は停止し、係数やmargin方式を導入しない。
`SUPPORTED`でもmargin方式の正しさは示さない。第3段階には別途、単一変更・開発split・
選択規則・budget・all-five gateを登録する。

分離監査、初期parity、base-hash rollback、resource上限のいずれかがFAILなら
`INVALID_RUN_STOP`または`RESOURCE_STOP`として保存し、勾配の科学的解釈をしない。

## 5. 成果物と検証

出力先は`runs/cnp_repair/r001d/rcnp001d_alignment1/`。`run_manifest.json`、
`data_manifest.json`、`report.json`に全5親の勾配・hash・accounting・resourceを保存する。
Python 3.12、単一RTX 5060 Tiで実行する。実行前に修復関連CPU/WSL test、ruff、mypyを通し、
実行後にreport算術、input hash、git diffを確認する。
