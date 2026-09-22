# R-CNP-001M: 親正解decision保持診断

日付: 2026-09-22 / ADR-0209 / `AUTHORIZED_SINGLE_OBJECTIVE_DIAGNOSTIC`

## 1. 目的と範囲

R-CNP-001Dの全5親で、MSE保持項は仮想初回task更新と強く逆向きだった。MSEの代わりに、
親が**正解した**旧replay要素だけを正解ラベル側のdecision境界に保つ制約を、単一変更として
検査する。親が誤った要素を親logitへ戻すことは避け、replay BCEだけで修正可能にする。

ユーザーの「その順序で進める」指示とADR-0208の機序支持に基づくこの段階は、固定親の
開発診断に限る。R-CNP-002、係数/余白探索、候補選択、bundle promotion、sealed access、
transfer/composition評価は認可しない。

## 2. 固定条件

| 項目 | 固定値 |
|---|---|
| 名称 | `R-CNP-001M` / `cnp_repair_decision_v1` |
| 親 | 保存済みCNP-003 MLP seed 610200–610204 |
| new panel | fresh root 620310、8 query×3 threshold×64 sets=1,536 |
| shadow | new/oldとも別role、120セル×128 sets |
| replay | immutable digest-selected R-CNP-001 corrected replay、1,536 sets |
| schedule | 新旧ともroot 620320の`DISPERSED_BALANCED`、256 batch×各16 sets |
| candidate | first-hidden rank-8 adapter、base 8,449 weights凍結、更新eligible 1,024 |
| optimizer | AdamW lr=0.001、wd=0、betas=(0.9,0.999)、eps=1e-8、clip=1 |
| arms | `TASK_ONLY` と `PARENT_CORRECT_DECISION`。候補は後者 |
| decision weight / margin | 1 / 0.0（探索しない） |
| limits | 7,200秒、12GiB CUDA、32GiB process RAM |
| selection / promotion / sealed | 0 / `NOT_AUTHORIZED` / 0 |

各batchの損失は、従来の集合等重みBCEを残して次で固定する。

```text
L_task = 0.5 * SetMean(BCE(new_logits, new_labels))
       + 0.5 * SetMean(BCE(replay_logits, replay_labels))
C = valid AND ((parent_replay_logits >= 0) == replay_labels)
L_decision = SetMean_C(relu(-signed(replay_labels, replay_logits)) ** 2)
L = L_task + weight * L_decision
```

`SetMean_C`はCが非空の集合だけの集合平均であり、Cが全空のbatchでは微分可能なゼロを返す。
親logitはCの固定maskを作るためだけに用い、親が誤った要素はCから除外する。margin=0なので
正しいpredictionを必要以上にlogit値へ引き戻さない。全対象がreplay targetに依存するため、
親の誤答を保持する教師にはならない。

## 3. 事前停止規則と判定

モデル構築前にrole/query/input/recordの分離、全shadowセルの128 set・正負support、
new/replay全体の正負supportを監査する。各候補は初期adapter parity、frozen base hash、
fresh-process checkpoint parity、parameter count=1,024、resource上限を満たさなければ
`INVALID_RUN_STOP`または`RESOURCE_STOP`で停止する。

候補armは各5親で、new/oldの全120セルにBA≥0.95かつF1≥0.90、かつ全旧セルで
candidate F1−parent F1≥−0.01、全invariant PASSを満たす必要がある。5/5なら
`M_DIAGNOSTIC_VIABILITY_PASS`、一つでも未達なら`M_DIAGNOSTIC_FAIL_STOP`である。
`TASK_ONLY`との差は説明用に保存するが、結果を見てweightやmarginを変更しない。

## 4. 成果物と検証

出力は新規namespace `runs/cnp_repair/r001m/rcnp001m_decision1/`。run/data/schedule manifest、
batch schedule、candidate checkpoint、fixed step 0/16/64/256 trace、親・候補セル/集合証跡、
bootstrap、fresh-process parity、parameter/resource accountingを保存する。実装はCPU fixture、
Ruff、mypyで検証後に、この一回だけ実行する。

この診断がPASSしても、修正版親、新しい確認split、G1/G2/G3、因果対照、転移/合成を必要とする
R-CNP-002は未認可のままである。
