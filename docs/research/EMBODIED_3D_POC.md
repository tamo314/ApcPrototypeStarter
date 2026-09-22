# Embodied 3D APC PoC

## Scope decision E3D-001 — independent continuous-control domain

Created from `main` at `319f3af8b16accf7347574247e4eae6ab30ad402` on
2026-09-22, on branch `feat/embodied-3d-primitive-poc`.
The user's request authorizes this environment, continuous-domain adapter and
bounded implementation verification. This is **not** a restart of archived
Phase B/C gates or a reversal of the latest CNP/Phase D conclusions. Existing
models, token operation IDs, research plans, sealed data and historical runs are
unchanged. This decision uses an independent scope identifier, not a new claim
against the global research ADR sequence.

The deliverable is an executable environment and APC integration harness. Its
initial control baseline intentionally makes body-state/goal binding and the
adaptation lifecycle inspectable before adding perception or complex locomotion.
A successful smoke test is **not evidence of autonomous general skill discovery**.

## 起動

リポジトリのルートで実行する。推奨環境は既存プロジェクトと同じ Python 3.12。
GPU、外部AIサービス、ゲームエンジン、ネットワーク接続を実行時には必要としない。
初回の依存パッケージ取得には通常のパッケージインデックスへの接続が必要。

```bash
git fetch origin feat/embodied-3d-primitive-poc
git switch --track origin/feat/embodied-3d-primitive-poc
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,embodied]"

# 既存APC PrimitiveBankを使う標準経路
python scripts/run_embodied_3d.py build \
  --config configs/embodied/reference.json \
  --output runs/embodied/build-001

# 学習済みバンクだけを読み、予約した別seedで評価。学習の暗黙実行はない。
python scripts/run_embodied_3d.py evaluate \
  --bundle runs/embodied/build-001/bank.json \
  --output runs/embodied/eval-001
```

`build-001/replay.html` と `eval-001/replay.html` をブラウザで開く。
再生、一時停止、シーク、速度変更、ドラッグによる視点回転、ホイール拡大縮小に対応。
HTMLに軌跡を内包しており、CDNやWebサーバーは不要。
リプレイは実際に記録した座標・姿勢・選択プリミティブを表示する。

既存の出力ディレクトリがある場合は失敗する。再実行では `build-002` 等の新しい名前を使う。
失敗したゲートを無視して後続を実行しない。`report.json` / `error.json` を確認する。

物理環境・学習制御の切り分け用に `--backend numpy` を明示指定できる。
これは同じ重みを使う参照実装であり、既存APCバンクを経由しない。
標準の `apc` 経路が壊れた場合に自動で参照実装へ切り替えることはない。

## 身体と時間

身体は中性浮力を仮定した球形推進体。地上の点を瞬間移動する環境ではない。
位置3成分、速度3成分、姿勢Quaternion、角速度3成分を持ち、身体座標での
推力3成分・トルク3成分で動く。並進慣性、球の回転慣性、線形抵抗、推力・トルクの
ノルム上限、身体半径、障害物・境界との衝突をモデル化している。
上下方向も実際の状態変数・制御対象であり、2D環境を3D表示したものではない。
初期タスクの姿勢はyaw方向にランダム化する。回転ダイナミクス自体は3軸を扱うが、
最初の学習ポリシーはトルクを0とし、姿勢制御の獲得を主張しない。

連続時間の力学を、既定では物理刻み0.01秒、制御周期0.05秒で数値積分する。
無限小刻みの厳密な連続時間計算ではない。制御周期は物理刻みの整数倍とする。
環境の40秒上限と、単一プリミティブの15秒上限は別に管理する。
成功には目標から0.16m以内、速度0.2m/s以下を0.15秒継続する必要がある。
衝突は失敗終了、時間切れはtruncationであり、どちらも成功扱いにしない。

衝突検出は、球半径で拡大した直方体に対するswept segment判定。
薄い壁のすり抜けを防ぐ一方、直方体の角では保守的に衝突と判定する。
衝突時刻まで進み、法線方向速度を除去してエピソードを終了する。
これは検証可能な軽量参照物理であり、MuJoCo等の検証済み剛体エンジンではない。
接触摩擦、関節、歩行、空力、流体、実機安全性は扱っていない。

## タスクと「何が未学習か」

| Family | 内容 | 明示している前提 |
|---|---|---|
| `planar` | 同じ高度の別地点へ移動して停止 | 水平移動の初期スキルを学習 |
| `spatial` | 高度を含む別地点へ移動して停止 | 初期水平スキルが失敗する能力差を設計 |
| `route` | 異なる高度の複数地点を順番に訪問 | 経由地点はTaskSpecとして与える |
| `detour` | 立体障害物の上を経由して移動 | 障害物回避経路の中間目標も与える |

`detour` は自律的な経路発見の試験ではない。低位制御プリミティブを時間方向に
再利用し、別の地点列に合成できるかを試す。位置・速度・姿勢・障害物形状は
正確なシミュレーター状態として観測可能で、画像認識や部分観測の問題は含まない。
これは、身体ダイナミクス、プリミティブ再利用、経路推論、知覚を最初から混ぜないための
初期条件である。後者の研究は別のタスク・教師・評価契約を必要とする。

## APCとの接続

```text
物理状態 → task-blind Observation / content()
                              ↓
TaskSpec → recipe → EmbodiedPrimitiveCall(operation, arguments={target})
                              ↓
                  選択されたプリミティブのみ実行
                              ↓
                 身体座標の推力 → 固定actuator decoder
                              ↓
                        物理積分 → 次の観測
```

目標を共有Content Coreに埋め込まず、選択されたプリミティブ内で初めて引数を結合する。
環境・actuator decoderは目標に向かう制御を代行しない。
各物理観測の更新ごとに閉ループで実行し、軌跡を固定長マクロとして再生しない。

既存 `apc.environments.primitive_call.PrimitiveCall` はトークン操作の登録・検証・IDに
結び付いているため、そこへ運動操作を追加して互換性を壊すことはしない。
`EmbodiedPrimitiveCall` が連続ドメイン用の引数契約を持つ。
標準バックエンドの `EmbodiedAPCPrimitive` は **実際の `PrimitiveBase` を継承**し、
**実際の `PrimitiveBank.add_primitive/get`** を通して登録・選択実行される。
`STABLE`、freeze、usage、forward-call数、パラメータ数には既存の機構を使う。

旧トークン用Core・プリミティブの学習済み重みを、そのまま運動制御に転用する実装ではない。
身体観測のContent Coreは最初はパラメータを持たない状態表現、ポリシーは新しい運動重み。
CNP/jevや既存の学習済みrouterを暗黙に挿入していない。

`MotionCompositionLibrary` はバンクと独立し、操作の順序だけを保存する。
単一操作のレシピは任意数の経由地点へ同じ重みを再利用する。複数操作のレシピは
地点ごとに別の操作を選択できる。地点座標を永久的なプリミティブとして登録しない。
選択は「最後に検証済みのレシピから順に試す」という明示的な決定的手順であり、
学習済み選択器や新規性推定器の性能ではない。失敗した試行・resetコストもログに残す。

## 学習・検証・追加・再利用

最初の学習器は、**明示的なPD制御教師の実シミュレーション軌跡からの模倣学習**。
身体座標の目標相対位置3、速度3、バイアス1から推力3を出す、21係数の線形モデルを
ridge最小二乗で学習する。係数を教師から直接コピーしたり、タスクの答えを保存したりはしない。
ただし、線形モデルという構造、教師、初期の水平→3Dという能力差は設計者が与えている。
この基準条件の成功を、汎用的な技能発見や非線形制御の学習と呼ばない。

`build` の固定手順:

1. 訓練seedの水平移動で `move_planar` を学習。教師・学習器の水平shadow検証後に登録。
2. 既存バンクを新しい高度変更タスクに試す。失敗がなければ追加学習しない。
3. 失敗後に訓練seedの3D軌跡から、独立した一時候補 `move_3d` を学習。
4. 固定したvalidation seedの4種類のタスクで、教師制御と候補を検証。
   全件成功が必要。失敗時は候補を別ファイルに保存し、バンクへ追加しない。
5. 合格後のみ別プリミティブとして登録しfreeze。旧スキルの重みの不変性を確認。
   呼び出し元が一時候補を解放する。新しいレシピをバンクとは別に登録。
6. 元の失敗タスク、別の地点列、障害物を越える地点列で再利用。
   保存・再読込後にも再学習なしで同一軌跡を確認する。

学習データに十分な独立方向の励起がない場合は、回帰のfeature rankを検査して停止する。
訓練損失だけが小さい未同定の制御器を採用しない。既存バンクは候補学習に渡さず、
shadow検証も別の一時バンクで行う。候補不採用時の旧バンク不変性もテストする。

これは初期2段階のカリキュラムを持つ実装検証ハーネスである。
未知の失敗原因から、新しい構造・教師・行動種類を無制限に自律発見する実装ではない。
公開された `try_library` / `consolidate_after_failure` と環境APIを使って後続の
学習器・課題ストリームへ拡張できるが、その能力は今回検証していない。

## データ分割と反証対照

既定のtrainは8 seed、validationは4 seed、予約testは8 seedで、集合の重なりを拒否する。
`build` はtestタスクを生成・評価しない。露出したタスクの内容hashをバンドルに記録する。
`evaluate` は予約seedで4種類×8件を評価し、露出hashとの重なりも拒否する。
このtestは小規模な開発用予約分割であり、既存研究のsealedデータではない。

評価にはCorrect、None、Wrong argument、Wrong primitiveを含む。
Noneではバンクを一度も呼び出さない。Wrong argumentでも環境側の正解目標は変わらない。
Wrong primitiveは別のインストール済み操作を実行する。
水平タスクでは水平プリミティブも正しい動作をするため、その対照はeffectfulではない。
タスク種類ごとに集計し、この条件を因果分離の成功として過大評価しない。

評価コマンドには学習・教師・consolidation呼出しがない。欠損バンドルはエラーになる。
保存形式は明示的なJSONで、schema・形状・有限値・checksum・操作参照を検証する。
旧 `PrimitiveBank.save/load` のトークン型レジストリには新しい型を追加していないので、
**このドメインでは `apc.embodied.bundle.save_bundle/load_bundle` を使う**。

## 独自タスクへの接続例

```python
from pathlib import Path
from apc.embodied.bundle import load_bundle
from apc.embodied.control import try_library
from apc.embodied.world import NavigationTask

config, splits, bank, recipes, exposure = load_bundle(
    Path("runs/embodied/build-001/bank.json"), backend="apc"
)
task = NavigationTask(
    start=(-2.0, 0.0, 1.0),
    goals=((0.0, 1.0, 3.0), (2.0, -1.0, 4.0)),
    name="custom-route",
)
attempts = try_library(config, task, bank, recipes, record_trace=True)
print(attempts[-1].success, attempts[-1].reason)
```

直接制御の場合は `EmbodiedEnv.reset(task)` と `step(action)` を使う。
`action` は有限な6成分 `[fx, fy, fz, tx, ty, tz]`。各3成分のノルムを最大1へ制限してから、
設定の最大推力・最大トルクを掛ける。身体座標であり、世界座標ではない。
`step` は `(observation, reward, terminated, truncated, info)` を返す。
Gymnasium依存・Gym登録は追加していない。終端後のstepはresetまで拒否する。

## 出力と検証

`bank.json`: 安定プリミティブ、レシピ、設定、分割、露出hash。
`report.json`: 学習方法・データ数、全shadow/評価結果、失敗試行、容量、再利用、
実行環境・コードhash・経過時間・プロセスpeak RSS。
`episodes.json` / `replay.html`: 実状態の軌跡と表示。
不採用候補とエラーは別ファイルに残す。元の `runs/` の測定値は変更しない。

```bash
python -m pytest -q tests/test_embodied_world.py tests/test_embodied_control.py \
  tests/test_embodied_apc_bridge.py
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

実装時の参照バックエンド検証: **47 passed / 2 skipped**。
スキップ2件は実際のAPCクラスとの統合テストであり、モックで合格扱いにしない。
この作業コンテナではGitHub APIによる読書きは可能だったが、Git cloneの通信ができず、
ローカルには変更対象のソースだけを配置して検証した。既存ソース一式での全体回帰と
ネイティブ統合は未実行。`ruff` / `mypy` は未導入で、取得もできず未実行。
Python 3.13.5 / NumPy 2.3.5のCPU参照環境での検証であり、推奨Python 3.12と
指定PyTorch版の完全な組合せは未検証。
ブラウザ再生はChromiumで、4エピソードの選択・シーク・再生、JavaScriptエラー0を確認。

参照実行では初期水平スキルは高度変更でoption timeout、追加後の固定shadowは成功し、
再読込時の軌跡も一致した。予約testは4種類各8件でCorrectが各8成功、Noneと
Wrong argumentが各0成功。Wrong primitiveは水平8成功、他の3種類は0成功。
これは上記の強い構造・教師・経由地点を与えた小規模な接続確認であり、
従来のプリミティブ汎用性問題が解決したという結論ではない。
