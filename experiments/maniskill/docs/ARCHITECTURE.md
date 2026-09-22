# 実装とデータの設計

## 実装済みと未実装を分ける

現在実装されているのは `RunConfig → ManiSkill → rollout → NPZ/JSONL → summary`。
単一環境、state観測、Box行動、random/zero方策、CPU/GPUの明示選択、任意動画に限定する。
GPU並列auto-reset、学習器、custom到達タスク、スキル銀行、selector、蒸留は未実装。
本体コードは `src/apc_maniskill/`。旧 `src/apc/` をimportしない独立Pythonパッケージ。

## 次に実装する経路

```text
観測 + 最終目標
    ↓
selector（初期は明示的/手動でもよい）
    ↓ skill_id + 局所目標 + 実行時間上限
skill policy(observation, local_goal, memory) → continuous action
    ↓
ManiSkillの物理step → 次の観測 → 同じskillへ戻る
    ↓ 成功 / 失敗 / 時間上限 / 切替
selectorへ戻る

必要に応じ temporary learner → separate compact candidate → skill bank
```

skillは一回の内部表現変換でなく、時間を持つ閉ループ方策。
skill選択周期と環境制御周期を分け、途中の動作修正を可能にする。
開始可能範囲・終了条件・最大時間をスキルの契約に含める。
旧PrimitiveBaseの容量/凍結管理は参考にするが、その学習済み重みを無理に転用しない。

## 最初のスキルインターフェース案（未実装）

`reset(batch_size)`, `act(observation, local_goal)`, `should_terminate(observation)` を基本にする。
記録にはskill_id、引数、checkpoint、robot、control_mode、観測schemaを含める。
共有encoderを更新するなら旧スキルへの影響を測る。小さい初版では固定encoderまたは
スキル独立MLPを選び、潜在空間の変更が見えなくなる構成を避ける。

temporaryとcandidateは別のstate_dict/保存先にする。置換前は銀行を保持し、
圧縮したcandidateを実環境で試して比較する。容量解放の主張には参照/optimizerも含む。
追加スキルの発火は初期には手動でよく、自動判定の成果とは区別する。

## データschema v1

`manifest.json` にschema_version、設定、num_envs=1、obs_mode=state、環境周波数、
action shape/bounds、開始/終了日時、完了episode数、実行status、依存/git/ホスト情報を保存。
statusはrunning/completed/failed/interrupted。completedは試行完了でありタスク成功ではない。
CPU状態観測時は `render_backend=none` を指定して、不要なVulkan依存を避ける。

各NPZは次を含む。単一環境でもManiSkillのbatch軸を保存し、勝手にsqueezeしない。

| key | 先頭軸/型 | 意味 |
|---|---|---|
| observations | T+1、数値配列 | 初期観測と各行動後の観測 |
| actions | T、数値配列 | 環境に実際に渡した行動 |
| rewards | T、float32 | step報酬 |
| terminated | T、bool | 環境の終了 |
| truncated | T、bool | 環境wrapperの時間打ち切り |
| success | T、int8 | -1: 情報なし、0: 未成功、1: 成功 |

runner独自のmax_steps打ち切りはepisode行のrunner_truncatedに記録する。
最終NPZのtruncatedを勝手に書き換えない。強化学習に取り込む際には、両方の時間制限を
扱い、真のterminalとは区別してbootstrap方針を決める。
summaryの成功率は成功情報を得たepisodeだけを分母にし、その数と欠測数も表示する。

これは簡単な状態/行動データ形式であり、ManiSkill公式のHDF5実演形式ではない。
物理エンジンの全状態を保存していないため、NPZだけで厳密な物理リプレイはできない。
必要になった時点でRecordEpisodeのtrajectory保存やstate_dictの保存を足す。
seedを記録しても異なるGPU/依存/PhysX版でbitwise再現を保証しない。
git dirty状態は記録するが、未コミットソース全体のsnapshotまでは初版で保存しない。
意味のある比較はコミット済みソースを使い、例外的なdirty runは差分を別途保管する。

## 終了・失敗・出力保護

環境のterminated/truncatedでそのepisodeを終了し、明示的reset後にのみ次へ進む。
最終episode後にcloseし、任意の動画をflushしてからcompletedにする。
捕捉できる例外はerror.txtとmanifestに残し非ゼロ終了する。失敗runを黙って再試行しない。
途中のstepログをflushする。SIGKILL/segfaultではfinallyは保証されず、runningのまま残り得る。
保存先の既存ディレクトリは拒否し、新しいrunで再試行する。

## 依存と拡張

ManiSkill 3.0.1を起点にし、PyTorch wheelは研究用GPUに合わせる。
rootのAPC依存は変更しない。バージョン解決/実測後に互換性メモを足す。
最初からベクトル環境のpartial resetを抽象化しない。速度が必要になった時点で
ManiSkillVectorEnv/上流PPOの方式を調べ、終了・最終観測・時間制限を扱う。
その時点で「並列収集」全体の統合テストを原則1本だけ足す。

上流APIの出典は [READMEの資料](../README.md) を参照。
