# B2 Reproducibility & Semantic-Relation Split Contract

## 1. 根拠と適用範囲

ADR-0080は `src/apc/evaluation/recurrence_benchmark.py` と `consolidation_benchmark.py` の `generate_benchmark_examples` にある `hash(operation) % 10000` を、別プロセスで入力が変わる原因として報告した。ADR-0081は、seedだけでなくrelationで分割することを求めた。[S1]

以下はそれを実装可能にする新規契約である。実関数名・引数は最初のタスクで確認する。Pythonの文字列hashのプロセス依存性については[S2]、digest APIについては[S3]を参照。[出典ノート](../research/B2_POST_D2_SOURCE_NOTES.md)

## 2. Generator v2

### 2.1 決定的なseed導出

組み込み `hash()`、オブジェクトaddress、順序不定のset走査、実行時刻から研究データのseedを作らない。`PYTHONHASHSEED=0` は旧runの診断補助手段として残せるが、新generatorの正しさをそれに依存させない。

一つの共通utilityに導出をまとめる。推奨契約は次のとおり（実装済みという意味ではない）。

```text
canonical_payload = {
  generator_version,
  master_seed,
  stream_namespace,
  task_semantics_version,
  task_key,
  sample_index
}
seed = first_63_bits(SHA256(canonical_UTF8_JSON(canonical_payload)))
```

JSONはkey順、数値型、Unicode、区切り、集合表現を固定する。generatorで使う集合と、SELECT実行時のargument semanticsを混同しない。順序に意味があるargumentを勝手にsortしない。短い `% 10000` に戻さない。digest衝突が数学的に不可能とは主張せず、manifest対象の重複/衝突を検出する。

`task_key`はgenerator内部のみ。model入力へseedやmanifestを渡さない。DataLoader worker数、cellの走査順、再開位置によってexample内容が変わらないよう、sample indexごとに再現可能にする。

### 2.2 Namespaceと比較の単位

最低限、`train / validation / support / query / reference / competitors / probe` を分ける。model初期化seedとexample seedを分ける。relation sealはseedに埋め込むのでなく、別のアクセス契約で管理する。

同じtask/episodeのN・hard-negative level・policy比較では、support/queryの入力列を共通にする。N/levelは原則としてcompetitor生成のnamespaceにだけ含める。bank sizeを変えたらsupportも変わる比較をprimaryにしない。分布自体を変更する実験は明示的に別task distribution IDを持つ。

固定32/64/128/新sequentialの比較は、同じ事前生成support streamのprefixを使う。各policyで独立に都合のよいdrawを引かない。停止後にprefixを再生成し直さない。

### 2.3 旧version

- 旧成果物のconfig、JSON、checkpointは変更しない。
- version欠落の過去runを自動的にv2とみなさない。
- 保存済み入力があれば `EXACT_REPLAY_AVAILABLE`、hash/randomization記録だけでは復元できなければ `EXACT_REPLAY_UNAVAILABLE`。
- 新生成例で古いcheckpointを測る結果は `V2_RECONSTRUCTION`。元runの完全再現ではない。
- legacy実行互換が必要なら明示modeに隔離する。新Gateがlegacy pathを呼んだらpreflightでFAIL。

## 3. Manifestと独立性

```text
code_commit / dependency_lock_hash / generator_version / generator_hash
model_init_seed / model_training_seed / data_seed / competitor_seed
checkpoint_sha256 / parent_checkpoint_sha256
bank_schema_version / bank_member_version_hashes
relation_split_version / exposure_manifest_hash
task_distribution_id / argument_schema_version
support_dataset_hash / query_dataset_hash / reference_dataset_hash
candidate_manifest_hash / support_look_schedule / metrics_schema_version
runtime_device / deterministic_algorithms / python_hash_seed_environment
```

`seed=24`と`model_seed=4`のような写像を隠さない。同じcheckpointを異なるdata seedで20回測っても20独立モデルとは数えない。モデル間集計は原則として5個以上の固有checkpointを単位とし、data replicateはその内側で平均する。

GPU数値出力の再現性と、generatorのexact-byte再現性を分離する。入力hashは跨processで完全一致を要求するが、異なるGPU/ライブラリversion間の全tensor bitwise一致を根拠なく保証しない。

## 4. 二つの評価suite

### A. Targeted repair regression

既に診断したCOUNT↔BIND、SELECT/BIND arguments、SHIFT実行性能を、新規development例と独立validation例で再現・修正する。履歴にある関係を「未見」と呼ばない。

### B. Repair-relation transfer

修正学習で露出させないrelation groupを評価する。既存primitiveのbase trainingが全familyを含んでいた場合、主張は **repair-time relation holdout** に限定する。新operation-family generalizationや完全未学習semantic relationの主張とは異なる。

## 5. 分割手順

1. 既存bankとhard-negative builderの実体を列挙する。実primitive ID、実装version、型、argument schema、competitor provenanceを取得する。
2. relation unitを `unordered_family_pair + structural_relation_subtype` を基本に定義する。向きは評価層として残すが、COUNT→BINDとBIND→COUNTを別splitにしてリークさせない。
3. 類似pair、対称・逆関係、同一機能の別IDが同じgroupに入るようalias mapを作る。
4. 勝敗やsealed成績を見る前に、`DEV / VALIDATION / SEALED` のrelation membershipをmanifest化する。
5. 旧失敗relationはtargeted regressionのDEVに置く。新sealedとして再利用しない。
6. 既存のreal learned primitivesで各partitionを構成する。semantic competitor数が不足する場合は `PROTOCOL_INSUFFICIENT_RELATIONS` でSTOP。random vectorをsemantic relationと呼び替えない。
7. 原則各partitionに2個以上の非alias relation groupを要求する。実bankで満たせない場合は統計や主張範囲を勝手に緩めず、その不足を報告する。この数は新規の最低設計条件である。

## 6. 暗黙のtraining exposure

relation holdoutはnegative miningだけの問題ではない。cross-entropyの全class denominator、replay、distillation、pair construction、argument negativesによってもheld-out pairを学習する可能性がある。

新repair trainingの露出台帳に、sample・positive・negative pair・loss typeを記録する。厳密なrepair-time pair holdoutなら、held-out pairを新lossの競合項から除外する。除外せず全class CEを使う場合は `MINING_HOLDOUT_ONLY` と記録し、relation holdoutのprimary PASSに使わない。

親checkpointの過去training exposureが不明なら `historical_exposure=UNKNOWN`。過去の学習を消したことにしない。全familyを事前学習したbankを使うこと自体は許容するが、修正時の未見relationという限定を残す。

## 7. Hard-negative妥当性

L0/L1/L2とL3/L4を同一視しない。L3はreal learned primitiveとそのprovenanceが必要。queryに合わせてwrong keyを恣意的に上乗せしたものは `SYNTHETIC_SCORE_ATTACK` として別表にする。実bank keyへのsemantic retrievalと混ぜない。

L4は同一physical primitiveの異なるargument候補であり、resident bankを増やさない。同一機能になるargumentはoracle generatorによる評価用機能同値判定で区別し、単にidentityが異なるだけの正解をwrong acceptanceに数えない。

COUNT↔BINDでkey norm・normalization・ID対応・固定人工score offsetを、修正前に点検する。識別に必要な情報がscore入力に存在しない設計なら、学習量だけ増やさず `BENCHMARK_CONTRACT_FAILURE` として止める。

## 8. Seedとpartitionの封印

partition membershipとsampling recipeはR3-002で固定する。sealedのmodel出力はR3-012まで測らない。R3-011では修正済みartifact hashと評価設定を最終lockする。seal後の変更は旧sealを無効化し、新versionへ移る。

ハッシュは「変更検出」であって、担当者の目視閲覧を物理的に防ぐものではない。どの成果物をいつ閲覧したかも台帳化する。後続repairのため失敗sealedを使い回さない。

## 9. 必須テスト

- `PYTHONHASHSEED`未設定・0・1・123の4 subprocessで、同一sample manifestの入力hashが一致。
- cellの順/逆順、単体再開、worker数変更で同じsample IDが同じ例。
- recurrence/consolidationのwrapperが同一utilityを通り、同じ引数なら同じ例。
- namespace変更で別streamになる。保存済み配列の意図しない共有を検出。
- 逆方向relation、alias pairがsplitをまたがない。
- loss exposureチェックがheld-out pairへの新trainingを拒否。
- 未記録checkpoint aliasや`seed % 5`を独立seedとして集計しない。
- history exact replay unavailableを架空のPASSに変換しない。
