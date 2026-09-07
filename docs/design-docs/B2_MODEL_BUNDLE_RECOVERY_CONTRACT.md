# Design Contract — Immutable ModelBundle & Explicit Recovery Build

**新規の提案契約。以下の型・API・pathは、現repoに既に実装されているという意味ではない。**

## 1. 同一性と依存関係

[S1: §3.3–3.5]が示した問題に対して、seed別directoryではなく、**実行に必要な全componentと意味上の契約**を一式として固定する。

```text
Core/content encoder ─── primitive weights / state ABI ─── decoder
Task encoder / query projection ─── router keys ─── key↔ID mapping
Task-side representation ─── ArgumentScorer / argument schema
router + scorer + formula + lambda + candidate policy ─── final ranking
bank versions + argument schema ─── recipes / recurrence state
runtime policy + verifier contract ─── allowed execution
```

content CoreとTask Encoderが独立か、重みを共有するかは実装で確認する。共有parameterは別ファイルに見えても同じdependency nodeとして扱う。

Coreのtask-blind性と、別Core間のlatent互換性は異なる。`h_content=f(content)`が成立しても、新Coreへのprimitive差替えを無条件に許可しない。

## 2. 推奨manifest

実装はtyped dataclass等でよいが、少なくとも以下の意味を保存する。

```text
schema_version
bundle_id / content_manifest_digest
source_commit / runtime_recipe_version / environment_record
model_id / model_seed / training_run_id
parent_bundle_ids / build_route: RESTORE | PARTIAL_BUILD | CLEAN_BUILD
scope: diagnostic | nominal / requested_capabilities
core, task_encoder, query_projection, decoder, vocabulary
  component_id, file_sha256, canonical_state_hash, schema_hash
primitives[]
  physical_id, operation_name, version, architecture_signature
  argument_schema_hash, state_abi_hash
  core_dependency_hash, decoder_dependency_hash
  weights_hash, source_artifact, training_receipt, provenance_status
router
  weights_hash, task_state_dependency_hash, key_to_primitive_mapping_hash
argument_scorer
  weights_hash, input_dependency_hash, argument_schema_hash
scoring_policy
  family_formula_version, argument_formula_version, lambda
  application_policy_version, calibrated_pair_id
controller / verifier / composition / replay policy signatures
build_recipe_hash / dataset_role_hashes / generator_version
known_defects / exposure_manifest / clean_build_exercised_stages
qualification_refs
```

operation nameは登録・監査用であり、novelty判断のoracleにはしない。較正が必要なrouter/scorer pairはtrainingまたはcompatibility evidenceのIDを保存し、同seedというだけで結合しない。

## 3. Hash設計

raw file hashは元byteの保存を確認する。canonical state hashはtensorの名前・shape・dtype・決定的byte順を記録してweight同一性を確認する。metadataを並べ替えただけの違いと、重みの違いを区別する。

`bundle_id`はcomponent content／dependency／execution policyを含む決定的manifest payloadから求める。自己IDや後付けcertificateをpayloadに循環参照させない。certificateはbundleのcontent digestと評価protocol digestを参照する別artifactにする。

重みhashだけではSELECTのsoftmax→sigmoidのようなformula-only変更を識別できない。従って**execution signature**にformula、schema、対象code version、実行dtype／eval-mode等のpolicyも含める。[S1: §2 R3-007]

## 4. Loaderはfail-closed

提案する概念的APIは次のとおり。exact signatureは実repoへ合わせる。

```text
load_bundle(explicit_manifest, mode, required_capabilities)
  → load + integrity + dependency + qualification checks
  → success または structured error
```

loaderは`get_or_build`、retraining、joint calibration、variant selection、`latest`／mtime／同seed directory探索を行わない。部分state_dict load、missing tensorのrandom補完、別seedのweightへのfallbackを成功扱いにしない。

例外ラベル例：`MISSING_ARTIFACT / CORE_DEPENDENCY_MISMATCH / ARGUMENT_SCHEMA_MISMATCH / UNTRAINED_COMPONENT / UNCERTIFIED_PAIR / INCOMPLETE_BUNDLE / CAPABILITY_NOT_QUALIFIED`。

legacyのunknown来歴は、監査で新たに機能確認できても「以前から検証済み」へ書き換えない。`LEGACY_REQUALIFIED_ON_RECOVERY_FIXTURE`等の新証拠として区別する。現時点の凍結componentの組合せを新fixtureで十分に再認証できた場合は、新たなbinding証拠として保存できるが、元training historyや独立seedの証拠を復元できたことにはならない。安全なdependency対応すら確認できなければnominal利用を拒否する。

## 5. RestoreとBuildを分ける

### Restore

整合候補を新namespaceへcopyし、hash／schemaを検査する。旧shared cacheへのaliasをactive runtimeへつなぎ続けない。元ファイルの変更が新bundleを変えるhard link等を使わない。

旧Coreが良い性能だったと仮定しない。R3-004／005はdevelopment Core自体が未学習だったと報告しているため、復元後の機能確認が不可欠。[S1: §2]

### Build

```text
explicit build plan
    ↓
Core／decoder／schemaの生成または厳密な再利用
    ↓
全16 primitiveの依存付き生成
    ↓
必要なrouter／scorer較正
    ↓
artifact completeness検査
    ↓
機能検証／資格付与
    ↓
新bundle IDへpublish
```

existing canonical recipeがjoint Core/operator learningならそのstageを一体として保存し、その後Coreをfreezeする。残り能力の追加でCoreを変える場合は新parentとして扱い、旧primitiveをそのまま流用しない。

全16行のcoverageに学習レシピまたはvalidated restorationの証拠が必要。default class生成と`requires_grad=False`だけでは学習済みにならない。学習receiptとraw機能測定の両方を要求する。

## 6. Cache／resume／publish

新build cache keyはstage ID、upstream content digest、architecture/schema、recipe version、training/data identityを含む。同seedだけのcache hitは無効。

resumeは固定planとstage hashが一致するときだけ許可。stepやCoreを変えた再開を同一runと呼ばない。予算内retry／resumeの理由も保存する。

stagingは最終bundleと分け、検査前はnominal loaderへ公開しない。publishは新directory・新IDへ行い、既存bundleを上書きしない。失敗stagingは証拠を残し、active bundleは変化させない。stage途中resumeを実装する場合はoptimizer／scheduler／RNG stateも保存し、欠落時は途中再開を拒否する。

旧cacheの開始・終了hashを比較する。保存先rootの変更によって既存runnerの暗黙読込先を変えるglobal設定を書き換えない。

## 7. Qualificationはscope付き

- `structural_complete`：全componentとdependencyが存在する。
- `functional_floor_pass`：復旧用絶対性能floorに合格した。
- `reference_adequacy`：R3の独立reference契約による状態。
- `nominal_capabilities`：そのmodel/version/distributionで通常実行を認証した範囲。
- `known_defects`：SHIFT、hard-ranking等の既知未達。未測定は別status。

certificateにはsample IDs、generator version、query/ref hashes、引数・長さ分布、verifier policy、計測値、status、commitを保存する。reference未確定を削除しない。

`COHERENT_LIMITED`は研究比較の明示状態であり、未学習bankを許容する逃げ道ではない。復旧cohortに許す既知機能例外は事前指定したSHIFTだけ。hard-ranking未達はraw executionと別に記録する。

## 8. R3修正artifactとの整合

parent content digestが変わったら、原則として修正の効果・数値は新親へ移植できない。修正重みをimportできるのはdependencyが一致したときだけ。

必要なら採用済みrecipeで別preparation runを行い、その出力を評価する。修正前後のsource／data／budgetを保存し、以前のsealed failureが直った証拠へ置換しない。

joint較正が必要なpairは一体として記録する。router／scorer双方が変わる較正をkey-only deltaと呼ばない。lambdaやSELECT formula等も条件として保存する。

**適用policyの注意：** レポートはscorerをL4限定に修正したと記すが、hard-negative levelは評価専用情報である。元実験条件の再現と、runtimeが評価labelを入力にして分岐することを区別する。TaskSpec／candidate argument schemaに基づく適用契約を確認し、label漏れなら独立したcontract violationとして報告する。復旧の名目で無断のpolicy変更はしない。

## 9. 実K/C/N/R注入

static matrixでは全componentを不変にする。一方、sequential lifecycleではNの学習・commitが測定対象である。両者のfreeze ruleを同じassertにしない。

```text
immutable initial bundle
    ↓ explicit load
isolated working state for condition Cj
    ↓ K / C: reuse / composition、許可しないweight変更なし
    ↓ N: 既定temporary learning / shadow / promotion / bounded router update
    ↓ release
persistent child bundle + declared replay/controller state
    ↓ fresh process
R: reuse、adaptationなし
```

shared cacheへのwriteは禁止。初期bundleはread-onlyのまま、runtime学習の許可された差分をchildとして残す。K/Cの認証済みrecipe cache更新等が既存policyにある場合はmemory更新とweight更新を分けて記録する。

初期bankにNを含めない。full16からviewを作るなら、Nのprimitiveだけでなくrouter key、recipe、replay、prototypeの残存も検査する。Coreの既往training exposureは別に記録し、この回帰をunseen-family研究と呼ばない。

query／referenceはN trainingにも渡さず、既存episode adaptation supportと独立shadowを使う。成功NだけのRと予定全streamの結果を分ける。evaluatorのK/C/N/Rラベルでcontrollerの誤actionを修正しない。

## 10. Cache無効化とrollback

primitive／Core／schema変更はrecipe intermediate、reference score、verification結果、retrieval prototype等の依存cacheを無効化する。同一physical IDでもversionが違えば別call signatureである。

SHIFTのrollback先は同じCoreに対応したtrained親。却下candidateのEM、installed親のEM、最終childのEMを別欄にする。弱い親へ安全にrollbackできたことを、SHIFT availability成功と呼ばない。

## 11. 最低限のテスト

1. 別Core＋同shape primitiveの結合を拒否する。
2. 空cacheで全16個に未学習項目が残ればpublishを拒否する。
3. loader／frozen evaluationからtrain／calibrate／select-variantが呼ばれたら失敗する。
4. formula-only変更がexecution signatureを変える。
5. C0とC5が両方0点でもabsolute floorが失敗する。
6. interrupted buildを正常bundleとして読めない。
7. parent変化で修正artifact・reference cacheを拒否する。
8. Nで許可された更新は通し、K/Cでの不正weight更新は拒否する。
9. Rのfresh-loadにtemporary workspaceや隠れoptimizerが不要である。
10. sealedデータがbuild／prep／selectionへ入れば拒否する。

CPU fixtureの契約試験と学習済みGPU artifactの機能検査を別milestoneで行う。
