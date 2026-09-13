# Design Contract — Phase D Five-Model Cohort Construction & Provenance Plan (Task D-001)

**Task D-003によりcohort構築の実行が認可された
（[ADR-0170](../DECISIONS_PHASE_D.md#adr-0170-d-003-phase-d-charter-authorization-decision-scoped-approval)、
`training_execution: AUTHORIZED`、本契約のseed 30-34・手順に厳密限定）。ただしD-003自体はcohortを
生成していない。以下の契約内容・数値は本認可によって変更されない。**

**新規の提案契約。本契約はcohortの構築・検証*手順*を確定するものであり、本タスク内でcohortを
生成しない。** `docs/design-docs/B2_MODEL_BUNDLE_RECOVERY_CONTRACT.md`（以下「REC契約」）が
定義した `ModelBundleManifest` / `load_bundle` / hash方式を**そのまま再利用**し、独自の
manifest形式を新設しない。REC契約・その実装（`src/apc/utils/model_bundle.py`）は本契約の
親契約であり、齟齬がある場合はREC契約側を優先する。

## 1. なぜ新規5モデルが必要か（旧4bundleを流用しない理由）

NRQ-005〜008が使用した「reconstructed bundle seed 1–4」は、NRQ-004（ADR-0164）が
seed 0のcheckpoint消失（`2026-09-13`のcheckpoint overwrite）を修復するために作られた
**4モデルのcohort**である。これを「seed 0を含む5モデルcohort」として扱うことはできない
（seed 0は物理的に失われており、復元されていない）。

したがって、Phase Dの確認実験（本タスクでは実施しない、将来のD-00X実行task向け）には、
**同一の登録済み構築手順**で新規に5つの独立したCore/bankモデルを構築する。旧4bundleは
reference evidence（比較対象・回帰対照）として保持し、新cohortの構成要素として混成しない。

## 2. cohort seed（結果を見る前に固定）

既存repo内で使用済みの全seed namespaceと衝突しないよう、次の5 seedを固定する。

| 用途 | 既存使用範囲 | 出典 |
|---|---|---|
| NRQ-005〜008 reconstructed bundle model seed | `1, 2, 3, 4`（seed 0はNRQ-004で消失確認） | `NRQ006_REVIEW_RECORD.json: bundle_seeds_evaluated` |
| Model Bundle Recovery development seed | `10, 11, 12, 13, 14`（`RECOVERY_DEV_SEEDS`） | `src/apc/evaluation/model_bundle_recovery.py:728` |
| NRQ-006/007 data seed | `101–105` | 同上review record |
| NRQ-008 unused data seed pool | `201–220` | `docs/research/REPLICATION_AND_SUPPORT_BUDGET_SENSITIVITY_NRQ008.md` |

**Phase D five-model cohort model seed（新規固定）：`30, 31, 32, 33, 34`。**
上表のどの既存namespaceとも重複しない。データ生成seed（学習batch・評価batch）は
model seedとは独立のnamespaceとして `d001_train:<model_seed>:<step>:<op>` 形式の
決定的導出（`_derive_local_seed`と同じ設計、`unified_oracle_causal_benchmark.py:139-141`）
を用いる。評価seedは別途 §6 で固定する。

この5 seedは事前登録であり、実行後の成績によるcohort除外・置換・追加を禁止する
（本タスクの上位charter・AGENTS.mdの「試行後の候補選抜禁止」原則に従う）。

## 3. 構築手順（同一の登録済み手順）

Phase D cohortの各モデルは、reconstructed bundle（seed 1–4）を実際に構築した手順、すなわち
`src/apc/evaluation/model_bundle_recovery.py` の `_build_stage_graph()`
（`CORE_RESTORE → CANONICAL_AND_BRANCH_B_BUILD → SHIFT_DEDICATED → ...`）を、
**restoreの余地がない新規seedとして**実行する。既存4bundleとの違いは次の点のみである。

| Stage | 既存4bundle（seed 1–4）での実際の経路 | Phase D cohort（seed 30–34）での経路 |
|---|---|---|
| `CORE_RESTORE` | `runs/phase_a1_shift_compact_structural_probe/seed_{seed}/shared_encoder.pt` から既存checkpointを復元（新規学習不要） | 対応するseedのcheckpointは存在しないため、fallback経路
  `_get_or_train_frozen_shared_core`（`core_train_steps=16000`、既定`core_lr=0.0003`）で
  **新規にCoreを事前学習**する。これは既存4bundleの構築コストには含まれていなかった追加費用であり、
  §7の予算表で別項目として計上する。 |
| `CANONICAL_AND_BRANCH_B_BUILD` | `_ensure_learned_routing_bank_and_core`で8 canonical + 2 Branch-B primitiveを新規学習 | 同一手順・同一既定step数（parameterized primitiveは`bank_train_steps=6000`、
  parameter-free primitive（SORT含む）は`bank_train_steps//2=3000`、Branch-B novel opは3000） |
| `SHIFT_DEDICATED` | seed 10/11/14はrestore、seed 12/13は`REBUILD_REQUIRED_NOT_EXECUTED`のまま | seed 30–34は全て新規seedのため、restore対象なし。既存のSHIFT専用修復recipe
  （R3-009、ADR-0090、`shift_functional_generalization_repair.py`の`variant='iid_baseline'`既定）
  をそのまま新規実行する。この手順自体は既存実装済みであり、本契約が新設するものではない。 |
| router / argument scorer calibration | `router_train_steps=400`（既定） | 同一 |

**Coreの構築方法が既存4bundleと異なる（restore vs 新規pretrain）ことは、意図的な設計判断として
明記する。** これはCoreのarchitectureやhyperparameterを変えるものではなく、単に「同一
architecture・同一config・別seedのrandom initから新規に学習する」ことを意味する
（AGENTS.mdの「Coreは実験が要求する場合freeze」要件は、学習後freezeする点で両者とも満たす）。

## 4. 各bundleに記録する最小フィールド（既存`ModelBundleManifest`を再利用）

新規フィールドを追加する場合を除き、以下は全て `src/apc/utils/model_bundle.py` の
既存dataclassにすでに存在する。

| 要求フィールド（task指示） | 既存実装での対応 |
|---|---|
| core hash | `ModelBundleManifest.core.canonical_state_hash` |
| bank hash | `PrimitiveManifestEntry.weights_hash`（primitiveごと。SORTは他primitiveと別id） |
| token schema | `ModelBundleManifest.vocabulary.schema_hash` / `core.schema_hash` |
| architecture/config signature | `PrimitiveManifestEntry.architecture_signature` / `state_abi_hash`、
  および `ModelBundleManifest` 全体の `compute_execution_signature()` |
| source revision | `ModelBundleManifest.source_commit` |
| training-data lineage | `ModelBundleManifest.dataset_role_hashes` / `generator_version` |
| cohort identity | **既存に存在しない。新規フィールドが必要（§5）** |
| strict fresh-load verification record | `load_bundle(..., mode="nominal")` の戻り値
  `LoadedModelBundle.checks_performed`（tuple、実行した検査名の列） |

## 5. 新規フィールド：`cohort_id` / `cohort_manifest`

既存`ModelBundleManifest`には「このbundleがどのcohortに属するか」を表すフィールドが存在しない
（`model_seed`はモデル単体の識別子であり、cohort全体の識別子ではない）。本契約は次を提案する
（**提案のみ。本タスクでは実装しない**）。

```text
ModelBundleManifest.cohort_id: str          # 例: "phase_d_five_model_cohort_v1"
ModelBundleManifest.cohort_member_seeds: tuple[int, ...]  # (30, 31, 32, 33, 34) 固定・登録時凍結
ModelBundleManifest.cohort_construction_recipe_hash: str  # §3のstage graph定義のhash
```

`cohort_id`は`content_manifest_digest`の計算対象に含めない（cohortへの後付け命名がcontent同一性
を変えてはならない）。`assert_distinct_model_identities`（`model_bundle.py:977-987`）を
cohortの5モデルに対して実行し、同一Core内容が異なるseed labelの下で重複登録されていないことを
確認する（既存関数のまま、変更不要）。

## 6. strict fresh-load検証（namespace保護）

新規cohortの保存先namespaceを次に固定する：`runs/phase_d_five_model_cohort/seed_{30..34}/`。
既存の `RECOVERY_NAMESPACE_ROOT`（`runs/phase_b_b2_model_bundle_recovery/staging`）や
NRQ系runs（`runs/nrq00{4,5,6,7,8}_*`）とは独立した新namespaceであり、既存ファイルへの
上書きを行わない。`load_bundle`のfail-closed性質（`model_bundle.py:692-948`、
`MissingArtifactError`/`CorruptedArtifactError`等の型付き例外）により、namespace内の
manifest不整合は機械的にloadが失敗する。

fresh-load検証は、cohort構築完了後に**別プロセス**で次を実行することを登録する：
1. `load_bundle(manifest, mode="nominal", required_capabilities=...)` を5モデル全てに対して実行し、
   `checks_performed`に `manifest_self_consistency`, `core_integrity`,
   `schema_integrity`, `primitive_integrity_and_core_dependency`, `router_integrity`,
   `argument_scorer_integrity` が全て含まれることを確認する。
2. `assert_distinct_model_identities`を5モデルのmanifestに対して実行し、
   `DuplicateTrainingIdentityError`が発生しないことを確認する。
3. fresh-load直後にSORT-onlyの `STANDALONE_QUALIFIED` 回帰チェック（§regression panel、
   `docs/phase_d/PHASE_D_D001_TARGET_PANEL_MANIFEST.md`）を1回実行し、保存前の評価値と
   再現すること（`Δ = 0`、NRQ-006/007と同じ「2独立プロセスでのbitwise再現」規約に従う）。

## 7. 予算（親cohort構築のみ。修復・評価予算は別文書で分離）

親cohort構築の予算は、修復pilot（`PHASE_D_D001_SORT_ONLY_REPAIR_PILOT_PREREGISTRATION.md`）の
6,000 updates/model・30,000 updates/5modelという上限には**含まれない**。別枠として次を
確定する（steps/examplesは既存コードの既定値そのもの、wall-time/GPU/RAMは実測が存在しないため
明示的に見積り[ESTIMATE]と記す。§8参照）。

| 段階 | steps（1モデルあたり） | 1stepの例数 | 出典 |
|---|---:|---:|---|
| Core pretrain（fallback、新規seedのため必須） | 16,000 | バッチ構成は`SharedEncoderArchitectureConfig`依存（既存実装） | `core_train_steps`既定値, `unified_oracle_causal_benchmark.py:169` |
| Canonical parameterized primitive ×4（SELECT/COUNT/BIND/SHIFT） | 6,000×4=24,000 | 32 | `bank_train_steps`既定, `learned_routing_benchmark.py:112` |
| Canonical parameter-free primitive ×4（COPY/REVERSE/SORT/NEGATE） | 3,000×4=12,000 | 32 | `bank_train_steps//2`, `learned_routing_benchmark.py:630-632` |
| Branch-B novel primitive ×2（SWAP_PAIRS/INVERT_HALF） | 3,000×2=6,000 | 32 | `learned_routing_benchmark.py:639` |
| SHIFT dedicated repair（既存R3-009 recipe） | 既存recipeの既定budget（ADR-0090参照。本契約では新規に数値を再導出しない） | 既存recipe依存 | `shift_functional_generalization_repair.py` |
| Router calibration | 400 | — | `router_train_steps`既定, `learned_routing_benchmark.py:104` |

1モデルあたりのCore以外の合計は 24,000+12,000+6,000+400 = **42,400 steps**（SHIFT dedicated
repairの追加step数を除く）。5モデル合計 212,000 steps + Core 80,000 steps（16,000×5）。

**resident/active/temporary parameter accounting：** Core・8 canonical primitive・
2 Branch-B primitive・router・argument scorerは全て`resident`。cohort構築時は
"active"= 現在学習中の1primitiveのみ（他は`freeze_all()`または未構築でaccess外）、
`temporary`parameterはcohort構築のどの段階でも0（AGENTS.mdの「temporary capacityは
consolidation後にのみ生成」原則により、初期cohort構築はconsolidationの対象ではない）。

## 8. 未確定事項（本契約が確定できないもの）

- **wall-time / GPU時間 / peak VRAM：** 既存artifactにこの規模のprimitive学習1step当たりの
  実測steps/secが記録されていない（`docs/HARDWARE_ENVIRONMENT.md`はモデル規模別の目安のみで、
  この~20kパラメータ級primitiveの実測ではない）。本契約は仮の上限として
  「1 step ≤ 50ms（`docs/HARDWARE_ENVIRONMENT.md`の"30–100M model: straightforward"という
  記述に対し、本primitiveは4桁小さいことを根拠とする保守的な見積り[ESTIMATE]）」を採用し、
  Core pretrain 16,000 steps ≈ 13分/モデル、primitive学習42,400 steps ≈ 35分/モデル、
  5モデル合計 ≈ 4時間、を**計画用の上限見積り**として記録する。この数値は実測ではなく、
  実行時に実測値で置き換える必要がある。
  SHIFT dedicated repairのwall-timeはこの見積りに含まれていない（既存recipeの実測記録を
  別途参照する必要がある）。
- **`cohort_id`等の新規manifestフィールドの実装：** §5は提案のみで未実装。
- 本契約は5モデルcohortを**生成しない**。上記は将来実行task向けの事前登録budgetである。
