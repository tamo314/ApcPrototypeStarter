# AI Coding Task — SciPy Dependency Contract Repair & REC-004L Environment Requalification

**正式ID：B-C005REC-004L-ENV1**  
**版：1.0 / 2026-09-10**  
**状態：環境・依存契約修復のみ。研究モデルの学習・評価修正は対象外。**  
**位置：REC-004L Stage 0 が `ENVIRONMENT_DEPENDENCY_BLOCKED` で停止した後、REC-004L本体を再実行する前。**

## 0. 目的と終了境界

直前のREC-004K / ADR-0106では、task固有31 testsはPASSした一方、full repositoryでは`ModuleNotFoundError: No module named 'scipy'`により28 testsが失敗した。報告上、失敗は`src/apc/evaluation/functional_metrics_v2.py`のSciPy importを経由し、SciPyはrepositoryの正式なdependency metadataに宣言されていなかった。REC-004Lはこの状態を検知して`ENVIRONMENT_DEPENDENCY_BLOCKED`で停止した。

本タスクの目的は、**今のshellへSciPyを手動導入することではなく、repositoryの正式なdependency contractからclean environmentを再構築したときにSciPy依存が必ず解決される状態へ修復すること**である。

実施範囲は以下だけ。

1. SciPyの実import siteと必要scopeをlive sourceから監査する。
2. authoritative dependency metadataを実依存と一致させる。
3. clean isolated environmentをprojectの正式install手順だけから構築する。
4. SciPy欠落で失敗したtests、REC-004K/004L周辺tests、full repository verificationを再実行する。
5. REC-004Lのenvironment readinessだけを確認してSTOPする。

**REC-004L本体、optimizer update、GPU研究実験、checkpoint診断、child bundle、RG3、REC-005、R3-011/012、B-C006、Task Inferenceは自動実行しない。**

```text
REC-004K: task-specific PASS / full repo 28 scipy-import failures
  ↓
REC-004L Stage 0: ENVIRONMENT_DEPENDENCY_BLOCKED
  ↓
REC-004L-ENV1
 A. import/dependency audit
 B. dependency metadata repair
 C. clean-install proof
 D. focused + full verification
 E. REC-004L preflight only
  ↓
STOP
```

## 1. Source factsと未確定事項

### 1.1 開始点として扱う報告事項

- REC-004K固有31 testsはPASS。
- full repoは`2293 passed, 28 failed, 4 warnings`。
- 28 failuresはSciPy import不可が共通原因と報告された。
- 少なくとも以下が影響対象として報告された。
  - `tests/test_adequacy_verifier.py`
  - `tests/test_functional_metrics_v2.py`
  - `tests/test_paired_baseline_repair.py`
  - `tests/test_safe_bounded_verification.py`
- import経路は`src/apc/evaluation/functional_metrics_v2.py`。
- ADR-0105時点では同じtest群を含むfull suiteがgreenだったため、以前の環境にはSciPyが暗黙に存在した可能性がある。

**現在checkoutの実ファイルで必ず再確認する。**

### 1.2 結果前に決めないこと

- SciPyをcore dependencyへ入れるべきか。
- dev/test-only dependencyなのか。
- optional extraなのか。
- 以前動いていたSciPy versionがminimum supported versionなのか。
- `pip install scipy`だけでrepository defectが解決するのか。
- SciPyをstdlib/PyTorch実装へ置換すべきか。

## 2. 変更範囲

### 許可

- `pyproject.toml`または現repoのauthoritative dependency manifest。
- 既存lockfile/constraints/environment fileが正式install contractの一部なら、その同期。
- dependency contract用の小さなtest。
- optional dependencyだと実コードから明確に確認できた場合だけ、既存packaging policyに従うoptional-extra宣言と明示error。
- install/dependency documentation、AGENTS/ADR/index追記。
- 本タスク固有のrun artifacts。

### 不変

- `src/apc/core/**`
- primitiveの数式・parameter構造・学習済みweight
- `functional_metrics_v2`の統計的意味、tau、alpha、look schedule、verdict semantics
- adequacy verifierの判定規則
- router / ArgumentScorer / controller
- REC-004D〜Kの全checkpoint/training state/run artifact
- shared cache
- sealed/final-query data
- 全歴史PASS/FAIL

### 禁止

- final fixをmanual `pip install scipy`だけで済ませる。
- SciPyをvendorする。
- SciPy統計関数を独自近似実装へ置換する。
- import失敗時に機能を黙ってskipする。
- failing testsをskip/xfailへ変更する。
- test discoveryを変更して失敗を隠す。
- dependency追加と統計ロジック変更を同じtaskで混ぜる。
- unrelated dependencyの一括upgrade。
- Python/PyTorch version変更。
- REC-004L本体の自動再実行。

## 3. Stage A — Import / dependency contract audit

### A1. SciPy import graph

repository全体で`scipy`を検索し、`dependency_audit.json`へ最低限以下を保存する。

```text
import_site
import_form
imported_symbols
module_import_or_function_local
runtime_path
test_only_or_runtime
optional_guard_present
known_callers
```

名前から`scipy.stats`等を推測せずlive sourceを正とする。

### A2. authoritative dependency metadata

以下を確認する。

- `pyproject.toml`
- requirements / constraints / lock files
- CI / README / AGENTSのcanonical install command
- editable install / dev extra / test extraの既存policy
- supported Python version
- 現environmentで`python -m pip show scipy`と`import scipy`
- ADR-0105/0106 artifactにworking SciPy version記録があるか

source-of-truthと生成物を区別し、`dependency_contract_decision.json`へ記録する。

### A3. dependency scope classification

#### `CORE_RUNTIME_REQUIRED`
installed APCの通常runtimeからSciPyへ到達し、optional設計でない場合。  
→ authoritative runtime dependencyへ追加。

#### `DEV_OR_TEST_ONLY`
SciPy到達がtest/toolingだけの場合。  
→ existing dev/test dependency groupへ追加。

#### `OPTIONAL_RUNTIME`
機能が既に明示的optionalで、通常runtimeはSciPy無しでも成立する場合。  
→ existing optional-extra policyへ追加。optional feature利用時だけ明示error。

現状がunconditional runtime importなら、今回都合よくoptionalへ再設計しない。

### A4. STOP

以下ならmetadataを変更せずSTOP。

```text
SCIPY_NOT_ACTUALLY_CAUSAL
DEPENDENCY_SCOPE_UNRESOLVED
AUTHORITATIVE_MANIFEST_UNRESOLVED
```

## 4. Stage B — Metadata repair

Stage Aでscopeが確定した場合だけ実行。

### B1. requirementの書式

repository既存styleへ従う。

- lower-bound policyなら同形式。
- range policyなら同形式。
- lockfileが正式なら通常手順で同期。
- **根拠のないversion boundを作らない。**

過去green環境のversionを確認できた場合は`KNOWN_WORKING_VERSION`として記録できるが、それをminimum supported versionと呼ばない。

minimumの根拠がなく、project policyがbare requirementを許容するなら、推測boundより正式に`scipy`依存を宣言し、clean installで解決されたversionをenvironment manifestへ記録する。

### B2. scientific code

`src/apc/evaluation/functional_metrics_v2.py`は原則変更しない。

optional dependencyであることがStage Aで明確な場合だけ、import boundary/error messageの最小変更を許可する。統計式・数値計算は変更しない。

### B3. dependency contract test

可能ならTOML/packaging metadataを構造的に読み、分類したscopeにSciPy requirementが存在することを確認するtestを追加する。単純grepだけに依存しない。

既存`functional_metrics_v2` fixtureの数値が修復前後で不変であることも確認する。

## 5. Stage C — Clean environment proof

### C1. isolated environment

現在のshellへSciPyを追加してacceptance proofにしない。

projectの既存canonical workflowで新しいtask-local isolated environmentを作る。

例示のみ：

```bash
python3.12 -m venv <task-local-env>
<env-python> -m pip install -e ".[existing-extra-if-required]"
```

実commandはrepo policyへ合わせる。

### C2. install source

acceptance environmentでは、先に個別`pip install scipy`を実行してはならない。

**修復済みproject metadata / lock / canonical install commandだけからSciPyが導入されること。**

package index/network unavailableなら`DEPENDENCY_RESOLUTION_UNAVAILABLE`でSTOPし、clean-install PASSを主張しない。

### C3. environment manifest

`environment_requalification.json`へ保存：

```text
python_version
python_executable
platform
torch_version
scipy_version
scipy_location
project_install_command
dependency_source
lock_or_constraints_hash
import_scipy
import_apc
import_functional_metrics_v2
pip_check_result
```

`python -m pip check`またはpackage managerの等価checkを実行。

## 6. Stage D — Verification

### D1. SciPy欠落で失敗したtest群

clean environmentで最低限：

```bash
python -m pytest -q   tests/test_adequacy_verifier.py   tests/test_functional_metrics_v2.py   tests/test_paired_baseline_repair.py   tests/test_safe_bounded_verification.py
```

0 failure必須。

SciPy importは直ったが別errorが出るなら：

```text
SCIPY_IMPORT_FIXED_BUT_TESTS_STILL_FAIL
```

としてSTOP。別logic修正を混ぜない。

### D2. REC-004K / REC-004L周辺

最低限：

```text
tests/test_mirror_temporal_mechanism_rollback_audit.py
REC-004Lのtest file（既に存在する場合）
```

REC-004L testがimport/collection可能なことまで確認する。GPU/main runはしない。

### D3. full repository

clean environmentから：

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src/apc
```

ADR-0106時点の既存test総数は`2293 + 28 = 2321`。新test追加で増えてよい。総数へ合わせるための削除/skipは禁止。

Gate：

```text
pytest failures = 0
ruff exit = 0
mypy exit = 0
```

pre-existing warningは内容と件数を報告し、warning修正をこのtaskへ混ぜない。

## 7. Stage E — REC-004L environment readiness only

full repo greenの場合だけ実施。

REC-004Lにpreflight-only入口が既にあるなら、それだけを実行して、

```text
scipy_importable = true
affected_tests_green = true
full_repo_green = true
environment_dependency_blocked = false
```

相当を確認する。

preflight-only入口が無ければREC-004L runner本体を起動せず、本task自身の確認から

```text
REC004L_ENVIRONMENT_READY
```

を記録する。

**REC-004L Stage A以降は実行しない。**

## 8. Acceptance Gate

全て必要：

```text
dependency_scope_resolved = true
authoritative_dependency_metadata_updated = true
clean_environment_created = true
scipy_installed_via_project_contract = true
dependency_check = PASS
affected_scipy_failure_tests = PASS
rec004k_focused_regression = PASS
rec004l_preflight_or_equivalent = PASS
full_pytest = PASS
ruff = PASS
mypy = PASS
scientific_semantics_changed = false
model_artifacts_mutated = false
```

最終status：

```text
implementation_status: COMPLETE / PARTIAL / BLOCKED
dependency_classification: CORE_RUNTIME_REQUIRED / DEV_OR_TEST_ONLY / OPTIONAL_RUNTIME / UNRESOLVED
dependency_contract_status: PASS / FAIL / BLOCKED
clean_install_status: PASS / FAIL / NOT_EXECUTED
affected_test_status: PASS / FAIL / NOT_EXECUTED
full_repo_verification: PASS / FAIL / NOT_EXECUTED
rec004l_environment_ready: true / false
rec004l_executed: false
new_optimizer_updates: 0
child_bundle: null
rg3_recheck: NOT_EXECUTED
rec005_eligible: false
```

## 9. Failure STOP rules

| 状況 | 処理 |
|---|---|
| SciPyが原因でない | `SCIPY_NOT_ACTUALLY_CAUSAL` |
| dependency scope不明 | `DEPENDENCY_SCOPE_UNRESOLVED` |
| source-of-truth不明 | `AUTHORITATIVE_MANIFEST_UNRESOLVED` |
| package resolution不能 | `DEPENDENCY_RESOLUTION_UNAVAILABLE` |
| clean installでSciPy無し | metadata/lock不成立を報告してSTOP |
| SciPy後も対象tests失敗 | `SCIPY_IMPORT_FIXED_BUT_TESTS_STILL_FAIL` |
| full suiteに別failure | failure保存、unrelated修正をせずSTOP |
| 全Gate PASS | REC-004L readiness確認だけしてSTOP |

## 10. 必須成果物

```text
dependency_audit.json
dependency_contract_decision.json
dependency_metadata_diff.txt
clean_install_log.txt
environment_requalification.json
affected_tests.txt
focused_regression_tests.txt
full_pytest.txt
ruff.txt
mypy.txt
rec004l_environment_preflight.json
freeze_audit.json
side_effect_audit.json
summary.json
report.md
```

tracked dependency fileのbefore/after hashを保存する。checkpoint/Core/bank/router/scorer/shared-cacheも開始前後で不変を確認する。

## 11. 完了報告テンプレート

```text
Task: B-C005REC-004L-ENV1
Root cause:
Actual SciPy import sites:
Dependency classification:
Authoritative dependency source:
Metadata change:
Resolved SciPy version in clean env:
Python / PyTorch versions:
Clean install command:
Dependency check:
Affected tests:
REC-004K focused regression:
REC-004L preflight:
Full pytest:
Ruff:
Mypy:
Scientific/statistical code changed: yes/no
Model/checkpoint/cache changed: yes/no
New optimizer updates: 0
implementation_status:
dependency_contract_status:
clean_install_status:
affected_test_status:
full_repo_verification:
rec004l_environment_ready:
rec004l_executed: false
Remaining blocker:
Next explicit task:
```

## 12. 研究状態は変更しない

このtaskがPASSして意味するのは、**dependency/environment contractが再現可能になった**ことだけ。

以下は意味しない。

- I03 downstream interaction問題の解決
- REC-004L PASS
- MIRROR_HALVES recovery floor達成
- child bundle作成可能
- RG3 PASS
- REC-005開始可能

PASS後に可能になるのは、ユーザーが明示した場合の**REC-004L再実行**だけ。

**STOP：environment/dependency修復結果を報告して終了する。REC-004L本体を自動実行しない。**
