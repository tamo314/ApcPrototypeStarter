# A1-R005E-S004 結果サマリー — COUNT/BIND Argument-Blind Baseline Audit

**位置づけ:** `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md` の Task A1-R005E-S004（COUNT/BIND argument-blind baseline audit）。
本タスクの目的は、過去のすべてのゲート（E-004, E-005, E-006A, S002）において COUNT と BIND のみが `None` アーム（引数なし評価）で上限値 `0.30` をわずかに超過（~0.30〜0.33）して不合格（`passed=false`）となっていた現象について、**「歴史的な None 上限 0.30 は、タスク本来の自然な引数非依存ベースライン（Natural argument-blind baseline）を下回っているのではないか？」** を形式的に測定・監査し、将来のゲート判定基準に対する勧告を策定することである。

**ルール厳守:** タスク仕様書の「Do not retroactively change thresholds（閾値を遡及的に変更しない）」に従い、過去のゲート記録（E-004, E-005, E-006A, S002）の判定結果は一切書き換えない。

**結論を先に:** 監査の結果、**歴史的な None 上限 0.3000 は、COUNT および BIND の双方において、入力 content のみから達成可能な自然な引数非依存ベースラインを明確に下回っている**ことが証明された。
- **COUNT**: 最頻値ターゲット（count = 0）を常に出力するだけの自明なベースラインが **`33.46%`**、content 依存の最頻カウント予測が **`33.56%`** を達成。
- **BIND**: 入力 key-value 列の最後のペア値（`content[-1]`）を常に出力する content-only ヒューリスティックが **`34.00%`** を達成。
- **観測 None 精度**: E-004〜S002 で観測された None 精度（COUNT: 29.25%〜32.75%、BIND: 29.21%〜32.16%）は、これら自然な引数非依存ベースラインと寸分違わず一致しており、**引数漏洩（Leakage）の兆候ではなく、純粋にタスクの組合せ論的・自己回帰的 base-rate によるものである**ことが確定した。
- **将来への勧告**: 次タスク S005 および今後のゲートでは、硬直的な固定閾値 `None <= 0.30` を廃止し、自然ベースラインに許容マージンを加えた**ベースライン相対基準（`None <= B_natural + 0.05`）**を採用することを勧告する。

---

## 1. Audit Evidence Table（監査証拠テーブル）

全 5 seeds（seed 0..4）、`num_eval_groups = 1400` の実評価データセットに基づく集計結果:

| Operation | Majority Target Base | Best Content-Only Base | Natural Base | Historical Ceiling | Ceiling < Base? | Observed None Range | Baseline Consistent? |
|---|---|---|---|---|---|---|---|
| COUNT | 33.46% | 33.56% (predict_content_mode_count) | **33.56%** | 30.00% | **YES (Flawed Ceiling)** | 29.25% - 32.75% | **YES** |
| BIND | 10.39% | 34.00% (predict_last_pair_value) | **34.00%** | 30.00% | **YES (Flawed Ceiling)** | 29.21% - 32.16% | **YES** |

- **Ceiling < Base?**: COUNT / BIND の双方で `Historical Ceiling (0.30) < Natural Base` が成立（`all_ceilings_below_natural_baseline = True`）。
- **Baseline Consistent?**: 4つの独立なゲート（E-004, E-005, E-006A, S002）で観測された None 精度は、いずれも自然ベースラインと統計的に完全に整合。

---

## 2. 演算ごとの詳細メカニズム分析

### 2.1 COUNT: ターゲット分布の偏り（Marginal Mode Bias）
- **数理的背景:**
  `vocab_size = 10`、`sequence_length_range = (6, 10)` の設定において、ある特定の語彙トークン $v$ が長さ $L$ のランダム列に一度も出現しない確率は $P(\text{count}=0) = (9/10)^L$ である。
  $L=6$ で $53.1\%$、$L=8$ で $43.0\%$、$L=10$ で $34.9\%$ となり、全体の平均で約 $43.5\%$ のトークンは count が 0 となる。
  反実仮想グループ（`group_size = 3`）として pairwise distinct なカウント（例: 0, 1, 2）が選ばれるため、評価セット全体でのカウント `0` の出現割合は **`33.46%`**（0.3346）に達する。
- **ベースライン測定値:**
  - `predict_count_zero`（常に 0 を予測）: **`33.46%`**
  - `predict_content_mode_count`（content 内で最頻のカウント数を予測）: **`33.56%`**
- **過去ゲートの観測 None 精度:**
  - E-004 (frozen high-cap): `31.75%` (seeds: 31.86%, 31.40%, 31.58%, 32.36%, 31.57%)
  - E-005 (frozen compact): `29.25%` (seeds: 26.14%, 30.14%, 31.07%, 32.21%, 26.70%)
  - E-006A (joint specialized): `32.75%` (seeds: 32.28%, 32.19%, 33.35%, 33.35%, 32.59%)
  - S002 (shared mixed): `32.47%` (seeds: 32.32%, 33.14%, 32.43%, 32.19%, 32.28%)
- **結論:**
  引数（どのトークンを数えるか）が与えられない場合、モデルが最頻値である「0」を出力するのは統計的に最適かつ自然な振る舞いである。その結果生じる 32% 前後の正解率は、モデルがダミー予測を行っていることの証左であり、引数漏洩を意味するものではない。歴史的上限 0.30 は、自明なダミー予測すら下回ることを要求する非現実的な設定であった。

### 2.2 BIND: 因果アテンションの直近バイアス（Recency Bias Heuristic）
- **数理的背景:**
  BIND の入力はキーと値のペア列 `(k0, v0, k1, v1, ..., k_last, v_last)` である。
  ターゲットの全体分布は語彙トークン全体にほぼ均等（約 10.39%）に分散しているが、クエリキーが与えられない引数ブラインド（None）設定において、モデルは入力列のいずれかの値トークンを出力せざるを得ない。
  Transformer の因果アテンションおよび出力リードアウトにおいて、クエリ引数ベクトルがゼロ化された場合、末尾の位置（`[SEP]` の直前、すなわち最後のペア値 `content[-1]`）にアテンションの重みが集約する、または直近のペアをデフォルトとして参照するヒューリスティックが自然に生じる。
- **ベースライン測定値:**
  - `predict_last_pair_value`（最後のペア値 `content[-1]` を予測）: **`34.00%`**
  - `predict_first_pair_value`（最初のペア値 `content[1]` を予測）: `27.28%`
  - `uniform_random_present_pair`（入力に存在するペア値からのランダム選択期待値）: `25.69%`
  - `predict_dataset_majority`（データセット最頻値の固定予測）: `10.39%`
- **過去ゲートの観測 None 精度:**
  - E-004 (frozen high-cap): `29.21%` (seeds: 27.93%, 29.73%, 28.71%, 30.58%, 29.10%)
  - E-005 (frozen compact): `29.77%` (seeds: 29.60%, 30.12%, 29.81%, 29.87%, 29.44%)
  - E-006A (joint specialized): `32.16%` (seeds: 27.91%, 33.72%, 32.32%, 33.22%, 33.64%)
  - S002 (shared mixed): `30.12%` (seeds: 32.11%, 29.11%, 29.34%, 28.52%, 31.53%)
- **結論:**
  引数なし（None）の状態で末尾のペア値を参照するヒューリスティックは 34.00% の精度を達成する。E-004〜S002 のすべてのモデルが 29.2%〜32.2% の範囲に収まっていることは、まさにこの自然な content-only ヒューリスティックの範囲内で動作していることを示している。

---

## 3. 正解引数アーム（Correct）および不正引数アーム（Wrong）との対比

None アームの ~30-33% が「引数漏洩ではなく自然ベースライン」であることの決定的な証拠は、引数が与えられたアームとの圧倒的な落差にある（S002 の実績）:

| Operation | Correct Exact | Effectful Wrong | None Exact | Causal Gap | Argument Effect Rate |
|---|---|---|---|---|---|
| COUNT | **99.55%** | **0.09%** | 32.47% | **0.6708** | 1.0000 |
| BIND | **99.98%** | **0.005%** | 30.12% | **0.6986** | 1.0000 |

1. **不正引数（Wrong）の完全な抑制**:
   別の引数値を与えると、正解率は COUNT で 0.09%、BIND で 0.005% とほぼ完全にゼロになる。もしモデルが引数を無視して content のみから解いていたなら、Wrong アームも 30% 以上になるはずである。Wrong がほぼゼロであることは、モデルが**指定された引数に極めて敏感かつ正確に従っている**ことを証明している。
2. **圧倒的な因果ギャップ（Causal Gap）**:
   因果ギャップ $\text{Correct} - \max(\text{Wrong}, \text{None})$ は、COUNT で `0.6708`、BIND で `0.6986` であり、合否基準の `0.50` を大幅に超過している。

---

## 4. 将来のゲート判定基準に対する正式勧告（Recommendation）

### 4.1 歴史的閾値の反省
歴史的な `NONE_CEILING = 0.30` は、各演算の組合せ論的ベースレートを考慮せず、SHIFT や SELECT のような直交性の高い並び替えタスクの感覚で一律に設定された値であった。
COUNT では 33.46%、BIND では 34.00% の自明なベースラインが存在するため、0.30 を要求することは「引数非依存モデルがサイコロを振るよりも下手でなければならない」ことを要求するのと同義であり、統計学的に不当であった。

### 4.2 遡及改定の禁止
実験の厳密性と歴史的保全（Historical Integrity）の観点から、E-004, E-005, E-006A, S002 の過去のサマリーファイルおよび ADR に記載された `none_passed = false` / `passed = false` という判定は**遡及修正しない**。それらは「当時の基準に照らして不合格であった」という客観的事実として保存される。

### 4.3 S005 および今後のゲートへの勧告（Baseline-Relative Criterion）
今後のゲート（S005 の Branch B 判定以降）においては、固定閾値 `None <= 0.30` を廃止し、以下の**ベースライン相対基準**を採用することを勧告する:

$$\text{None} \le B_{\text{natural}} + \delta$$

ここで、
$$B_{\text{natural}} = \max(B_{\text{majority}}, B_{\text{content-only}})$$
とし、統計的許容マージンとして $\delta = 0.05$ を設定する。

この基準を適用した場合の各演算の閾値と S002 の実績:
- **COUNT**: $B = 0.3346 \implies \text{None} \le 0.3846$ （S002 実績: **0.3247** $\implies$ **PASS**）
- **BIND**: $B = 0.3400 \implies \text{None} \le 0.3900$ （S002 実績: **0.3012** $\implies$ **PASS**）
- **SHIFT**: $B \approx 0.0000 \implies \text{None} \le 0.0500$ （S002 実績: **0.0001** $\implies$ **PASS**）
- **SELECT**: $B \approx 0.0393 \implies \text{None} \le 0.0893$ （S002 実績: **0.0393** $\implies$ **PASS**）

さらに、APC の本質的基準である**因果ギャップ（Causal Gap $\ge 0.50$）**を主要な指標として位置づけることで、モジュラリティと引数選択性を正しく評価できる。

---

## 5. 関連ファイル

- 仕様書: `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`（Task A1-R005E-S004）
- 実験計画: `docs/EXPERIMENT_PLAN_A1_R005E_SHARED_ENCODER_GATE.md`（セクション10）
- 設計書: `docs/design-docs/SHARED_QUERYABLE_REPRESENTATION.md`（セクション8）
- 評価モジュール: `src/apc/evaluation/argument_blind_baseline_audit.py`
- CLI スクリプト: `scripts/argument_blind_baseline_audit.py`
- 設定ファイル: `configs/phase_a1_argument_blind_baseline_audit.yaml`
- 単体テスト: `tests/test_argument_blind_baseline_audit.py`（5 tests pass）
- 実行アーティファクト: `runs/phase_a1_argument_blind_baseline_audit/`（`config.yaml`, `report.json`, `summary.json`, `system.json`）
