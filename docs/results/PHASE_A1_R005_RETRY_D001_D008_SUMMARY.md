# A1-R005 再試行 結果サマリー(A1-R005D-001〜A1-R005D-008)

**位置づけ:** `docs/CODEX_TASKS_A1_R005_RETRY.md` の A1-R005D-001〜A1-R005D-008。個別の設計判断・測定値はすべて `docs/DECISIONS.md` ADR-0030〜ADR-0037 に記録済みで、本レポートはそれらを結果ベースで串刺しにしたもの。**新規の実験・実装はこのレポートに含まない**(既存 ADR の要約と、次の仮説検討のための整理のみ)。

**結論を先に:** A1-R005D-001〜003 は診断/前提整備タスクとして完了(SELECT のエンコーダ欠陥を発見・修正)。**A1-R005D-004(COUNT)・A1-R005D-007(BIND)・A1-R005D-008(SHIFT/SELECT)はいずれも STOP GATE FAIL**。A1-R005D-005(容量/学習量スイープ)は全4ステージ未達、A1-R005D-006(conditioning architecture比較)は3variant中どれも未達。**パラメータ化された4演算(COUNT/BIND/SHIFT/SELECT)すべてが、共有のベースラインアーキテクチャ(V0 additive `ConditionedPrimitive`, rank=8, arg_dim=16)で STOP GATE に失敗**という収束的な結果になっている。`AGENTS.md` の STOP GATE discipline により A1-R006 は引き続きブロック中。

---

## 1. 背景: 何を検証している一連のタスクか

`docs/DECISIONS.md` ADR-0029(元の A1-R005, STOP GATE FAIL)が起点。`SHIFT`/`SELECT`/`COUNT`/`BIND` の4演算について「正しい family **かつ** 正しい引数のときに高精度、それ以外(誤引数/誤family/primitiveなし)で materially 低い精度になる」ことを示せなかった(Correct 0.308、causal gap 0.053)。

`docs/design-docs/PARAMETERIZED_PRIMITIVE_RETRY.md` はこの失敗の候補説明を4つ挙げていた:

- **F1**: Wrong-argument control が部分的に非因果的(→ ADR-0030 で否定)
- **F2**: 引数エンコーダの表現衝突(collision)(→ ADR-0031/0032 で SELECT のみ確認・修正)
- **F3**: 加算的conditioning(`B(phi(A(h)+C(e_a)))`)が弱すぎる
- **F4**: 学習が引数使用を強制していない(i.i.d.サンプリングでは content だけで損失を最小化できてしまう)

A1-R005D-001〜003 は F1/F2 を診断・解消する前提整備タスク、A1-R005D-004〜008 は F3/F4 を実際に潰しにいく counterfactual gate 群(`docs/AGENTS_A1_R005_RETRY_ADDENDUM.md` の "Counterfactual training rule": 同一content・複数引数を同一学習グループに提示し、引数以外に正解を説明する情報を残さない)。

---

## 2. タスク別結果

### A1-R005D-001 — 既存R005アーティファクトの再分析(再学習なし)— **完了**(ADR-0030)

**目的:** 既存の A1-R005 実行結果から、再学習せずに診断情報を追加抽出する。

**実装:** `apc.evaluation.parameterized_primitive_gate_reanalysis` が既存アーティファクトを読み取り専用で再解析。`argument_effect_rate`(誤引数が実際に正解を変えるか)とtarget長を、モデルなしで決定的に再計算。

**結果:**

| Operation | Correct | Wrong argument (raw) | gap | argument_effect_rate | mean target length |
|---|---|---|---|---|---|
| SHIFT | 0.161 | 0.117 | 0.044 | **1.000** | 8.04 |
| SELECT | 0.047 | 0.042 | 0.005 | **0.992** | 3.80 |
| COUNT | 0.595 | 0.443 | 0.152 | 0.667 | 1.00 |
| BIND | 0.420 | 0.411 | 0.009 | 0.900 | 1.00 |

**含意:** F1(Wrong-argument controlの非因果性)は否定 — 誤引数は圧倒的多数のケースで実際に正解を変えている。COUNTは4演算中最もargument_effect_rateが低い(0.667)にもかかわらず最大のgap(0.152)を持ち、相対的に最も引数選択的。token accuracyと有効(effectful)wrong-argument絞り込みは、per-example予測が保存されていないため再計算不可能と正直に報告(捏造・省略せず)。

---

### A1-R005D-002 — 引数エンコーダ整合性監査 — **完了、SELECTは非単射でSTOP GATE発動**(ADR-0031)

**目的:** 学習前に表現衝突(collision)を検出する。

**実装:** `apc.primitives.argument_encoder_audit` がモデル構築・学習なしで各`ArgumentEncoder`を直接監査。

**結果:**
- **SELECT(`IndexSetArgumentEncoder`, mean pooling)**: `[1,3]` vs `[3,1]` を含む5つの順列ペアすべてが、3つの独立シードすべてで衝突(構造的・重み非依存 — meanは可換なため)。
- **SHIFT/COUNT/BIND(`IntBucketArgumentEncoder`)**: 実際の合法ドメイン上ではすべて単射(SHIFT: `range(10)`の100ペア、COUNT/BIND: `range(10)`の45ペアずつ、衝突ゼロ)。範囲外の意図的wraparoundは正常動作として確認。

**含意:** SELECTの構造的欠陥がADR-0029/ADR-0030の`SELECT` Correct最低値(0.047)・gap最小値(0.005)と整合。STOP GATE発動: 今後`IndexSetArgumentEncoder`経由でSELECTを学習することを禁止。SHIFT/COUNT/BINDはこの観点からは問題なし(=これらの弱い引数選択性はエンコーディングの問題ではない)。

---

### A1-R005D-003 — Order-preserving SELECT引数エンコーダ — **実装完了**(ADR-0032)

**目的:** D-002が順序損失を確認したため、mean poolingを置き換える。

**実装:** 新規`OrderPreservingIndexSetArgumentEncoder`(index embedding + position embedding → 1層のself-attention → masked mean pool)。`default_argument_encoder("SELECT", ...)`のデフォルトに昇格。

**結果:** `[1,3]`/`[3,1]`が単一シードでも5シード横断でも区別可能に。可変長・パディングマスクのテストも通過。

**含意:** これは**構造的な区別可能性**を示したに過ぎず、「学習によって実際に引数選択的になるか」はD-008/D-009で初めて測定される、と明記して保留(この判断は正しかった — D-008で実際に測定した結果は下記)。

---

### A1-R005D-004 — Counterfactual COUNT gate(STOP GATE)— **FAIL**(ADR-0033)

**目的:** F4(学習が引数使用を強制していない)を切り分ける。

**実装:** `apc.evaluation.count_counterfactual_gate` — 同一contentに対し複数の`COUNT.target`(出力が pairwise distinct になるよう探索選定)を同一学習ステップに含める。task-blindなStable Coreは通常のi.i.d.学習のまま。

**結果**(5 seeds, RTX 5060 Ti, commit `7ae9ec8`, `primitive_rank=8`/`arg_dim=16`/`primitive_train.steps=10000` — A1-R005と同一の学習量):

| Arm | 平均 | 判定基準 |
|---|---|---|
| Correct | **0.450**(stdev 0.035, min 0.392, max 0.480) | ≥0.90 |
| effectful Wrong argument | 0.262 | ≤0.30 |
| None | 0.336 | ≤0.30(未達) |
| causal gap | **0.114** | ≥0.50 |
| argument_effect_rate | **1.000**(全seed、構成上保証) | — |

**含意:** F4は「単独では十分な説明ではない」ことが判明。引数多様性を学習データ中で最大限強制しても(argument_effect_rate 0.667→1.000)、gapはA1-R005の元の値(0.152)より*改善しなかった*(0.114)。F3(加算的conditioningが弱い)がより有力な残存候補として浮上。

---

### A1-R005D-005 — 最小容量/学習量スイープ — **全4ステージ未達**(ADR-0034)

**目的:** F4が排除できないままF3を疑う前に、容量/学習量だけで説明がつくかを検証する。

**実装:** D-004のゲートをそのまま再利用し、1変数ずつ独立にスイープ(D-004ベースラインに対して各1ステージ)。

**結果**(各5 seeds、D-004と同じ受け入れ基準):

| Stage | 変更内容 | Correct | effectful Wrong | causal gap | params | 判定 |
|---|---|---|---|---|---|---|
| (D-004 baseline) | rank=8, steps=10000, arg_dim=16 | 0.450 | 0.262 | 0.114 | — | FAIL |
| `steps_x2` | primitive_train.steps=20000 | 0.500 | 0.235 | 0.164 | 3360 | FAIL |
| `steps_x4` | primitive_train.steps=40000 | 0.558 | 0.209 | **0.222** | 3360 | FAIL |
| `rank_16` | primitive_rank=16 | 0.539 | 0.214 | 0.203 | 6560 | FAIL |
| `arg_dim_x2` | arg_dim=32 | 0.473 | 0.250 | 0.137 | 3648 | FAIL |

**含意:** 4つのレバーはすべてgapを改善する方向に動くが、いずれも要求(0.50)の半分にも届かない。`rank_16`が`steps_x4`と同程度の改善を追加学習コストなしで達成 — 容量不足も部分的要因だが、`arg_dim_x2`の効果が最も弱いことから、ボトルネックは「エンコーダの表現力不足」ではなく「加算的な混合formula自体」に絞り込まれる(F3を支持)。Noneは全ステージで完全に同一(0.336) — frozen Stable Coreは容量スイープの影響を受けないため、期待通り。

---

### A1-R005D-006 — Conditioning architecture comparison(STOP GATE)— **FAIL(3variant中どれも未達)**(ADR-0035)

**目的:** 加算的conditioning自体がボトルネックかを検証する。

**実装:** `ArgumentConditionedPrimitive`基底クラスを新設し、V0(additive, 既存)・V1(`FiLMConditionedPrimitive`, ゲート型乗算)・V2(`BasisModulatedConditionedPrimitive`, 共有基底への凸結合)を実装。D-004のベースライン容量・学習量・データを完全固定し、formulaのみ変更。

**結果**(各5 seeds、D-004ベースライン容量固定):

| Variant | Params | Correct | effectful Wrong | causal gap | 判定 |
|---|---:|---|---|---|---|
| V0 additive(再実行、ADR-0033と完全一致) | 3360 | 0.450 | 0.262 | 0.114 | FAIL |
| V1 FiLM | 3504 | **0.510** | **0.231** | **0.174** | FAIL |
| V2 basis(num_basis=8) | 3432 | 0.368 | 0.318 | 0.032 | FAIL |

`disqualified_variants=[]`(CorrectとWrong argumentを同時に悪化させたvariantはない)。`best_by_causal_gap="film"`だが`selected_variant=null`。

**含意:** V1(FiLM)はCorrectを上げつつWrong argumentを下げる形で改善(F3を部分的に支持) — しかし要求の1/3にとどまる。V2(basis)は全指標で悪化(「conditioning mapを自由にしすぎている」という別解釈を否定する材料)。「加算的conditioningが完全に壊れている」という単純な説明は否定されたが、formula変更だけで問題は解決しない。

---

### A1-R005D-007 — Counterfactual BIND gate — **FAIL(causal gapが統計的にゼロ)**(ADR-0036)

**目的:** D-004と同じcounterfactualプロトコルをBINDに適用する(D-006のアーキテクチャ比較はBINDでは繰り返さない — D-004のベースラインをそのまま使用)。

**実装:** `apc.evaluation.bind_counterfactual_gate` — BINDは偶数長のkey/value入力が必須、`query_key`はcontent中に実在するkeyからのみサンプルするため専用のグループ生成器を新設。

**結果**(5 seeds, RTX 5060 Ti, commit `7ae9ec8`, D-004と同一ベースライン):

| Arm | 平均 | 判定基準 |
|---|---|---|
| Correct | 0.332(stdev 0.004) | ≥0.90 |
| effectful Wrong argument | 0.338 | ≤0.30 |
| None | 0.335 | ≤0.30 |
| causal gap | **-0.005** | ≥0.50 |
| argument_effect_rate | 1.000 | — |

**含意:** COUNTと質的に異なる、より深刻な失敗。Correct・Wrong argument・Noneが統計的に見分けがつかない——正しい引数を強制しても間違った引数を強制しても引数なしでも結果が同じ。オフライン検証により、「コンテンツ内の**最後のkey-value対の値**を引数を無視して出力する」ヒューリスティックが0.338の精度を出し、観測された3つのarmすべてとほぼ一致することを確認。BINDは真の連想検索(query_keyが指す**位置**を探して隣接値を読む)を要求するが、rank-8の加算的残差conditioningは引数によって位置ごとの注意を誘導する構造を持たない、という新しい候補説明(COUNT/BIND間の相違を説明するもの)を提示。

---

### A1-R005D-008 — SHIFT/SELECT sequence gates — **FAIL(両方)**(ADR-0037)

**目的:** SHIFT/SELECT(マルチトークン出力)にcounterfactualプロトコルを適用する。exact match/token accuracy/長さ別内訳/effectful Wrong argumentを測定。「COUNT/BINDが強い結果であるにもかかわらず失敗した場合」にのみクロスアテンション代替primitiveを実装する、という条件付き指示があったが、COUNT(D-004〜006)・BIND(D-007)いずれも既に失敗しているため前提が成立せず、**代替primitiveは実装していない**。

**実装:** `apc.evaluation.sequence_counterfactual_gate`(SHIFT/SELECT共通、`operation`引数で切替)。D-004/D-007のcounterfactual group discipline を再利用。「選定済みの低rank conditioned architecture」はD-005/D-006がいずれも「何も選ばれなかった」ため、これまでのV0 additiveベースラインをそのまま使用。SELECTのorder-preserving encoder必須要件はD-003で既に実装済み。

**結果**(各5 seeds、D-004/D-007と同一ベースライン):

| Arm | SHIFT | SELECT | 判定基準 |
|---|---|---|---|
| Correct exact match | 0.171 | 0.095 | ≥0.85 |
| Correct token accuracy | 0.258 | 0.487 | ≥0.95 |
| effectful Wrong argument(exact/token) | 0.119 / 0.206 | 0.039 / 0.366 | Correctより有意に低いこと |
| None(exact/token) | 0.132 / 0.219 | 0.047 / 0.390 | ≤0.30(exact) |
| causal gap(exact match) | **0.039** | **0.049** | ≥0.50 |
| argument_effect_rate | 1.000 | 1.000 | — |

長さ別Correct exact match(平均):

| 入力長 | SHIFT | SELECT |
|---|---|---|
| 6 | 0.223 | 0.189 |
| 7 | 0.190 | 0.135 |
| 8 | 0.175 | 0.071 |
| 9 | 0.147 | 0.053 |
| 10 | 0.120 | 0.032 |

**含意:** BINDとは異なるパターン——causal gapは小さいが両演算とも全5seedで一貫して**正**(引数選択性はゼロではない)。主因は「マルチトークンexact matchの複合誤り」: Correct自身のtoken accuracyがすでに低く(SHIFT 0.258、SELECT 0.487)、6〜10位置すべてを同時に当てる必要があるexact matchはこの土台では0.85に届き得ない。長さが伸びるほどexact matchが単調に(SELECTでは急激に)低下することを確認。SELECTの学習は`itertools.combinations`による候補列挙のせいでSHIFTの約6倍遅い(計算コストの問題であり結果の正しさには影響しない)。

---

## 3. 総合考察(次の仮説検討のために)

### 3.1 排除できた候補説明

- **F1(Wrong-argument controlの非因果性)**: D-001で否定。4演算ともargument_effect_rateは高い(0.667〜1.000)。
- **F2(エンコーダ衝突)**: D-002でSELECTのみ確認、D-003で修正。SHIFT/COUNT/BINDは元から単射で問題なし。
- **F4(学習が引数使用を強制していない)単独説**: D-004(COUNT)・D-007(BIND)いずれも、引数多様性を学習データ中で最大化しても(argument_effect_rate=1.000)causal gapは改善しなかった(COUNT: 0.152→0.114、BIND: ほぼ0のまま)。F4は実在する要因の一つかもしれないが、単独の説明としては棄却されている。
- **「容量不足」単独説(COUNTのみ検証)**: D-005で4レバー(steps×2/×4, rank×2, arg_dim×2)すべて試したが、最良でもgap 0.222(要求の半分未満)。
- **「加算的conditioningが完全に壊れている」単純説**: D-006でV1(FiLM)が部分的に改善(gap 0.174)することを示し否定。ただし完全な解決にもならない。

### 3.2 演算ごとに異なる残された論点

- **COUNT**: 部分的な引数選択性が一貫して存在し、容量↑・formula変更(FiLM)のいずれも独立にgapを改善する(が不十分)。ADR-0035が示唆する未検証の組み合わせ: FiLM + rank_16(D-005のベスト容量発見)を同時に適用したらどうなるか、は未実行。
- **BIND**: causal gapが統計的にゼロ——COUNTより明確に深刻。しかもD-005/D-006のような容量/formula探索はBINDに対して一度も行われていない(D-007はD-004のベースラインをそのまま使っただけ)。ADR-0036は「位置を探して読む」連想検索という構造的に異なる計算であることを候補説明として挙げているが、これは未検証の仮説であり、FiLM/basisをBINDに適用した場合にCOUNTと同様の部分改善が見られるのか、それとも構造的に効かないのか、は不明。
- **SHIFT/SELECT**: causal gapは小さいが正——BINDのような「引数が全く効いていない」パターンではない。むしろCorrect自体の絶対性能(token accuracy)がすでに低く、マルチトークンの複合ペナルティが支配的。D-005/D-006の容量/formula探索はSHIFT/SELECTに対しても一度も行われていない。同じレバー(rank↑, FiLM)がSHIFT/SELECTの低いtoken accuracyの底上げに効くのかは未検証。

### 3.3 収束的な所見と、そこから生じる分岐点

D-004・D-006・D-007・D-008を通じて、**4つのパラメータ化演算すべてが共有ベースラインアーキテクチャでSTOP GATEに失敗**している。これは2つの読み方ができ、どちらが正しいかは今回の一連のタスクでは決着していない:

1. **アーキテクチャ全体(rank-8加算的残差 + frozen backbone)が、演算に関わらず構造的に不十分**という説。この場合、BIND向けの「位置を狙った注意機構」のような新しい primitive クラス設計が必要になる可能性がある。COUNTで得られたFiLMの部分改善や、SHIFT/SELECTの小さいが正のgapは、「完全に無力ではないが、根本的に非力」という一貫したパターンとして読める。
2. **演算ごとに別々のボトルネックがあり、単に探索が不完全なだけ**という説。COUNTだけがD-005/D-006の容量/formula探索を受けており、BIND/SHIFT/SELECTには一度も適用されていない。BIND/SHIFT/SELECTに同じ探索(容量↑、FiLM、あるいはFiLM+rank_16の組み合わせ)を行えば、少なくとも部分的な改善が得られる可能性は排除できていない。

D-008自身の「COUNT/BINDが強い結果にもかかわらず失敗した場合のみクロスアテンション代替を実装する」という条件は、今回はいずれも前提(強い結果)が成立しなかったため発動しなかった。ただし、上記1の読み方が正しい場合、この条件自体が「4演算のうち少なくとも1つは強いベースラインを持つ」ことを暗黙に仮定しており、その前提が今回の一連の結果全体では成立していない、という点は仮説検討の材料になり得る。

---

## 4. 現在のブロック状況

`AGENTS.md` の STOP GATE discipline により:

- **A1-R006 は引き続きブロック中**(ADR-0029以来変わらず)。
- A1-R005D-009(全4演算での最終パラメータ化primitive再試行)・A1-R005D-010(再試行監査レポート)はいずれも未着手。
- 本レポート自体は新規実験・実装を含まない(既存ADRの整理のみ)。

**関連ファイル:**
- ADR: `docs/DECISIONS.md` ADR-0030〜ADR-0037
- タスク仕様: `docs/CODEX_TASKS_A1_R005_RETRY.md`
- 実行アーティファクト: `runs/a1_r005d_001_reanalysis/`, `runs/a1_r005d_002_argument_encoder_audit/`, `runs/phase_a1_count_counterfactual_gate/`, `runs/phase_a1_count_capacity_sweep_gate/`, `runs/phase_a1_count_conditioning_architecture_gate/`, `runs/phase_a1_bind_counterfactual_gate/`, `runs/phase_a1_shift_counterfactual_gate/`, `runs/phase_a1_select_counterfactual_gate/`
- 実装: `apc.evaluation.parameterized_primitive_gate_reanalysis`, `apc.primitives.argument_encoder_audit`, `apc.primitives.conditioning`(`OrderPreservingIndexSetArgumentEncoder`/`ArgumentConditionedPrimitive`/`FiLMConditionedPrimitive`/`BasisModulatedConditionedPrimitive`), `apc.evaluation.count_counterfactual_gate`, `apc.evaluation.count_capacity_sweep_gate`, `apc.evaluation.count_conditioning_architecture_gate`, `apc.evaluation.bind_counterfactual_gate`, `apc.evaluation.sequence_counterfactual_gate`

---

## 5. クロージング(A1-R005E-001)

本セクションはA1-R005E-001(「R005再試行のクローズ」)の実施記録であり、上記1〜4節(既存内容)は変更していない。**新規の実験・実装はここにも含まれない。**

### 5.1 正式なクローズ判定

D-001〜D-008を**A1-R005再試行の最終診断結果として確定**する。SHIFT/SELECT/COUNT/BINDの4演算すべてが、共有ベースラインアーキテクチャ(V0 additive `ConditionedPrimitive`, rank=8, arg_dim=16)のもとでSTOP GATEに失敗するという収束的証拠が得られており、これ以上同じベースラインで診断を続けても新しい情報は得られない(5.2節)。

### 5.2 A1-R005D-009(全4演算最終再試行)を実行しない理由

`docs/CODEX_TASKS_A1_R005_RETRY.md`のA1-R005D-009は「D-001〜D-008で正当化された最小構成」を使って4演算を再試行するタスクとして定義されているが、D-001〜D-008の結果はその前提を満たしていない:

- D-009が要求する「正当化された最小構成」を得るための容量/formula探索(D-005/D-006)は**COUNTにしか行われていない**。BIND/SHIFT/SELECTには一度も適用されていない(3.2節)。
- COUNTについてさえ、D-005(容量スイープ)・D-006(conditioning architecture比較)は共にSTOP GATE未達(causal gap最良0.222、要求0.50)であり、「これなら通る」と言える構成は存在しない。
- したがって仮にD-009を今実行しても、それは「D-004/D-006/D-007/D-008で個別に既に検証済みの同一ベースライン(V0 additive, rank=8, arg_dim=16)を4演算まとめて再実行するだけ」になり、新しい仮説やアーキテクチャ変更を含まない。結果はD-004/D-006/D-007/D-008の結果から容易に予測でき(4演算いずれも単独でFAILしているものが、まとめて実行して急にPASSする理由がない)、STOP GATE FAILを追加で1回記録する以上の診断的価値を持たない。
- 一方でD-009には非自明な実行コスト(5 seeds×4演算の学習)がかかる。「同じ結果が予測できる実験を新しい仮説なしに繰り返す」ことは`AGENTS.md`のSTOP GATE disciplineが禁じる「negative resultを隠すためのスケール増加」とは異なるが、それと対称的に「negative resultを追認するためだけの計算コスト」でもあり、正当化されない。

よって**A1-R005D-009は意図的にskip/supersededとする**。D-009は実行済みとして報告しない。

### 5.3 新しい診断フェーズへの移行

D-001〜D-008が排除・支持した候補説明(3.1〜3.3節)はいずれも「同じアーキテクチャ内でのパラメータ/formula調整」の範囲にとどまり、根本的な問い ── frozen task-blind `h_content` が引数条件付き計算に十分な情報を保持しているか、それともprimitive/operator自体がボトルネックか、あるいは表現そのものの再設計が必要か ── には答えていない。この問いを切り分けるため、後続フェーズを`docs/exec-plans/active/A1_R005E_DIAGNOSTIC.md`(A1-R005E, Representation / Operator Isolation)として新設し、D-001〜D-008の結果を出発点として引き継ぐ。

**ポインタ更新:**
- `docs/CODEX_TASKS_A1_R005_RETRY.md`: D-009 skip/supersededの注記を追加、A1-R005E-001への導線を追加。
- `docs/exec-plans/active/A1_R005_RETRY.md`: ステータスを `closed — negative diagnostic result after D-001 through D-008` に更新。
- `docs/exec-plans/active/PHASE_A1_POST_CORRECTION.md`: 「A1-R006 remains blocked / Active work is A1_R005E_DIAGNOSTIC.md」の注記を追加。
- `AGENTS.md`: Active research phaseセクションに、A1-R006着手前に読むべき診断キュー(A1-R005E一式)へのポインタを追加。

### 5.4 A1-R005E-001 受け入れ基準チェック

- history intact: D-001〜D-008の測定値・結論(1〜4節)は変更していない。D-009/D-010のタスク定義(`docs/CODEX_TASKS_A1_R005_RETRY.md`)もテキストとして保持。
- D-009 not reported as executed: 5.2節の通り、D-009は未実行のままskip/supersededとして記録。
- A1-R006 remains blocked: 5.3節のポインタ更新4箇所すべてで再確認済み。
- new diagnostic phase active: `docs/exec-plans/active/A1_R005E_DIAGNOSTIC.md` / `docs/CODEX_TASKS_A1_R005E_DIAGNOSTIC.md` が active として`AGENTS.md`・`PHASE_A1_POST_CORRECTION.md`から参照されている。
