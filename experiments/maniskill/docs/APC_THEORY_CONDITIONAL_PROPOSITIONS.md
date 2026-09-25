# APC 条件付き命題の定式化と導出（W8: T23）

作成日：2026-09-25（JST）  
対象タスク：W8 / T23（三つの条件付き命題の定式化、仮定、導出、破綻条件の同定）  
対象命題：H1（局所獲得）、H2（適用可能性）、H3（行動定着）、H4（実解放）、H6（合成転移）、H7（継続保持）、H8（資源利得）

---

## 1. 理論化の背景と前提

Adaptive Primitive Consolidation (APC) のこれまでの物理実証（W0〜W7）において、局所小型決定木（CART）による補正、一時資源の実解放（35.8 KB配布物）、同一ルーターでの因果的アブレーション、および実測多タスク列での破滅的忘却ゼロ（$F=0.0$）が確認された。
一方、未学習の極端な目標座標（Seed 3014）におけるゼロショット合成の破綻や、手設計回復ガードへの依存など、明確な適用限界も実測同定された。

本稿（W8 / T23）の目的は、単なる実験結果の記述にとどまらず、**「どのような前提条件（Assumptions）のもとで APC が成立し、どのような境界で破綻するか」を予測可能な数学的・概念的命題として定式化**することである。

---

## 2. 命題 1：非介入領域における行動不変性と忘却ゼロ（Invariance Under Local Non-Intervention）

### 2.1 前提仮定（Assumptions）
- **仮定 1.1（基盤方策の凍結）**: 基盤方策 $\pi_{\text{base}}$ の重み・パラメータ、内部状態遷移、および観測特徴量変換 $\phi(s)$ は、新能力追加時にも完全に不変（Frozen）である。
- **仮定 1.2（決定論的遷移と再現性）**: 物理環境は同一初期状態 $s_0 \sim \mathcal{D}_{\text{past}}$ および同一乱数シードのもとで決定論的（または統計的に同一）に遷移する。
- **仮定 1.3（誤介入ゼロ / Zero False Positive）**: 過去タスク $\mathcal{T}_{\text{past}}$ の誘発する軌跡分布 $\mathcal{D}_{\text{past}}$ 上において、学習済みルーター $R(s)$ が新モジュール $\pi_{\text{new}}$ を選択する確率が 0 である：
  $$P_{s \sim \mathcal{D}_{\text{past}}}(R(s) = \text{new}) = 0$$

### 2.2 命題の主張（Proposition 1）
上記の仮定 1.1〜1.3 が満たされるならば、新モジュール $\pi_{\text{new}}$ を統合した合成方策 $\pi_{\text{comp}}$ を過去タスク $\mathcal{T}_{\text{past}}$ に適用したときの実行軌跡 $\tau_{\text{comp}} = (s_0, a_0, s_1, a_1, \dots, s_H)$ は、更新前の基盤方策による実行軌跡 $\tau_{\text{base}}$ と完全に一致し、**破滅的忘却率は厳密に 0** となる：
$$\tau_{\text{comp}} \equiv \tau_{\text{base}} \implies \text{SuccessRate}(\pi_{\text{comp}}, \mathcal{T}_{\text{past}}) = \text{SuccessRate}(\pi_{\text{base}}, \mathcal{T}_{\text{past}})$$

### 2.3 物理的検証との対応と破綻条件
- **実証された裏付け**: W7（T20–T22）において、新モジュール（Transit）獲得後も、過去タスクである Pick（744 steps）および Place（833 steps）が全く同一のステップ数・同一の物理挙動で 100% 成功し、実測忘却率 $F = 0.0$ が達成された。
- **破綻条件（Failure Condition）**:
  - ルーターの過剰汎化により、過去タスクの正常領域で新モジュールが誤発動した場合（False Positive $> 0$）。
  - または、特徴変換 $\phi(s)$ の共有部分が更新され、Base への入力が変質した場合。

---

## 3. 命題 2：局所合成による蒸留誤差伝播の縮小上界（Bounded Error Propagation Under Local Composition）

### 3.1 前提仮定（Assumptions）
- **仮定 2.1（有限ホライズン）**: タスクは最大 $H$ ステップで終了する。
- **仮定 2.2（局所介入率）**: 合成方策の実行中、新モジュールが有効化される確率（介入率）を $\rho_t = P(R(s_t) = \text{new})$ とする。APC の局所適応においては、タスク全体の大部分が Base で実行され、局所区間のみ介入するため、平均介入率 $\bar{\rho} = \frac{1}{H} \sum_{t=1}^H \rho_t \ll 1$ である。
- **仮定 2.3（局所判断不一致確率）**: 介入区間（$R(s_t) = \text{new}$）において、軽量 Candidate CART $\pi_C$ が教師/Temporary $\pi_T$ と異なる行動を出力する条件付き確率（局所蒸留誤差）を $\epsilon_t = P(\pi_C(s_t) \neq \pi_T(s_t) \mid R(s_t) = \text{new})$ とする。

### 3.2 命題の主張（Proposition 2）
同一の Base 方策・同一ルーターのもとで、Temporary 合成方策 $\pi_{\text{comp}}^T$ と Candidate 合成方策 $\pi_{\text{comp}}^C$ の物理実行軌跡が、時刻 $H$ までに少なくとも 1 回分岐する確率 $P(\tau_C \neq \tau_T)$ には、以下の累積上界が存在する：
$$P(\tau_C \neq \tau_T) \le \min\left(1, \; \sum_{t=1}^H \rho_t \cdot \epsilon_t\right)$$

したがって、両方策のタスク達成成功率の差の絶対値も、この上界によって抑えられる：
$$|\text{Success}(\pi_{\text{comp}}^C) - \text{Success}(\pi_{\text{comp}}^T)| \le \sum_{t=1}^H \rho_t \cdot \epsilon_t$$

### 3.3 理論的インプリケーションと直接Candidate方式の成立
- 全体方策を単一の決定木へ蒸留する場合、全ステップで介入するため $\rho_t \equiv 1$ となり、累積誤差は $\sum \epsilon_t$ に達する。
- 対照的に、APC の局所モジュールでは必要な局面（例: 運搬中の水平補正 19 steps / 844 steps $\approx 2.2\%$）のみ介入するため、**有効介入率 $\rho_t$ が極めて小さく、蒸留誤差の伝播が構造的に圧縮される**。
- これにより、W5（T16）で確認されたように、Temporary MLP を介さず直接 Candidate CART を fit しても（直接Candidate方式）、決定木の表現容量不足による破綻を起こさず、高精度な安定成功が達成できる。

---

## 4. 命題 3：再出現頻度と定着費用の損益分岐点（Breakeven Analysis of Consolidation vs Relearning）

### 4.1 費用モデルの定義
ある能力要求（Capability Requirement）が長期ライフサイクルにおいて $n$ 回出現する環境を考える。
- **APC（定着・再利用型）の総費用**:
  - 初回遭遇時の適応費用: $C_{\text{acq}}$（不足プローブ探索 ＋ データ収集）
  - 定着・統合費用: $C_{\text{cons}}$（Candidate fit ＋ ルーター再学習）
  - 各再出現時の推論・適用費用: $C_{\text{use}}$
  $$C_{\text{APC}}(n) = C_{\text{acq}} + C_{\text{cons}} + n \cdot C_{\text{use}}$$
- **毎回再学習方式（Relearning / Scratch Baseline）の総費用**:
  - 出現するたびに再探索・再学習を行うため、1 回あたりの費用を $C_{\text{relearn}}$ とすると:
  $$C_{\text{scratch}}(n) = n \cdot C_{\text{relearn}}$$
- **共有モデル＋リプレイ方式（Shared Model with Replay）の総費用**:
  - 新タスクの勾配更新 $C_{\text{grad}}$ に加え、過去タスク忘却を防ぐためのリプレイバッファ再生計算 $C_{\text{replay}}$ が毎回発生する：
  $$C_{\text{replay}}(n) = C_{\text{init}} + n \cdot (C_{\text{grad}} + C_{\text{replay}})$$

### 4.2 命題の主張（Proposition 3）
APC が毎回再学習方式に対して累積費用上の優位性を持つための必要十分条件は、タスク再出現回数 $n$ が以下の損益分岐回数 $n^*$ を超えることである：
$$n > n^* = \frac{C_{\text{acq}} + C_{\text{cons}}}{C_{\text{relearn}} - C_{\text{use}}}$$

### 4.3 実測値による損益分岐点の計算
W7 実測ベンチマーク（`runs/apc-t20-lifelong-empirical-20260925-b`）の実測値を代入：
- $C_{\text{acq}} + C_{\text{cons}} = 5.50\,\text{s}$（Block 3 でのプローブ ＋ ルーター CART fit 時間）
- $C_{\text{use}} = 0.0\,\text{s}$（再出現時の追加学習費用は 0 秒）
- 毎回再学習方式が一回あたり同等の適応費用を要すると仮定（$C_{\text{relearn}} \approx 5.50\,\text{s}$）：
$$n^* = \frac{5.50\,\text{s}}{5.50\,\text{s} - 0.0\,\text{s}} = 1.0$$
- **理論的帰結**: **2 回目の再遭遇（$n=2$）の時点で APC の累積学習費用は再学習方式を下回り、以降の再遭遇ごとに利得が線形に拡大する**。

---

## 5. まとめ

定式化された三つの条件付き命題により、APC の中核的挙動は以下の数学的前提と結び付けられた：
1. **忘却ゼロの条件**: ルーターの過去分布上での False Positive が 0 であること（命題 1）。
2. **小型決定木で安定する条件**: 局所介入率 $\rho_t \ll 1$ により誤差累積が抑制されること（命題 2）。
3. **資源利得が生じる条件**: 再出現回数 $n > 1$ において定着費用の償却が成立すること（命題 3）。
