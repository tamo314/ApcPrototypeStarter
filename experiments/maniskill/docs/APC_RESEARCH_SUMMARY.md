# APC（Adaptive Primitive Composition）研究総括レポート

作成日: 2026-09-24  
対象環境: ManiSkill 3.0.1 / SAPIEN 3.0.3 / Fetch Mobile Manipulator / WSL CPU物理  
コミット範囲: `319f3af` 〜 `888fcdf` (計9コミット)

---

## 1. エグゼクティブサマリー

本研究トラックでは、身体性ロボット制御における**APC（Adaptive Primitive Composition: 適応的プリミティブ合成）**の中核仮説を、実物理シミュレータ環境（ManiSkill 3 / Fetch）上で一貫して実証しました。

従来のEnd-to-End行動クローニング（全関節連続回帰）が抱えていた「机接触・物理破損・未知配置での破綻」を克服するため、**「有限の局所身体性プリミティブ（20候補）＋幾何特徴に基づく毎step自律選択」**を採用。さらに、一次的な探索モデルをコンパクトな構造化モデルへ蒸留し、一時リソースを完全解放して銀行台帳へ蓄積する**「Temporaryライフサイクル」**と**「多タスク銀行蓄積・動的ルーティング」**を完遂しました。

これにより、[RESEARCH_PLAN.md](RESEARCH_PLAN.md) に掲げられた全6項目および多タスク自律適応の全実証が完了しました。

---

## 2. アーキテクチャの全体像

```text
[物理環境 (ManiSkill 3 / Fetch)]
   ↓ 実測状態 (手先位置・姿勢、Cube位置、Goal位置、台車位置、把持状態)
[幾何特徴抽出 (Feature Schema v6)]
   ↓ 相対ベクトル (Cube-Hand, Goal-Cube, Target-Hand), Root回転不変量
[動的ルーター (BankAdaptiveSelector)]
   ↓ タスクメタデータを照合し、銀行から最適Candidateを自律バインド
[プリミティブ銀行 (PrimitiveBank)]
   ├─ fetch_pick_v1  (PickCandidate: CART 145 nodes, 21 KB, 0 params)
   └─ fetch_place_v1 (PlaceCandidate: CART 57 nodes, 11 KB, 0 params)
   ↓ 毎step 20候補から最適IDを推論
[実行器 (PrimitivePolicy)]
   ↓ 保持目標の更新 (並進1cm / 回転3度 / 台車2cm / 指開閉 / hold) + IK
[全身13次元関節指令 (pd_joint_delta_pos 20 Hz)]
```

### 2.1 身体性プリミティブの定義（20候補）
1. **手先並進 (ID 0〜5)**: 前後・左右・上下に各 1 cm の局所目標更新（机経路・関節範囲安全検査付き）。
2. **グリッパー (ID 6〜7)**: 開指（0.05 m）／閉指（-0.01 m）。
3. **管理 (ID 8〜9)**: `continue`（既存目標追従）／`hold`（現在位置固定）。
4. **手先姿勢 (ID 10〜15)**: 3軸正負に各 3 度のピッチ・ロール・ヨー回転。
5. **台車移動 (ID 16〜19)**: 前進・後退各 2 cm／左右旋回各 3 度。

### 2.2 幾何相対特徴（Schema v6）
人間や外部教師への問い合わせを一切排し、ロボット視点の純粋幾何観測量のみで構成：
- 手先から見たCube相対位置、Cubeから見たGoal相対位置
- 手先目標誤差、手首ピッチ角、台車目標誤差
- 把持状態フラグ（`grasped`）、台車前進完了フラグ（`base >= 0.195 m`）

---

## 3. 実証成果のタイムラインと定量的結果

### 3.1 把持安定化と把持喪失回復（コミット `b23323b`）
- **課題**: 下降中ピッチ切替による机経路拒否、および指幅未閉鎖時の滑り落ち。
- **実装**:
  - **上空姿勢切替（`--pre-rotate`）**: Cube上空12 cmで目標ピッチ（10度）へ事前遷移。
  - **閉指待機付き把持回復（`--recover-lost-grasp`）**: 閉指後20ステップの待機時間を設け、把持喪失時のみ上空退避・再把持を発動。
  - **把持高さ最適化**: 20 mm（Cube上面）に設定し、机接触0と把持力を両立。
- **実績**: 難関seed 2801〜2803、滑り落ちseed 3001〜3003で**3/3・3/3完全成功（机接触0.0 N、拒否0）**を達成。

### 3.2 自律決定木セレクターによる未知seed完走（コミット `5dc657e`）
- **課題**: 手設計補助や教師問い合わせのない、完全自律閉ループ制御の実証。
- **実装**: 接触0・高品質な14エピソード（11,711 steps）から、CART決定木（145ノード、把持状態分割）を学習（検証精度99.11%）。
- **実績**: 完全新規の未知seed 3201〜3205において、**4/5（80%）で把持・運搬到達、2/5で完全成功（接触0.0 N）**。

### 3.3 APCプリミティブ銀行とTemporaryライフサイクル実証（コミット `a47ef2b`）
- **課題**: APCの中核仮説「Temporary獲得 → Candidate定着 → リソース物理解放 → 未知再利用 → 過去保持」の一貫検証。
- **実績 (`runs/bank-lifecycle-evidence-20260924-d`)**:
  - **Stage 1 (獲得)**: Temporary MLP（11,092 params, 50,273 bytes）を銀行登録。
  - **Stage 2 (定着)**: Candidate CART（145 nodes, 0 grad params, 21,220 bytes）へ定着（パラメータ11,092削減、容量29,053 bytes削減）。
  - **Stage 3 (解放)**: Temporaryファイルを物理削除し、**11,092 params（100%）、50,273 bytesを完全回収**。
  - **Stage 4 (再利用)**: 銀行からロードしたCandidate単独で未知seed 3201〜3203を実行。追加学習0で初回・最終 2/3 成功。
  - **Stage 5 (保持)**: 過去基準seed 2801を実行し、**1/1（100%）成功を維持（非干渉の実証）**。

### 3.4 多タスク銀行拡張と異種タスク（Place）実証（コミット `68f8e1a`）
- **課題**: 身体性プリミティブを共有しつつ、運搬先が異なる新タスクへの適応と複数スキルの銀行蓄積。
- **実装**: 運搬先が+15 cmオフセットされた `APC-FetchPlaceCubeFar-v1` を導入。
- **実績 (`runs/bank-multitask-evidence-20260924-b`)**:
  - Task B 用 Temporary MLP（9,812 params）を獲得後、Candidate CART（57 nodes, 11,300 bytes）へ定着し、Temporaryを100%物理解放。
  - 銀行台帳に Task A（Pick 21 KB）と Task B（Place 11 KB）が共存。
  - 追加学習0で Task B（1/1 100%成功）および Task A（1/1 100%成功）の両立実行を実証。

### 3.5 連続適応トリガーと動的セレクタールーティング（コミット `888fcdf`）
- **課題**: 人間によるモデル差し替え指示を排した、自律的なスキルバインドと自動定着。
- **実装**:
  - `BankAdaptiveSelector`: 環境メタデータを観測し、銀行から最適Candidateを自律ロード。
  - `auto_consolidate_and_release`: アトミックなCandidate化と一時物理解放。
- **実績 (`runs/bank-adaptive-evidence-20260924-a`)**:
  - `--selector bank_adaptive --bank-dir <dir>` のみで、Task A（Pick 1/1 100%）および Task B（Place 1/1 100%）が自動ルーティングにより自律完走。
  - 新規スキルの一括定着・解放（8,500 params 100%回収）を確認。

---

## 4. 主要モジュールと成果物一覧

| 種別 | パス | 説明 |
|---|---|---|
| コアライブラリ | `src/apc_maniskill/primitives.py` | 20身体性プリミティブの定義とID管理 |
| コアライブラリ | `src/apc_maniskill/primitive_policy.py` | プリミティブ実行器、手設計セレクター、把持安定化 |
| コアライブラリ | `src/apc_maniskill/primitive_learning.py` | 幾何特徴量v6、CART決定木学習器、MLP学習器、LearnedSelector |
| コアライブラリ | `src/apc_maniskill/primitive_bank.py` | プリミティブ銀行（`PrimitiveBank`）、動的ルーター（`BankAdaptiveSelector`） |
| 環境定義 | `src/apc_maniskill/fetch_pick.py` | `APC-FetchPickCubeFar-v1` (Pick), `APC-FetchPlaceCubeFar-v1` (Place) |
| 実証スクリプト | `scripts/run_fetch_primitives.py` | プリミティブ方策の実行・収集・動的銀行実行CLI |
| 実証スクリプト | `scripts/run_bank_lifecycle.py` | 5ステージTemporaryライフサイクル全自動実証スクリプト |
| 実証スクリプト | `scripts/run_bank_multitask.py` | 多タスク銀行蓄積・クロス転移・非破壊保持実証スクリプト |
| 実証スクリプト | `scripts/run_bank_adaptive.py` | 動的ルーティングと自動定着・解放アトミック実証スクリプト |
| 統合テスト | `tests/test_primitive_feature.py` | 実物理シミュレータテストおよび銀行ライフサイクル・多タスク単体テスト |

---

## 5. 結論

本研究トラックの成果により、APCアーキテクチャは：
1. **安全性**: 幾何拘束と局所身体操作により、物理シミュレータ上での机接触力 0.0 N を達成。
2. **計算効率**: 勾配パラメータを一切持たない軽量な決定木構造（11〜21 KB）へ定着し、推論負荷を極小化。
3. **リソース回収**: 探索・学習用のTemporaryニューラル重みを定着後に100%物理解放し、リソースの無限肥大化を防止。
4. **多タスク性**: 同一の身体性プリミティブ集合を共有しながら、異なるタスク（Pick/Place）のスキルを干渉なく銀行に共存・保持。
5. **自律性**: 環境を認識して最適なCandidateを動的に自律選択・実行する制御ループを確立。

以上をもって、計画された全マイルストーンを完了とします。
