# APC Prototype Implementation Plan

このファイルはプロジェクト所有者向けの実装計画サマリーです。Codexが実際に作業するときの詳細な受け入れ条件は `docs/exec-plans/active/PHASE_A.md` を正とします。

## 目的

最初のプロトタイプで検証するのはLLM性能ではなく、次の循環学習が安定して成立するかです。

```text
Sparse Stable System
  -> Search / Recomposition
  -> Plastic Resource Allocation
  -> Fast Learning
  -> Consolidation
  -> Shadow Validation
  -> Resource Release
  -> Sparse Stable System
```

## 開発原則

- 小さなモデルから開始する。
- 未知の「入力」と未知の「計算」を区別する。
- 既存primitiveの未知compositionでは原則としてcapacityを増やさない。
- 真に新しいoperationでのみtemporary capacityを追加する。
- temporary capacityをそのまま永続化しない。
- consolidation後はshadow validationに合格するまでtemporary capacityを保持する。
- 性能だけでなくresident parameters、temporary peak、active parameters、VRAM、forgetting、reuseを同時に測定する。

## 実装順序

### Step 0 — Repository bootstrap

Python/PyTorchプロジェクト、テスト、設定、ログ、seed管理を作る。

### Step 1 — Synthetic task universe

COPY / SELECT / COMPARE / COUNT / SHIFT / BIND / NEGATE / ACCUMULATE等を組み合わせられるsymbolic environmentとreference interpreterを実装する。

重要なのはデータ生成側が、

- Known task
- Novel composition
- Novel operation
- Recurrence

をground truthとして区別できること。ただしこのoracle情報はMeta Controllerには渡さない。

### Step 2 — Fixed dense baseline

小型Transformerを学習し、以降の比較基準を確立する。

### Step 3 — Stable Core + Primitive Bank

低ランクresidual transformをprimitiveとして実装し、top-k Routerで疎に選択する。

まず固定bankで既知タスクとheld-out compositionを評価する。

### Step 4 — Plastic Workspace

temporary low-rank transformsを追加できる仕組みを実装する。Stable Coreとpersistent primitivesを凍結したまま新能力を学習可能にする。

### Step 5 — Novelty + finite-state Meta Controller

最初はRLを使わず、error、uncertainty、router entropy等を記録し、STABLE / SEARCH / PLASTIC / CONSOLIDATE / SHADOWをrule-basedで遷移させる。

### Step 6 — Consolidation

temporary transformsの活動を収集し、不要部分を削除して、より低ランク・少数のcandidate primitivesへdistillする。

### Step 7 — Shadow Validation + Release

temporary solutionとcandidate solutionを並列評価し、current taskと過去taskの双方で基準を満たした場合だけcandidateをpersistent bankに昇格しtemporary resourceを解放する。

### Step 8 — Sequential lifelong benchmark

Novel operationを少なくとも2回学習させ、学習→圧縮→解放の循環を複数回実行する。以前学んだoperationが再登場した際に再学習せずreuseできることを確認する。

### Step 9 — Baselines / Ablations

固定Dense、固定Sparse、Grow-only、Grow+Replay、APC full loopを比較する。SEARCH、Replay、Shadow Validation等を外したablationも実施する。

## Phase Aの成功条件

最低限、以下を同時に満たすことを目標とする。

1. Novel operationでtemporary capacityを追加できる。
2. Novel compositionでは多くの場合capacity追加なしで対処できる。
3. consolidation後の性能がtemporary solutionの95%以上を保持する。
4. 過去taskの低下を初期目標2 percentage points以内に抑える。
5. temporary capacityの大部分を解放できる。
6. 学習したprimitiveを後のtaskで再利用できる。
7. resident parameter growthがGrow-only baselineより明確に小さい。
8. 同等性能のbaselineに対してlifetime computeまたはactive computeに優位性の兆候がある。

これらの閾値は研究上の初期仮説であり、測定結果に応じてDecision Logへ理由を残したうえで変更する。

## 5060 Ti 16GB向けスケール

Phase Aは10M〜60M程度のStable Coreから開始する。必要があれば100M級まで拡張するが、最初から大きくしない。

Plastic Workspaceも数M〜数十Mparameter単位で段階的に追加する。循環学習そのものが成立してから数億parameter級へ進む。

自然言語モデルを使うPhase Dでは、0.5B〜2B pretrained model + LoRA/QLoRAを基本とし、full pretraining/full fine-tuningを前提としない。

## Codexとの作業方法

`docs/CODEX_TASKS.md` のTask 001から一つずつ進める。

最初のCodex指示例:

```text
Read AGENTS.md and every document listed under "Read first". Implement only Task 001 from docs/CODEX_TASKS.md. Do not start Task 002. Run all verification commands that are applicable, and report changed files, tests run, assumptions, and limitations.
```

各task完了後に人間が差分・tests・architecture driftを確認してから次へ進む。
