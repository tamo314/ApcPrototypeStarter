# 開発開始ガイド

## このPCで可能か

可能です。初期プロトタイプはむしろ、この16GB VRAM制約を前提に小さく設計した方が研究上の切り分けが容易です。

最初の目的は大規模LLMを学習することではなく、以下の循環を成立させることです。

```text
既知能力を疎に実行
  ↓
未知を検知
  ↓
一時リソースを追加
  ↓
新能力を学習
  ↓
小さなprimitiveへ圧縮
  ↓
検証
  ↓
一時リソースを解放
  ↓
以後はprimitiveを再利用
```

30M〜100M級のPhase AであればRTX 5060 Ti 16GBで十分です。数億parameter級は設定を選びますが実験可能です。0.5B〜2B級の既存言語モデルへの移植は、後段でLoRA/QLoRAを使う前提にします。

## Codexで最初に行うこと

リポジトリのルートにこのスターターパックを置き、Codexへ次のように依頼します。

```text
Read AGENTS.md and all documents it marks as required. Then implement Task 001 from docs/CODEX_TASKS.md. Do not start Task 002. Run the required verification commands and summarize changes, tests, and any assumptions.
```

Task 001が完了したらレビューし、その後Task 002を同様に依頼します。

一度にPhase A全体を実装させるより、issueサイズで分割した方が設計逸脱や検証漏れを見つけやすくなります。

## 推奨の進め方

1. `AGENTS.md` と `docs/design-docs/ARCHITECTURE.md` を人間側でも確認する。
2. CodexにTask 001を実装させる。
3. テスト結果を確認しコミットする。
4. Task 002〜005まで進め、まず固定primitive modelを成立させる。
5. ここで最初のベースラインを保存する。
6. その後にPlastic Workspace、Novelty、Consolidationを一つずつ追加する。
7. 各段階で固定baselineとの差を記録する。

## 重要な方針

失敗をモデルサイズで隠さないでください。

たとえばConsolidationがうまくいかないときに、すぐStable Coreを10倍にすると、「回路圧縮の問題」なのか「容量不足」なのか分からなくなります。

Phase Aでは小さなモデルのまま、どの仮説が失敗したかを切り分けることを最優先にします。
