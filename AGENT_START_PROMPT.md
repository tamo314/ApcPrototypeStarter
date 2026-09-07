# AIコーディングエージェントへの開始指示

**B-C005REC-001のみ**を実施してください。

このパックを既存repoのドキュメント構成に沿って導入し、以下を順に読んでください。

- `docs/exec-plans/active/PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`
- `docs/CODEX_TASKS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`
- `docs/EXPERIMENT_PLAN_PHASE_B_B2_MODEL_BUNDLE_RECOVERY.md`
- `docs/design-docs/B2_MODEL_BUNDLE_RECOVERY_CONTRACT.md`
- `docs/AGENTS_PHASE_B_B2_MODEL_BUNDLE_RECOVERY_ADDENDUM.md`
- `docs/research/B2_MODEL_BUNDLE_RECOVERY_SOURCE_NOTES.md`

実施範囲は、既存artifactの保存・台帳化、Core／bank／router／scorerの依存関係の確認、全16 primitiveの生成経路の確認、復元可能性の分類までです。対象moduleの実在と契約はコードで確認してください。

`get_or_build_16_primitive_bank`等をそのまま呼んで再構築を始めないでください。読むつもりの呼出しに学習・cache書込み・暗黙初期化がないか先に調べてください。

共有キャッシュの削除／上書き、重み更新、他タスクの自動実行、sealed出力の測定は禁止です。過去Coreを見つけても、来歴と機能確認なしに正常な復元先とみなさないでください。

完了報告には、復元できる組合せ、出自不明・欠落artifact、全16個のbuild coverage、想定する復旧経路、変更ファイル、検査結果、ADR、block状態を含めてください。復元不能は正直に`REBUILD_REQUIRED`と記録し、不明な来歴を補完しないでください。

**REC-001が完了した時点でSTOP。REC-002以降やR3-011へは進まないでください。**
