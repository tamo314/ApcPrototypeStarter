# SciPy Dependency Block — Source Notes

REC-004K / ADR-0106は、task-specific 31 tests PASSに対しfull repoで28 failuresを報告した。共通原因は`ModuleNotFoundError: No module named scipy`で、`functional_metrics_v2.py`を経由するとされた。前のADR-0105環境ではfull suiteがgreenだったため、SciPyが暗黙に環境へ存在した可能性がある。

このpackではSciPyをcore/dev/optionalのどこへ宣言するかを事前固定しない。live import graphと現repoのpackaging policyをStage Aで確認する。
