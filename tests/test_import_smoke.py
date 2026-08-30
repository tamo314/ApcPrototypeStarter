"""CI-friendly CPU smoke test: every apc package/subpackage imports cleanly."""

import importlib

SUBPACKAGES = [
    "apc",
    "apc.core",
    "apc.primitives",
    "apc.plastic",
    "apc.consolidation",
    "apc.meta",
    "apc.environments",
    "apc.evaluation",
    "apc.utils",
    "apc.utils.config",
    "apc.utils.seed",
    "apc.utils.system_info",
]


def test_all_subpackages_import() -> None:
    for module_name in SUBPACKAGES:
        importlib.import_module(module_name)


def test_version_is_defined() -> None:
    import apc

    assert isinstance(apc.__version__, str)
    assert apc.__version__
