"""System metadata logger tests. Must pass with or without CUDA present."""

from apc.utils.system_info import get_system_info


def test_get_system_info_has_required_keys() -> None:
    info = get_system_info(seed=123)

    for key in (
        "seed",
        "git_commit",
        "platform",
        "python_version",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "device_name",
        "peak_vram_bytes",
    ):
        assert key in info

    assert info["seed"] == 123
    assert isinstance(info["cuda_available"], bool)


def test_cuda_fields_consistent_with_availability() -> None:
    info = get_system_info()

    if info["cuda_available"]:
        assert info["device_name"] is not None
    else:
        assert info["device_name"] is None
        assert info["cuda_version"] is None
        assert info["peak_vram_bytes"] is None
