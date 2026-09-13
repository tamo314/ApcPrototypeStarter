"""Seed utility reproducibility tests. CPU-only; CUDA not required."""

import torch

from apc.utils.seed import set_seed


def test_set_seed_reproduces_torch_random() -> None:
    set_seed(1234)
    a = torch.rand(8)
    set_seed(1234)
    b = torch.rand(8)
    assert torch.equal(a, b)


def test_set_seed_reproduces_python_random() -> None:
    import random

    set_seed(7)
    a = [random.random() for _ in range(5)]
    set_seed(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_different_seeds_diverge() -> None:
    set_seed(1)
    a = torch.rand(8)
    set_seed(2)
    b = torch.rand(8)
    assert not torch.equal(a, b)


def test_set_seed_bounds_pythonhashseed() -> None:
    import os

    # 63-bit integer exceeds 32-bit unsigned int
    large_seed = (1 << 63) - 1
    set_seed(large_seed)
    hash_seed = int(os.environ["PYTHONHASHSEED"])
    assert 0 <= hash_seed <= 4294967295
    assert hash_seed == (large_seed & 0xFFFFFFFF)

