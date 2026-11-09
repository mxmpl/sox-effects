import functools
import gc
import operator
import random
import shutil
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal
from torch.utils.data import DataLoader, Dataset

import sox_effects as se
from sox_effects import EffectChain

jax.config.update("jax_platforms", "cpu")

SR = 16000
BLOCKED = ["input", "output", "spectrogram", "noiseprof", "noisered", "splice"]
HEAVY = [["pitch", "200"], ["rate", str(SR)], ["reverb", "50"], ["channels", "1"]]


@dataclass(frozen=True)
class Lib:
    """An array library: how to build its arrays from NumPy, and what type it returns."""

    name: str
    asarray: Callable[[np.ndarray], Any]
    array_type: type


LIBS = [
    Lib("numpy", np.asarray, np.ndarray),
    Lib("torch", torch.from_numpy, torch.Tensor),
    Lib("jax", jnp.asarray, jax.Array),
]


@pytest.fixture(params=LIBS, ids=lambda lib: lib.name)
def lib(request: pytest.FixtureRequest) -> Lib:
    return request.param


@pytest.fixture
def x() -> np.ndarray:
    t = np.arange(SR, dtype=np.float32) / SR
    return np.stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.25 * np.sin(2 * np.pi * 220 * t)]).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Every library
# --------------------------------------------------------------------------------------------


def test_identity(lib: Lib, x: np.ndarray) -> None:
    y, sr = se.apply_effects(lib.asarray(x), SR, [])
    assert sr == SR
    assert isinstance(y, lib.array_type)
    assert np.asarray(y).dtype == np.float32
    assert_allclose(np.asarray(y), x, atol=1e-7, rtol=0)


@pytest.mark.parametrize(
    ("effects", "shape", "rate"),
    [
        ([["rate", "8000"]], (2, 8000), 8000),
        ([["speed", "1.5"], ["rate", str(SR)]], (2, 10667), SR),
        ([["channels", "1"]], (1, SR), SR),
        ([["reverb", "50"]], (2, SR), SR),
    ],
)
def test_shape_and_rate(lib: Lib, x: np.ndarray, effects: list[list[str]], shape: tuple[int, int], rate: int) -> None:
    y, sr = se.apply_effects(lib.asarray(x), SR, effects)
    assert (tuple(y.shape), sr) == (shape, rate)


def test_same_result_as_numpy(lib: Lib, x: np.ndarray) -> None:
    ref, _ = se.apply_effects(x, SR, HEAVY)
    y, _ = se.apply_effects(lib.asarray(x), SR, HEAVY)
    assert_array_equal(np.asarray(y), ref)


def test_layouts_and_strides(lib: Lib, x: np.ndarray) -> None:
    a = lib.asarray(x)
    ref, _ = se.apply_effects(x, SR, HEAVY)
    last, _ = se.apply_effects(a.T, SR, HEAVY, channels_first=False)  # a view for numpy and torch
    assert isinstance(last, lib.array_type)
    assert_array_equal(np.asarray(last), ref.T)
    half, _ = se.apply_effects(a[:, ::2], SR, [["vol", "0.5"]])
    assert_allclose(np.asarray(half), 0.5 * x[:, ::2], atol=1e-7, rtol=0)


def test_empty(lib: Lib) -> None:
    y, sr = se.apply_effects(lib.asarray(np.zeros((2, 0), np.float32)), SR, [["rate", "8000"]])
    assert (tuple(y.shape), sr) == ((2, 0), 8000)


def test_clipping(lib: Lib) -> None:
    y, _ = se.apply_effects(lib.asarray(np.array([[2.0, -2.0, 0.5]], dtype=np.float32)), SR, [])
    assert_array_equal(np.asarray(y), [[1.0, -1.0, 0.5]])


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.int16, np.int32])
def test_bad_dtypes(lib: Lib, dtype: type) -> None:
    a = lib.asarray(np.zeros((2, 10), dtype))
    if np.asarray(a).dtype != dtype:  # e.g. JAX without x64 turns float64 into float32
        pytest.skip(f"{lib.name} has no {np.dtype(dtype).name}")
    with pytest.raises(TypeError, match="must be float32"):  # no silent conversion
        se.apply_effects(a, SR, [])


@pytest.mark.parametrize(
    ("shape", "match"), [((10,), "2D, got 1D"), ((1, 2, 10), "2D, got 3D"), ((0, 10), "at least one channel")]
)
def test_bad_shapes(lib: Lib, shape: tuple[int, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        se.apply_effects(lib.asarray(np.zeros(shape, np.float32)), SR, [])


def test_output_outlives_everything(lib: Lib, x: np.ndarray) -> None:
    y, _ = se.apply_effects(lib.asarray(x.copy()), SR, [["vol", "0.5"]])
    gc.collect()  # input gone, and the extension's buffer is only kept alive by `y`
    assert_allclose(np.asarray(y), 0.5 * x, atol=1e-7, rtol=0)


def test_effect_chain(lib: Lib, x: np.ndarray) -> None:
    base = EffectChain().vol(0.5)
    chain = base.pitch("-q", lambda: random.randint(-300, 300)).rate(SR)
    assert base.effects == (("vol", 0.5),)  # immutable
    draws = {chain.resolve()[1][2] for _ in range(20)}
    assert len(draws) > 1  # callables re-evaluated
    y, sr = chain.apply(lib.asarray(x), SR)
    assert isinstance(y, lib.array_type)
    assert sr == SR
    assert y.shape[0] == x.shape[0]
    assert abs(y.shape[1] - x.shape[1]) <= 1  # SoX's pitch effect can round the length by one sample
    last, _ = base.apply(lib.asarray(x).T, SR, channels_first=False)
    assert_allclose(np.asarray(last), 0.5 * x.T, atol=1e-7, rtol=0)


def test_threads_match_serial(lib: Lib, x: np.ndarray) -> None:
    a = lib.asarray(x)
    ref = [np.asarray(se.apply_effects(a, SR, HEAVY)[0]) for _ in range(8)]
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(lambda _: np.asarray(se.apply_effects(a, SR, HEAVY)[0]), range(8)))
    assert all(np.array_equal(r, o) for r, o in zip(ref, out, strict=True))


# --------------------------------------------------------------------------------------------
# Library specifics
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "broadcast",
    [lambda a: np.broadcast_to(a[:1], (3, a.shape[1])), lambda a: torch.from_numpy(a[:1]).expand(3, -1)],
    ids=["numpy", "torch"],
)
def test_zero_stride(x: np.ndarray, broadcast: Callable[[np.ndarray], Any]) -> None:
    y, _ = se.apply_effects(broadcast(x), SR, [])
    assert_allclose(np.asarray(y), np.repeat(x[:1], 3, axis=0), atol=1e-7, rtol=0)


def test_numpy_layouts(x: np.ndarray) -> None:
    fortran, _ = se.apply_effects(np.asfortranarray(x), SR, [])
    assert_allclose(fortran, x, atol=1e-7, rtol=0)
    x.flags.writeable = False
    read_only, _ = se.apply_effects(x, SR, [])
    assert_allclose(read_only, x, atol=1e-7, rtol=0)


def test_writable_outputs(x: np.ndarray) -> None:
    y, _ = se.apply_effects(x, SR, [])
    y += 1
    t, _ = se.apply_effects(torch.from_numpy(x), SR, [])
    t += 1
    assert_allclose(y, t.numpy())


def test_buffer_protocol(x: np.ndarray) -> None:
    y, _ = se.apply_effects(memoryview(x), SR, [])
    assert_allclose(np.asarray(y), x, atol=1e-7, rtol=0)


def test_not_an_array() -> None:
    with pytest.raises(TypeError, match="incompatible"):
        se.apply_effects([[0.0, 0.0]], SR, [])


def test_torch_requires_grad(x: np.ndarray) -> None:
    t = torch.from_numpy(x).clone().requires_grad_()
    with pytest.raises(ValueError, match=r"differentiable.*\.detach\(\)"):
        se.apply_effects(t, SR, [])
    with pytest.raises(ValueError, match="differentiable"):
        se.apply_effects(2 * t, SR, [])  # non-leaf tensor in a graph
    y, _ = se.apply_effects(t.detach(), SR, [])
    assert not y.requires_grad
    assert_allclose(y.numpy(), x, atol=1e-7, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_torch_cuda_rejected(x: np.ndarray) -> None:
    with pytest.raises(TypeError, match="CPU"):
        se.apply_effects(torch.from_numpy(x).cuda(), SR, [])


class _Augmented(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, i: int) -> torch.Tensor:  # ty: ignore[invalid-method-override]
        return se.apply_effects(torch.randn(1, SR) * 0.1, SR, HEAVY)[0]


def test_torch_dataloader() -> None:
    batches = list(DataLoader(_Augmented(), batch_size=4, num_workers=2))
    assert [b.shape for b in batches] == [(4, 1, SR)] * 2


def test_no_framework_imported() -> None:
    code = (
        "import sys, numpy as np, sox_effects as se\n"
        "se.apply_effects(np.zeros((1, 10), np.float32), 8000, [['vol', '0.5']])\n"
        "print(sorted({'torch', 'jax'} & sys.modules.keys()))"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout == "[]\n"


# --------------------------------------------------------------------------------------------
# Effects and arguments
# --------------------------------------------------------------------------------------------


def test_effect_names() -> None:
    names = se.effect_names()
    assert {"rate", "pitch", "reverb", "vol"} <= set(names)
    assert not set(BLOCKED) & set(names)


@pytest.mark.parametrize(
    ("effects", "match"),
    [
        ([["nope"]], "unsupported effect: 'nope'"),
        ([["rate", "abc"]], "invalid options for effect 'rate'"),
        ([[]], "empty effect"),
        *[([[name]], f"unsupported effect: '{name}'") for name in BLOCKED],
    ],
)
def test_bad_effects(x: np.ndarray, effects: list[list[str]], match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        se.apply_effects(x, SR, effects)


def test_bad_sample_rate(x: np.ndarray) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        se.apply_effects(x, 0, [])


def test_channels_first_is_keyword_only(x: np.ndarray) -> None:
    with pytest.raises(TypeError):
        se.apply_effects(x, SR, [], False)  # ty: ignore[too-many-positional-arguments]


@pytest.mark.skipif(shutil.which("sox") is None, reason="sox CLI not installed")
def test_matches_sox_cli(x: np.ndarray) -> None:
    mono = x[:1]
    cmd = [
        "sox",
        "-q",
        "-D",
        "-t",
        "f32",
        "-r",
        str(SR),
        "-c",
        "1",
        "-",
        "-t",
        "f32",
        "-",
        *functools.reduce(operator.iadd, HEAVY, []),
    ]
    raw = subprocess.run(cmd, input=mono.tobytes(), capture_output=True, check=True).stdout
    y, _ = se.apply_effects(mono, SR, HEAVY)
    assert_array_equal(y, np.frombuffer(raw, dtype=np.float32)[None])
