import sys
from collections.abc import Callable
from functools import cache
from typing import Any

from . import _C as _C  # ty: ignore[import-not-found]

__all__ = ["EffectChain", "apply_effects", "effect_names"]


def _same_library(like: Any, out: Any) -> Any:
    """Import the DLPack array ``out`` (zero-copy) into the library ``like`` comes from."""
    if hasattr(like, "__array_namespace__"):  # Array API: NumPy >= 2, JAX, ...
        return like.__array_namespace__().from_dlpack(out)
    library = sys.modules.get(type(like).__module__.partition(".")[0])
    if hasattr(library, "from_dlpack"):  # PyTorch, older NumPy, ...
        return library.from_dlpack(out)
    return out  # supports DLPack and the buffer protocol


def apply_effects[A](
    array: A,
    sample_rate: int,
    effects: list[list[str]],
    *,
    channels_first: bool = True,
) -> tuple[A, int]:
    """Apply a sox effects chain, e.g. ``[["gain", "-n"], ["speed", "0.9"], ["rate", "16000"]]``.

    ``array`` is a 2D float32 CPU array from any library supporting DLPack or the buffer protocol
    (NumPy, PyTorch, JAX, ...), ``(channels, time)`` if ``channels_first`` else ``(time, channels)``.
    Returns the processed array (same library and layout) and its sample rate.
    """
    if getattr(array, "requires_grad", False):  # otherwise torch's DLPack export fails with an obscure error
        raise ValueError("SoX effects aren't differentiable: call .detach() first to apply them without gradients")
    out, sr = _C.apply_effects(array, sample_rate, effects, channels_first)
    return _same_library(array, out), sr


@cache
def effect_names() -> list[str]:
    """Names of the effects available in the linked libsox."""
    return _C.effect_names()


class EffectChain:
    """WavAugment-style effects builder.

    Every sox effect is a method; arguments can be callables, which are re-evaluated on each ``apply``
    (for random augmentation)::

        chain = EffectChain().pitch("-q", lambda: random.randint(-300, 300)).rate(16000).reverb(50)
        y, sr = chain.apply(x, 16000)

    Chains are immutable: each method returns a new chain, so prefixes can be shared.
    """

    def __init__(self, effects: tuple[tuple[str, ...], ...] = ()) -> None:
        self.effects = effects

    def __getattr__(self, name: str) -> Callable[..., "EffectChain"]:
        if name.startswith("_") or name not in effect_names():
            raise AttributeError(f"{type(self).__name__!r} has no attribute or sox effect {name!r}")
        return lambda *args: EffectChain((*self.effects, (name, *args)))

    def resolve(self) -> list[list[str]]:
        """The concrete effect list for one application, with callables evaluated."""
        return [[e[0], *(str(a() if callable(a) else a) for a in e[1:])] for e in self.effects]

    def apply[A](self, array: A, sample_rate: int, *, channels_first: bool = True) -> tuple[A, int]:
        return apply_effects(array, sample_rate, self.resolve(), channels_first=channels_first)

    def __repr__(self) -> str:
        def fmt(a: Any) -> str:
            return getattr(a, "__name__", "<callable>") if callable(a) else repr(a)

        return "EffectChain()" + "".join(f".{e[0]}({', '.join(map(fmt, e[1:]))})" for e in self.effects)
