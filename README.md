# sox-effects

Apply [SoX](https://sox.sourceforge.net/) audio effects (pitch, reverb, tempo, gain, …) directly to arrays from
NumPy, PyTorch, JAX, or any library that supports DLPack or the buffer protocol. It has no Python dependencies,
and replaces `torchaudio.sox_effects.apply_effects_tensor`, which was removed from torchaudio.

## Installation

```sh
pip install sox-effects
```

You'll also need the SoX library on your system:

```sh
apt install libsox3      # Debian / Ubuntu
dnf install sox          # Fedora / RHEL
brew install sox         # macOS
```

It works with Python >= 3.12, on Linux and macOS.

## Quick start

```python
import numpy as np, sox_effects

wav = np.random.randn(2, 16000).astype(np.float32) * 0.1  # (channels, time), float32, CPU
out, sr = sox_effects.apply_effects(
    wav, 16000, [["gain", "-n"], ["pitch", "200"], ["reverb", "50"], ["rate", "16000"]]
)
sox_effects.effect_names()  # list the available effects
```

Effects and their arguments are the same as on the `sox` command line. The output comes back in the same
library as the input (a torch tensor in, a torch tensor out), without copies at the boundaries.
For `(time, channels)` arrays, pass `channels_first=False`.

### Random augmentation

`EffectChain` lets you build a chain of effects step by step. Every SoX effect is available as a
method, and any argument given as a function is re-drawn each time you call `apply`:

```python
import random
from sox_effects import EffectChain

chain = EffectChain().pitch("-q", lambda: random.randint(-300, 300)).rate(16000).reverb(50).channels(1)
out, sr = chain.apply(wav, 16000)
```

## Good to know

- **Inputs are strict.** Arrays must be 2D, float32 and on the CPU: anything else raises an error.
  SoX effects aren't differentiable, so torch tensors that require grad are rejected too: call
  `.detach()` first (no copy) to cut the gradient explicitly. Dataset tensors used for augmentation
  don't require grad, so this only concerns outputs of a network.
- **Keep values in [-1, 1].** SoX silently clips anything outside this range, including what an
  effect pushes past full scale (e.g. `gain +10`). Use `["gain", "-n"]` or `vol` to stay safe.
- **The output may change shape.** `speed`, `pitch` and `tempo` change the length, and `speed` and
  `pitch` also change the sample rate: add `["rate", "<sr>"]` to get back to your rate. `reverb`
  turns mono into stereo: add `["channels", "1"]`. Nothing is added for you, so with a
  `DataLoader` you may need to crop/pad or use a custom `collate_fn`.
- **Always pass arguments.** SoX's defaults aren't always sensible: `["tempo"]` on its own never
  returns.
- **Go faster with `-q`.** For augmentation, quick mode trades a little quality for speed:
  `["pitch", "-q", "200"]`, `["tempo", "-q", "1.1"]`, `["rate", "-q", "16000"]` (the latter can
  alias). Put `-q` before the effect's other arguments.
- **Parallelism.** Threads run in parallel, but `DataLoader` worker processes are the safer choice.
  With `spawn`/`forkserver` workers, use regular functions instead of lambdas in an `EffectChain`
  (lambdas can't be pickled).
- Effects that read or write files (`spectrogram`, `noiseprof`, `noisered`, `splice`) aren't supported.

## Troubleshooting

`import sox_effects` always works; the SoX library is only loaded on the first call. If it can't be
found, install it as shown above, or point `LIBSOX_PATH` to the library file (e.g.
`libsox.so.3`).

## Development

The extension is built with [nanobind](https://github.com/wjakob/nanobind) and
[scikit-build-core](https://github.com/scikit-build/scikit-build-core). Building from source only needs the
`sox.h` header, which is looked up in `$SOX_ROOT/include`, next to the `sox` executable on your `PATH`, or in
`/usr`, `/usr/local` and `/opt/homebrew`.

```sh
uv run --group test pytest    # or: pip install . --group test && pytest
```

The tests run against NumPy, PyTorch and JAX, which are test dependencies only.

The extension uses the Python stable ABI, so a single build works across all Python versions >= 3.12.

## License

sox-effects is released under the MIT License. It builds on the approach from
[torchaudio](https://github.com/pytorch/audio) (BSD 2-Clause License) and
[WavAugment](https://github.com/facebookresearch/WavAugment) (MIT License).
