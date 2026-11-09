#include <dlfcn.h>
#include <sox.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace nb = nanobind;

// Throws a RuntimeError (on the Python side) whose message is the concatenation of `args`.
template <typename... Args> [[noreturn]] void fail(const Args&... args) {
  std::ostringstream msg;
  (msg << ... << args);
  throw std::runtime_error(msg.str());
}
#define CHECK(cond, ...)                                                                           \
  do {                                                                                             \
    if (!(cond)) fail(__VA_ARGS__);                                                                \
  } while (0)

// ================================================================================================
// 1. libsox: runtime loading
// ================================================================================================
namespace libsox {

// Every libsox function used below. Their types come from sox.h, so a mismatch is a compile error.
#define LIBSOX_FUNCTIONS(X)                                                                        \
  X(sox_init)                                                                                      \
  X(sox_get_globals)                                                                               \
  X(sox_get_effect_fns)                                                                            \
  X(sox_find_effect)                                                                               \
  X(sox_create_effect)                                                                             \
  X(sox_effect_options)                                                                            \
  X(sox_create_effects_chain)                                                                      \
  X(sox_add_effect)                                                                                \
  X(sox_flow_effects)                                                                              \
  X(sox_delete_effects_chain)

struct Api {
#define DECLARE(name) decltype(&::name) name;
  LIBSOX_FUNCTIONS(DECLARE)
#undef DECLARE
};

// Last error reported by libsox on this thread; cleared before each libsox call we check.
thread_local std::string last_error;

void on_message(unsigned level, const char*, const char* fmt, va_list ap) {
  if (level > 1) return; // keep only failures
  char msg[1024];
  std::vsnprintf(msg, sizeof msg, fmt, ap);
  last_error = msg;
}

Api load() {
  std::vector<std::string> candidates = {
      "libsox.so.3", "libsox.so", "libsox.3.dylib", "libsox.dylib",
#ifdef __APPLE__
      "/opt/homebrew/lib/libsox.3.dylib", "/opt/homebrew/lib/libsox.dylib",
#endif
  };
  if (const char* path = std::getenv("LIBSOX_PATH")) candidates = {path};
  void* handle = nullptr;
  std::string errors;
  for (const auto& name : candidates) {
    if ((handle = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL))) break;
    const char* err = dlerror();
    errors += "\n  " + (err ? std::string(err) : name);
  }
  CHECK(handle != nullptr,
        "could not load libsox. Install it (apt install libsox3, dnf install sox, "
        "brew install sox), add its directory to the library search path, or set "
        "LIBSOX_PATH to the library file. Tried:", errors);
  Api api;
#define RESOLVE(name)                                                                              \
  api.name = reinterpret_cast<decltype(api.name)>(dlsym(handle, #name));                          \
  CHECK(api.name != nullptr, "libsox is missing the symbol " #name);
  LIBSOX_FUNCTIONS(RESOLVE)
#undef RESOLVE
  api.sox_get_globals()->output_message_handler = on_message;
  api.sox_get_globals()->verbosity = 1;
  CHECK(api.sox_init() == SOX_SUCCESS, "sox_init() failed");
  return api;
}

// Loaded and initialized on first use, so importing the package works without libsox.
// If loading throws, the next call retries.
const Api& api() {
  static const Api instance = load();
  return instance;
}

} // namespace libsox

// ================================================================================================
// 2. chain: sox effects chain on interleaved sample buffers
// ================================================================================================
namespace chain {

using Buffer = std::vector<sox_sample_t>;
using Effects = std::vector<std::vector<std::string>>;

// Effects that read or write files, or are the CLI's own file input/output.
bool unsupported(const std::string& name) {
  for (const char* n : {"input", "output", "spectrogram", "noiseprof", "noisered", "splice"})
    if (name == n) return true;
  return false;
}

// First effect of the chain: drains samples from a buffer.
struct InPriv {
  const Buffer* buf;
  size_t pos;
  unsigned channels;
};

int in_drain(sox_effect_t* e, sox_sample_t* obuf, size_t* osamp) {
  auto* p = static_cast<InPriv*>(e->priv);
  size_t n = std::min(*osamp - *osamp % p->channels, p->buf->size() - p->pos);
  std::memcpy(obuf, p->buf->data() + p->pos, n * sizeof(sox_sample_t));
  p->pos += n;
  *osamp = n;
  return p->pos == p->buf->size() ? SOX_EOF : SOX_SUCCESS;
}

// Last effect of the chain: appends everything it receives to a buffer.
struct OutPriv {
  Buffer* buf;
};

int out_flow(sox_effect_t* e, const sox_sample_t* ibuf, sox_sample_t*, size_t* isamp, size_t* osamp) {
  Buffer* buf = static_cast<OutPriv*>(e->priv)->buf;
  buf->insert(buf->end(), ibuf, ibuf + *isamp);
  *osamp = 0;
  return SOX_SUCCESS;
}

const sox_effect_handler_t in_handler{"buffer_input", nullptr, SOX_EFF_MCHAN, nullptr, nullptr,
                                      nullptr, in_drain, nullptr, nullptr, sizeof(InPriv)};
const sox_effect_handler_t out_handler{"buffer_output", nullptr, SOX_EFF_MCHAN, nullptr, nullptr,
                                       out_flow, nullptr, nullptr, nullptr, sizeof(OutPriv)};

// sox_add_effect copies the effect (taking over its priv), so we only free the struct itself.
struct FreeEffect {
  void operator()(sox_effect_t* e) const { std::free(e); }
};
using Effect = std::unique_ptr<sox_effect_t, FreeEffect>;

// Applies `effects` to the interleaved samples `in`, described by `sig`.
// Returns the output samples, and updates `sig` to describe them (rate, channels).
Buffer run(const Buffer& in, sox_signalinfo_t& sig, Effects effects) {
  const libsox::Api& sox = libsox::api();
  const sox_signalinfo_t in_sig = sig; // target signal: effects keep rate/channels unless told otherwise
  sox_encodinginfo_t enc{SOX_ENCODING_FLOAT, 32, 0, sox_option_default, sox_option_default,
                         sox_option_default, sox_false};
  auto delete_chain = [&sox](sox_effects_chain_t* c) { sox.sox_delete_effects_chain(c); };
  std::unique_ptr<sox_effects_chain_t, decltype(delete_chain)> ch(
      sox.sox_create_effects_chain(&enc, &enc), delete_chain);
  Buffer out;

  auto add = [&](Effect e, const std::string& name) {
    libsox::last_error.clear();
    CHECK(sox.sox_add_effect(ch.get(), e.get(), &sig, &in_sig) == SOX_SUCCESS,
          "failed to add effect '", name, "': ", libsox::last_error);
  };

  Effect input(sox.sox_create_effect(&in_handler));
  *static_cast<InPriv*>(input->priv) = {&in, 0, sig.channels};
  add(std::move(input), "input");

  for (auto& args : effects) {
    CHECK(!args.empty(), "empty effect");
    const std::string& name = args[0];
    const sox_effect_handler_t* h = unsupported(name) ? nullptr : sox.sox_find_effect(name.c_str());
    CHECK(h != nullptr, "unsupported effect: '", name, "'");
    Effect e(sox.sox_create_effect(h));
    std::vector<char*> argv;
    for (size_t i = 1; i < args.size(); ++i) argv.push_back(args[i].data());
    libsox::last_error.clear();
    if (sox.sox_effect_options(e.get(), static_cast<int>(argv.size()), argv.data()) != SOX_SUCCESS) {
      std::free(e->priv); // not yet handed over to the chain
      fail("invalid options for effect '", name, "': ", libsox::last_error);
    }
    add(std::move(e), name);
  }

  Effect output(sox.sox_create_effect(&out_handler));
  static_cast<OutPriv*>(output->priv)->buf = &out;
  add(std::move(output), "output");

  libsox::last_error.clear();
  sox.sox_flow_effects(ch.get(), nullptr, nullptr);
  CHECK(libsox::last_error.empty(), "sox_flow_effects failed: ", libsox::last_error);
  return out;
}

} // namespace chain

// ================================================================================================
// 3. ops: Python bindings; waveforms: any CPU float32 2D array (numpy, torch, jax, ...) via DLPack
// ================================================================================================
namespace ops {

// No dtype/shape/device constraints, so nanobind never converts (copies) the input: anything that
// doesn't match exactly is rejected below with an explicit message. `ro` accepts read-only arrays.
using Input = nb::ndarray<nb::ro>;
using Output = nb::ndarray<nb::array_api, float, nb::ndim<2>>;

std::tuple<Output, int64_t> apply_effects(Input waveform, int64_t sample_rate,
                                          chain::Effects effects, bool channels_first) {
  if (waveform.device_type() != nb::device::cpu::value)
    throw nb::type_error("waveform must be on the CPU");
  if (waveform.dtype() != nb::dtype<float>()) throw nb::type_error("waveform must be float32");
  if (waveform.ndim() != 2)
    throw nb::value_error(("waveform must be 2D, got " + std::to_string(waveform.ndim()) + "D").c_str());
  if (sample_rate <= 0) throw nb::value_error("sample_rate must be positive");

  const int cdim = channels_first ? 0 : 1;
  const int64_t C = static_cast<int64_t>(waveform.shape(cdim));
  const int64_t T = static_cast<int64_t>(waveform.shape(1 - cdim));
  const int64_t sc = waveform.stride(cdim), st = waveform.stride(1 - cdim); // in elements
  if (C == 0) throw nb::value_error("waveform must have at least one channel");
  const float* x = static_cast<const float*>(waveform.data());

  float* yp = nullptr;
  int64_t oC = 0, oT = 0, rate = 0;
  {
    // `waveform` keeps the input alive, so its memory can be read without the GIL.
    nb::gil_scoped_release no_gil;

    // float32 array (any strides) -> interleaved int32 samples. Values outside [-1, 1] clip.
    SOX_SAMPLE_LOCALS;
    size_t clips = 0;
    chain::Buffer in(static_cast<size_t>(C * T));
    for (int64_t t = 0; t < T; ++t)
      for (int64_t c = 0; c < C; ++c)
        in[t * C + c] = SOX_FLOAT_32BIT_TO_SAMPLE(x[c * sc + t * st], clips);

    sox_signalinfo_t sig{static_cast<sox_rate_t>(sample_rate), static_cast<unsigned>(C), 32,
                         static_cast<sox_uint64_t>(in.size()), nullptr};
    const chain::Buffer out = chain::run(in, sig, std::move(effects));

    // interleaved int32 samples -> float32 buffer in the input layout
    oC = sig.channels;
    oT = static_cast<int64_t>(out.size()) / oC;
    rate = static_cast<int64_t>(sig.rate);
    yp = new float[static_cast<size_t>(oC * oT)];
    for (int64_t t = 0; t < oT; ++t)
      for (int64_t c = 0; c < oC; ++c)
        yp[channels_first ? c * oT + t : t * oC + c] = SOX_SAMPLE_TO_FLOAT_32BIT(out[t * oC + c], clips);
  }

  nb::capsule owner(yp, [](void* p) noexcept { delete[] static_cast<float*>(p); });
  const size_t rows = static_cast<size_t>(channels_first ? oC : oT);
  const size_t cols = static_cast<size_t>(channels_first ? oT : oC);
  return {Output(yp, {rows, cols}, owner), rate};
}

std::vector<std::string> effect_names() {
  std::vector<std::string> names;
  for (const sox_effect_fn_t* fn = libsox::api().sox_get_effect_fns(); *fn; ++fn) {
    const sox_effect_handler_t* h = (*fn)();
    if (h && h->name && !(h->flags & SOX_EFF_DEPRECATED) && !chain::unsupported(h->name))
      names.emplace_back(h->name);
  }
  return names;
}

} // namespace ops

NB_MODULE(_C, m) {
  m.def("apply_effects", &ops::apply_effects, nb::arg("waveform"), nb::arg("sample_rate"),
        nb::arg("effects"), nb::arg("channels_first") = true,
        "Apply a sox effects chain to a CPU float32 2D array. Returns a DLPack array and the sample rate.");
  m.def("effect_names", &ops::effect_names, "Names of the effects available in the linked libsox.");
}
