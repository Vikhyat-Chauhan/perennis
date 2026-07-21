# perennis

An LLM inference engine targeting session-aware serving for agentic workloads: treating the agent session, not the request, as the primary unit of scheduling and caching.

**Status:** Phase 1 (tensor foundations) scaffolded — built on [ggml](https://github.com/ggml-org/ggml).

See [`docs/design.md`](docs/design.md) for the full problem statement, design thesis, and phased roadmap.

## Building

Requires CMake 3.20+ and a C++20 compiler (GCC or Clang). ggml is vendored as a git submodule.

```sh
git submodule update --init --recursive
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Sanitizer builds:

```sh
cmake -S . -B build -DPERENNIS_ENABLE_ASAN=ON -DPERENNIS_ENABLE_UBSAN=ON
```

## API tour

`perennis` builds directly on [ggml](https://github.com/ggml-org/ggml) (vendored under `third_party/ggml`) for tensor allocation, memory contexts, and the compute graph. Include `perennis/perennis.hpp`, which currently just re-exports `ggml.h`.

```cpp
#include "perennis/perennis.hpp"

ggml_init_params params{/*.mem_size=*/1024 * 1024, /*.mem_buffer=*/nullptr, /*.no_alloc=*/false};
ggml_context* ctx = ggml_init(params);

ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
ggml_tensor* c = ggml_add(ctx, a, b);

ggml_cgraph* graph = ggml_new_graph(ctx);
ggml_build_forward_expand(graph, c);
ggml_graph_compute_with_ctx(ctx, graph, /*n_threads=*/1);

ggml_free(ctx);
```

## License

MIT — see [LICENSE](LICENSE).
