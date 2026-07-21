#include "perennis/perennis.hpp"

#include <array>
#include <cstring>

#include <gtest/gtest.h>

TEST(GgmlSmoke, AddAndMulComputeExpectedResult) {
  const size_t ctx_size = (4 * ggml_tensor_overhead()) + ggml_graph_overhead() + 1024;
  const ggml_init_params params{
      /*.mem_size   =*/ctx_size,
      /*.mem_buffer =*/nullptr,
      /*.no_alloc   =*/false,
  };

  ggml_context* ctx = ggml_init(params);
  ASSERT_NE(ctx, nullptr);

  ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
  ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);

  const std::array<float, 4> a_data = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<float, 4> b_data = {1.0F, 1.0F, 1.0F, 1.0F};
  std::memcpy(a->data, a_data.data(), sizeof(float) * a_data.size());
  std::memcpy(b->data, b_data.data(), sizeof(float) * b_data.size());

  ggml_tensor* sum = ggml_add(ctx, a, b);
  ggml_tensor* product = ggml_mul(ctx, sum, b);

  ggml_cgraph* graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, product);
  ggml_graph_compute_with_ctx(ctx, graph, /*n_threads=*/1);

  const auto* result = static_cast<const float*>(product->data);
  const std::array<float, 4> expected = {2.0F, 3.0F, 4.0F, 5.0F};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(result[i], expected[i]);
  }

  ggml_free(ctx);
}
