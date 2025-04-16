#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel/embedding.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("RoPE array of ones", "[kernel::rope]")
{
    // {bs, n_heads, seqlen, head_dim}
    std::size_t head_dim = 4;
    auto input = full<bf16>({1, 4, 6, head_dim}, 1.0);

    metalchat::device gpu0("metalchat.metallib");
    metalchat::rope<bf16> rope(gpu0, head_dim);

    auto output = rope(input);

    float total_sum = 0;

    for (auto i = 0; i < output.size(0); i++) {
        for (auto j = 0; j < output.size(1); j++) {
            for (auto k = 0; k < output.size(2); k++) {
                for (auto l = 0; l < output.size(3); l++) {
                    total_sum += output[i][j][k][l];
                }
            }
        }
    }

    REQUIRE_THAT(total_sum, Catch::Matchers::WithinAbs(46.0, 0.1));
}


TEST_CASE("Embedding batched", "[kernel::embedding]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::embedding<float> emb(gpu0);

    auto input = full<int32_t>({3, 4}, 0.0);
    input[0, 0] = 0;
    input[0, 1] = 1;
    input[0, 2] = 2;
    input[0, 3] = 3;
    input[1, 0] = 2;
    input[1, 1] = 4;
    input[1, 2] = 1;
    input[1, 3] = 0;
    input[2, 0] = 4;
    input[2, 1] = 3;
    input[2, 2] = 3;
    input[2, 3] = 2;

    auto weight = rand<float>({5, 128256});
    auto output = emb(input, weight);

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 3);
    REQUIRE(output.size(1) == 4);
    REQUIRE(output.size(2) == 128256);

    for (auto i = 0; i < output.size(0); i++) {
        for (auto j = 0; j < output.size(1); j++) {
            for (auto k = 0; k < output.size(2); k++) {
                REQUIRE(output[i, j, k] == weight[input[i, j], k]);
            }
        }
    }
}
