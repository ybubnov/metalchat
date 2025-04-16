#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel/embedding.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


/*
TEST_CASE("RoPE for Llama", "[kernel::rope]")
{
    std::size_t head_dim = 64;
    std::size_t n_heads = 8;
    std::size_t max_seq_len = 1024;
    float theta = 500000.0;

    metalchat::device gpu0("metalchat.metallib");
    metalchat::nn::rope<float> rope(head_dim, max_seq_len, theta, gpu0);

    std::size_t seq_len = 7;
    auto input = full<float>({1, seq_len, n_heads, head_dim}, 1.0);
    auto output = rope(input);

    REQUIRE(output.dim() == 4);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == seq_len);
    REQUIRE(output.size(2) == n_heads);
    REQUIRE(output.size(3) == head_dim);

    std::cout << output << std::endl;
}


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
*/


TEST_CASE("Embedding batched", "[kernel::embedding]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::embedding<float, 16, 128> emb(gpu0);

    auto input = shared_tensor(full<int32_t>({3, 4}, 0.0));
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

    auto weight = shared_tensor(rand<float>({5, 128256}));
    auto output = emb(input, weight).get();

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
