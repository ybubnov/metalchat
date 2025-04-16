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


// TEST_CASE("RoPE for Llama", "[kernel::rope]")
// {
//     auto input = rand<bf16>({1, 32, 4, 64});
//
//     metalchat::device gpu0("metalchat.metallib");
//     metalchat::rope<bf16> rope(gpu0, 64);
// }
