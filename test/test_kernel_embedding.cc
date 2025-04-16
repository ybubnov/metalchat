#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel/embedding.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


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
