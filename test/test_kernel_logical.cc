#include <catch2/catch_test_macros.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel/logical.h>
#include <metalchat/tensor.h>

using namespace metalchat;


TEST_CASE("Greater-than for 3-dimensional tensors", "[kernel::gt]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::gt<float> gt(gpu0);

    auto input = shared_tensor(rand<float>({1, 4, 2048}));
    auto output = gt(input, 0.5f).get();

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 4);
    REQUIRE(output.size(2) == 2048);

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            for (std::size_t k = 0; k < output.size(2); k++) {
                REQUIRE((output[i, j, k]) == (input[i, j, k] > 0.5f));
            }
        }
    }
}
