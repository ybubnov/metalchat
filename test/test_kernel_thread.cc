#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/device.h>
#include <metalchat/format.h>
#include <metalchat/kernel/add.h>
#include <metalchat/tensor_shared.h>


using namespace metalchat;


TEST_CASE("Kernel thread", "[kernel::thread]")
{
    metalchat::device gpu0("metalchat.metallib", 2);
    metalchat::add<float> add(gpu0);

    auto input1 = shared_tensor(full<float>({3, 4, 5}, 1.0));
    auto output1 = add(input1, input1);
    auto output2 = add(output1, output1);
    auto output3 = add(output2, output2);

    auto result = output3.get();
    REQUIRE(result.dim() == 3);
    REQUIRE(result.size(0) == 3);
    REQUIRE(result.size(1) == 4);
    REQUIRE(result.size(2) == 5);

    for (std::size_t i = 0; i < result.size(0); i++) {
        for (std::size_t j = 0; j < result.size(1); j++) {
            for (std::size_t k = 0; k < result.size(2); k++) {
                REQUIRE((result[i, j, k]) == 8.0f);
            }
        }
    }
}
