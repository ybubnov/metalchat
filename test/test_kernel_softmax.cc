#include <algorithm>
#include <functional>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel/softmax.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Softmax predefined array", "[kernel::softmax]")
{
    auto input = shared_tensor(empty<bf16>({5}));
    for (std::size_t i = 0; i < 5; i++) {
        input[i] = bf16(i);
    }

    metalchat::device gpu0("metalchat.metallib");
    metalchat::softmax<bf16> softmax(gpu0);

    auto output = softmax(input).get();

    REQUIRE(input.dim() == output.dim());
    REQUIRE(input.size(0) == output.size(0));

    std::array<bf16, 5> expect({0.0116577, 0.0317383, 0.0859375, 0.234375, 0.636719});
    for (std::size_t i = 0; i < 5; i++) {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expect[i], 0.00001));
    }
}


TEST_CASE("Softmax sum should be 1.0", "[kernel::softmax]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::softmax<float> softmax(gpu0);

    auto input = shared_tensor(rand<float>({30}));
    auto output = softmax(input).get();

    auto sum = std::reduce(output.data_ptr(), output.data_ptr() + output.numel());
    REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(1.0, 0.01));
}


TEST_CASE("Softmax for 4-dimensional tensor", "[kernel::softmax]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::softmax<float> softmax(gpu0);

    auto input = shared_tensor(rand<float>({1, 32, 4, 4}));
    auto output = softmax(input).get();

    for (std::size_t i = 0; i < input.size(0); i++) {
        for (std::size_t j = 0; j < input.size(1); j++) {
            for (std::size_t k = 0; k < input.size(2); k++) {
                auto tensor = output[i][j][k];
                auto sum = std::reduce(tensor.data_ptr(), tensor.data_ptr() + input.size(3));
                REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(1.0, 0.00001));
            }
        }
    }
}
