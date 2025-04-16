#include <algorithm>
#include <functional>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/functional/softmax.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Softmax predefined array", "[functional::softmax]")
{
    auto input = empty<bf16>({5});
    for (std::size_t i = 0; i < 5; i++) {
        input[i] = bf16(i);
    }

    metalchat::device gpu0("metalchat.metallib");
    metalchat::softmax<bf16> softmax(gpu0);

    auto output = softmax(input);

    REQUIRE(input.dim() == output.dim());
    REQUIRE(input.size(0) == output.size(0));

    std::array<bf16, 5> expect({0.0116577, 0.0317383, 0.0859375, 0.234375, 0.636719});
    for (std::size_t i = 0; i < 5; i++) {
        REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expect[i], 0.00001));
    }
}


TEST_CASE("Softmax sum should be 1.0", "[functional::softmax]")
{
    auto input = rand<bf16>({30});

    metalchat::device gpu0("metalchat.metallib");
    metalchat::softmax<bf16> softmax(gpu0);

    auto output = softmax(input);
    std::cout << output << std::endl;

    auto sum = std::reduce(output.data_ptr(), output.data_ptr() + output.numel());
    REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(1.0, 0.01));
}
