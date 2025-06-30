#include <catch2/catch_test_macros.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/nn/linear.h>


using namespace metalchat;


TEST_CASE("Test function parameters", "function")
{
    hardware_accelerator gpu0(16);
    nn::linear<float, random_memory_container<float>> linear(gpu0);

    linear.set_parameter("weight", full<float>({3, 5}, 4.0));

    auto weight = linear.get_parameter("weight");
    REQUIRE(weight.dimensions() == 2);
    REQUIRE(weight.size(0) == 3);
    REQUIRE(weight.size(1) == 5);

    auto output = linear(shared_tensor(full<float>({10, 5}, 2.0))).get();
    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == 10);
    REQUIRE(output.size(1) == 3);

    auto params = linear.get_parameters();
    REQUIRE(params.size() == 1);
}
