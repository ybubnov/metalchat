#include <ranges>

#include <catch2/catch_test_macros.hpp>


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/functional.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Tensor concatenate", "[concatenate]")
{
    auto t = rand<float>({3, 4, 2});
    auto tn = std::views::repeat(std::move(t), 5);

    auto output = concatenate(tn.begin(), tn.end(), 2);
    std::cout << output.sizes() << std::endl;

    REQUIRE(output.dim() == 4);
    REQUIRE(output.size(0) == 3);
    REQUIRE(output.size(1) == 4);
    REQUIRE(output.size(2) == 5);
    REQUIRE(output.size(3) == 2);
    // std::cout << output << std::endl;
}


TEST_CASE("Tensor concatenate the first dim", "[concatenate]")
{
    auto t0 = full<float>({2, 4}, 1.0);
    auto t1 = full<float>({2, 4}, 2.0);
    auto t2 = full<float>({2, 4}, 3.0);

    auto tn = concatenate({std::cref(t0), std::cref(t1), std::cref(t2)}, 1);
    std::cout << tn << std::endl;
}
