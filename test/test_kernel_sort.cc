#include <algorithm>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <metalchat/format.h>
#include <metalchat/kernel/sort.h>


using namespace metalchat;


TEST_CASE("Test sorting", "[kernel::sort]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::sort<float, 32> sort(gpu0);

    auto input = shared_tensor(rand<float>({1, 1, 2048}));
    auto output = sort(input).get();

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == input.size(1));
    REQUIRE(output.size(1) == input.size(1));
    REQUIRE(output.size(2) == input.size(2));

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            auto output_ij = output[i][j];
            std::cout << output_ij << std::endl;
            REQUIRE(std::is_sorted(output_ij.begin(), output_ij.end(), std::greater<float>()));
        }
    }
}
