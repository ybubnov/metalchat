#include <algorithm>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <metalchat/format.h>
#include <metalchat/kernel/sort.h>


using namespace metalchat;


TEST_CASE("Test sorting", "[kernel::sort]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::sort<float, 8> sort(gpu0);

    auto input = shared_tensor(rand<float>({1, 1, 1 << 10}));

    // std::vector<float> v =
    // {0.541837,0.152851,0.304635,0.143152,0.0548083,0.675422,0.751769,0.132389};
    // std::vector<float> v =
    // {0.740266,0.383472,0.76658,0.00598811,0.346917,0.986587,0.635585,0.34211,0.369472,0.814,0.344635,0.561184,0.0855372,0.181075,0.276945,0.738361};
    // std::copy(v.begin(), v.end(), input.begin());
    std::cout << "input=" << std::endl << input << std::endl;
    auto output = sort(input).get();

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == input.size(1));
    REQUIRE(output.size(1) == input.size(1));
    REQUIRE(output.size(2) == input.size(2));

    std::cout << "output=" << std::endl << output << std::endl;
    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            auto output_ij = output[i][j];
            std::cout << output_ij << std::endl;
            REQUIRE(std::is_sorted(output_ij.begin(), output_ij.end(), std::greater<float>()));
        }
    }
}
