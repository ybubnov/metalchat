#include <algorithm>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <metalchat/format.h>
#include <metalchat/kernel/sort.h>


using namespace metalchat;


TEST_CASE("Test sorting", "[kernel::sort]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::sort<float, 2048> sort(gpu0);

    auto input = shared_tensor(rand<float>({1, 1, 128256}));
    auto [values_future, indices_future] = sort(input);

    auto values = values_future.get();
    auto indices = indices_future.get();

    REQUIRE(values.dim() == 3);
    REQUIRE(values.size(0) == input.size(0));
    REQUIRE(values.size(1) == input.size(1));
    REQUIRE(values.size(2) == input.size(2));

    REQUIRE(indices.dim() == 3);
    REQUIRE(indices.size(0) == input.size(0));
    REQUIRE(indices.size(1) == input.size(1));
    REQUIRE(indices.size(2) == input.size(2));

    std::cout << input << std::endl;
    std::cout << values << std::endl;
    std::cout << indices << std::endl;

    for (std::size_t i = 0; i < values.size(0); i++) {
        for (std::size_t j = 0; j < values.size(1); j++) {
            auto values_ij = values[i][j];
            REQUIRE(std::is_sorted(values_ij.begin(), values_ij.end(), std::greater<float>()));

            auto indices_ij = indices[i][j];
            std::vector<float> values_out(indices_ij.size(0));

            for (std::size_t k = 0; k < values_out.size(); k++) {
                values_out[k] = input[i][0][values_ij[k]];
            }
            REQUIRE(std::is_sorted(values_out.begin(), values_out.end(), std::greater<float>()));
        }
    }
}
