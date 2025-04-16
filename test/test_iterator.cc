#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;
using namespace metalchat::indexing;


TEST_CASE("Iterate 3d tensor slice", "[tensor_iterator]")
{
    auto T = rand<bf16>({10, 7, 6});
    auto S = T[slice(2, 7), slice(4, 7), slice(2, 4)];

    std::vector<bf16> data;

    auto last = S.end();
    for (auto first = S.begin(); first != last; ++first) {
        data.push_back(*first);
    }

    REQUIRE(data.size() == S.numel());

    auto first = data.begin();

    for (std::size_t i = 0; i < S.size(0); i++) {
        for (std::size_t j = 0; j < S.size(1); j++) {
            for (std::size_t k = 0; k < S.size(2); k++) {
                REQUIRE(S[i][j][k] == (*first));
                ++first;
            }
        }
    }
}
