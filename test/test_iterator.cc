#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;
using namespace metalchat::indexing;


TEST_CASE("Iterate 2d tensor", "[tensor::begin]")
{
    auto T = rand<bf16>({10, 7});
    std::cout << T << std::endl;

    auto S = T[slice(2, 6), slice(4, 7)];
    std::cout << S << std::endl;

    std::vector<bf16> data;

    auto last = S.end();
    for (auto first = S.begin(); first != last; ++first) {
        data.push_back(*first);
    }

    REQUIRE(data.size() == S.numel());
}
