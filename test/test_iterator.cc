#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;
using namespace metalchat::indexing;


TEST_CASE("Iterate 3d tensor slice", "[tensor_iterator]")
{
    auto t = rand<float>({10, 7, 6});
    auto s = t[slice(2, 7), slice(4, 7), slice(2, 4)];

    std::vector<float> data;

    auto last = s.end();
    for (auto first = s.begin(); first != last; ++first) {
        data.push_back(*first);
    }

    REQUIRE(data.size() == s.numel());

    auto first = data.begin();

    for (std::size_t i = 0; i < s.size(0); i++) {
        for (std::size_t j = 0; j < s.size(1); j++) {
            for (std::size_t k = 0; k < s.size(2); k++) {
                REQUIRE(s[i, j, k] == (*first));
                ++first;
            }
        }
    }
}
