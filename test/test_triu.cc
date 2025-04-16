#include <catch2/catch_test_macros.hpp>


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/functional.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::indexing;


TEST_CASE("Tensor", "triu")
{
    auto t = full<float>({10, 10}, 3.0f);
    triu<float>(t);

    for (std::size_t i = 0; i < t.size(0); i++) {
        for (std::size_t j = 0; j < t.size(1); j++) {
            if (j > i) {
                REQUIRE(t[i][j] == 3.0f);
            } else {
                REQUIRE(t[i][j] == 0.0f);
            }
        }
    }
}
