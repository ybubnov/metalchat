#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE indexing

#include <boost/test/included/unit_test.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;
using namespace metalchat::indexing;


BOOST_AUTO_TEST_SUITE(TestIndexing)


BOOST_AUTO_TEST_CASE(tensor_slice)
{
    auto T = full<bf16>({4, 5}, 5.0);
    for (auto i = 0; i < T.size(0); i++) {
        for (auto j = 0; j < T.size(1); j++) {
            BOOST_REQUIRE_EQUAL(T[i][j], 5.0);
            T[i][j] = bf16((i+1)*10 + j);
        }
    }

    std::cout << T << std::endl;
    auto S = T[slice(1, 3), slice(2, 4)];
    for (auto i = 0; i < S.size(0); i++) {
        for (auto j = 0; j < S.size(1); j++) {
            S[i][j] = 0.0;
        }
    }

    std::cout << T[slice(1, 3), slice(2, 4)] << std::endl;
    std::cout << T << std::endl;
}


BOOST_AUTO_TEST_SUITE_END()
