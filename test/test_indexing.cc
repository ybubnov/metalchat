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
            T[i][j] = bf16((i + 1) * 10 + j);
        }
    }

    auto S = T[slice(1, 3), slice(1, 4)];
    BOOST_REQUIRE_EQUAL(S.size(0), 2);
    BOOST_REQUIRE_EQUAL(S.size(1), 3);

    BOOST_REQUIRE_EQUAL(S.stride(0), 5);
    BOOST_REQUIRE_EQUAL(S.stride(1), 1);
    BOOST_REQUIRE(!S.is_contiguous());

    for (auto i = 0; i < S.size(0); i++) {
        for (auto j = 0; j < S.size(1); j++) {
            S[i][j] = 0.0;
        }
    }

    // Ensure that writing through the tensor view updates values in the
    // underlying storage.
    for (auto i = 1; i < 3; i++) {
        for (auto j = 1; j < 4; j++) {
            BOOST_REQUIRE_EQUAL(T[i][j], 0.0);
        }
    }
}


BOOST_AUTO_TEST_SUITE_END()
