#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE functional

#include <boost/test/included/unit_test.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/functional/softmax.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;

namespace test = boost::test_tools;


BOOST_AUTO_TEST_SUITE(TestSoftmax)


BOOST_AUTO_TEST_CASE(tensor_softmax)
{
    auto input = empty<bf16>({5});
    for (std::size_t i = 0; i < 5; i++) {
        input[i] = bf16(i);
    }

    metalchat::device gpu0("metalchat.metallib");
    metalchat::softmax<bf16> softmax(gpu0);

    auto output = softmax(input);

    BOOST_REQUIRE_EQUAL(input.dim(), output.dim());
    BOOST_REQUIRE_EQUAL(input.size(0), output.size(0));

    std::array<bf16, 5> expect({0.0116577, 0.0317383, 0.0859375, 0.234375, 0.636719});
    for (std::size_t i = 0; i < 5; i++) {
        BOOST_TEST(output[i] == expect[i], test::tolerance(0.00001));
    }
}


BOOST_AUTO_TEST_SUITE_END()
