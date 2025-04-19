#include <catch2/catch_test_macros.hpp>


#include <metalchat/tensor_polymorphic.h>

using namespace metalchat;


TEST_CASE("Tensor polymorphic", "[tensor_polymorphic::tensor_polymorphic]")
{
    auto t = polymorphic_tensor(rand<float>({3, 2}));

    REQUIRE(t.dimensions() == 2);
    REQUIRE(t.size(0) == 3);
    REQUIRE(t.size(1) == 2);

    REQUIRE(t.strides().size() == 2);
    REQUIRE(t.offsets().size() == 2);
}
