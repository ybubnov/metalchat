#include <ranges>

#include <catch2/catch_test_macros.hpp>


#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/functional.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Tensor concatenate", "[concatenate]")
{
    auto t0 = shared_tensor(full<float>({3, 4, 2}, 1.0));
    auto t1 = shared_tensor(full<float>({3, 4, 2}, 2.0));
    auto t2 = shared_tensor(full<float>({3, 4, 2}, 3.0));
    auto t3 = shared_tensor(full<float>({3, 4, 2}, 4.0));
    auto t4 = shared_tensor(full<float>({3, 4, 2}, 5.0));

    metalchat::hardware_accelerator gpu0;

    auto tensors = {t0, t1, t2, t3, t4};

    auto output0 = concatenate(tensors, 0, gpu0).get();
    REQUIRE(output0.dim() == 3);
    REQUIRE(output0.size(0) == 15);
    REQUIRE(output0.size(1) == 4);
    REQUIRE(output0.size(2) == 2);
    REQUIRE(output0[0, 0, 0] == 1.0);
    REQUIRE(output0[14, 3, 1] == 5.0);

    auto output2 = concatenate(tensors, 2, gpu0).get();
    REQUIRE(output2.dim() == 3);
    REQUIRE(output2.size(0) == 3);
    REQUIRE(output2.size(1) == 4);
    REQUIRE(output2.size(2) == 10);
    REQUIRE(output2[0, 0, 0] == 1.0);
    REQUIRE(output2[2, 3, 9] == 5.0);
}
