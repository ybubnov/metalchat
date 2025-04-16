#include <catch2/catch_test_macros.hpp>

#include <metalchat/kernel/copy.h>
#include <metalchat/tensor_shared.h>


using namespace metalchat;


TEST_CASE("Copy 2-dimensional tensors", "[kernel::copy]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::cpy<float> copy(gpu0);

    auto input = shared_tensor(rand<float>({16, 64}));
    auto output = shared_tensor(empty<float>({16, 64}, gpu0));

    copy(input, output).wait();

    for (auto i = 0; i < input.size(0); i++) {
        for (auto j = 0; j < input.size(1); j++) {
            REQUIRE((input[i, j]) == (output[i, j]));
        }
    }
}


TEST_CASE("Copy into slice", "[kernel::copy]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::cpy<float> cpy(gpu0);

    auto input = shared_tensor(rand<float>({1, 6, 8, 1, 64}));
    auto output = shared_tensor(full<float>({1, 6, 8, 4, 64}, 0.0, gpu0));

    auto target = output.narrow(/*dim=*/3, /*offset=*/2, /*length=*/1);
    cpy(input.view({-1, 64}), target.view({-1, 64})).wait();

    for (auto i0 = 0; i0 < input.size(0); i0++) {
        for (auto i1 = 0; i1 < input.size(1); i1++) {
            for (auto i2 = 0; i2 < input.size(1); i2++) {
                for (auto i4 = 0; i4 < input.size(1); i4++) {
                    REQUIRE((input[i0, i1, i2, 0, i4]) == (output[i0, i1, i2, 2, i4]));
                }
            }
        }
    }
}
