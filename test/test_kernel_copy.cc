#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/format.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/tensor/shared.h>


using namespace metalchat;


TEST_CASE("Copy 2-dimensional tensors", "[kernel::copy]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::cpy<float> copy(gpu0);

    auto input = shared_tensor(rand<float>({16, 64}));
    auto output = shared_tensor(empty<float>({16, 64}, gpu0.get_allocator()));

    copy(input, output).wait();

    for (std::size_t i = 0; i < input.size(0); i++) {
        for (std::size_t j = 0; j < input.size(1); j++) {
            REQUIRE((input[i, j]) == (output[i, j]));
        }
    }
}


TEST_CASE("Copy into slice", "[kernel::copy]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::cpy<float> cpy(gpu0);

    auto input = shared_tensor(rand<float>({1, 6, 8, 1, 64}));
    auto output = shared_tensor(full<float>({1, 6, 8, 4, 64}, 0.0, gpu0.get_allocator()));

    auto target = output.narrow(/*dim=*/3, /*offset=*/2, /*length=*/1);
    cpy(input.view({-1, 64}), target.view({-1, 64})).wait();

    for (std::size_t i0 = 0; i0 < input.size(0); i0++) {
        for (std::size_t i1 = 0; i1 < input.size(1); i1++) {
            for (std::size_t i2 = 0; i2 < input.size(1); i2++) {
                for (std::size_t i4 = 0; i4 < input.size(1); i4++) {
                    REQUIRE((input[i0, i1, i2, 0, i4]) == (output[i0, i1, i2, 2, i4]));
                }
            }
        }
    }
}


TEST_CASE("Inplace index set", "[kernel::scatter]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::scatter<float> scatter(gpu0);

    auto input = shared_tensor(empty<float>({16, 128}, gpu0.get_allocator()));
    auto mask = shared_tensor(empty<bool>({16, 128}));

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    std::generate_n(mask.data_ptr(), mask.numel(), [&]() -> bool {
        return distribution(generator) > 0.5 ? true : false;
    });

    auto output = scatter(input, mask, 9.0f).get();
    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == input.size(0));
    REQUIRE(output.size(1) == input.size(1));

    for (std::size_t i = 0; i < input.size(0); i++) {
        for (std::size_t j = 0; j < input.size(1); j++) {
            if (mask[i, j]) {
                REQUIRE_THAT((input[i, j]), Catch::Matchers::WithinAbs(9.0f, 0.0001));
            }
        }
    }
}


TEST_CASE("Gather by index", "[kernel::gather]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::gather<float> gather(gpu0);

    auto input = shared_tensor(rand<float>({16, 128}));
    auto index = shared_tensor(empty<int32_t>({16, 10}));

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int32_t> distribution(0, 127);

    std::generate_n(index.data_ptr(), index.numel(), [&]() -> int32_t {
        return distribution(generator);
    });

    auto output = gather(input, index).get();
    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == index.size(0));
    REQUIRE(output.size(1) == index.size(1));

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            REQUIRE((output[i, j]) == (input[i, index[i, j]]));
        }
    }
}
