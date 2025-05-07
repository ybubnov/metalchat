#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel/rmsnorm.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("RMSNorm array of ones", "[kernel::rmsnorm]")
{
    auto input = shared_tensor(full<bf16>({4, 3, 5, 7}, 1.0));
    auto weight = shared_tensor(full<bf16>({7}, 3.0));

    metalchat::hardware_accelerator gpu0;
    kernel::rmsnorm<bf16> rms(gpu0);

    auto output = rms(input, weight).get();
    REQUIRE(output.numel() == input.numel());
    REQUIRE(output.dim() == input.dim());

    for (std::size_t i = 0; i < input.dim(); i++) {
        REQUIRE(output.size(i) == input.size(i));
    }

    for (auto& n : output) {
        REQUIRE(n == 3.0);
    }
}


TEST_CASE("RMSNorm array of random numbers", "[kernel::rmsnorm]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::rmsnorm<float> rms(gpu0);

    auto input = shared_tensor(rand<float>({3, 5, 2048}));
    auto weight = shared_tensor(rand<float>({2048}));

    auto output = rms(input, weight).get();
    REQUIRE(output.numel() == input.numel());
    REQUIRE(output.dim() == input.dim());

    for (std::size_t i = 0; i < input.dim(); i++) {
        REQUIRE(output.size(i) == input.size(i));
    }

    for (std::size_t b0 = 0; b0 < input.size(0); b0++) {
        for (std::size_t b1 = 0; b1 < input.size(1); b1++) {
            float sum_of_squares = 0.0;
            for (std::size_t i = 0; i < input.size(2); i++) {
                sum_of_squares += (input[b0, b1, i] * input[b0, b1, i]);
            }

            float inv_rms = 1 / std::sqrt((sum_of_squares / weight.size(0)) + 1e-5);
            for (std::size_t i = 0; i < input.size(2); i++) {
                auto result = weight[i] * input[b0, b1, i] * inv_rms;
                REQUIRE_THAT((output[b0, b1, i]), Catch::Matchers::WithinAbs(result, 0.00001));
            }
        }
    }
}
