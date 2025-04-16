#include <cmath>

#include <catch2/catch_test_macros.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel/rmsnorm.h>
#include <metalchat/tensor.h>
#include <metalchat/format.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("RMSNorm array of ones", "[kernel::rmsnorm]")
{
    auto input = full<bf16>({4, 3, 5, 7}, 1.0);
    auto weight = full<bf16>({7}, 3.0);

    metalchat::device device("metalchat.metallib");
    metalchat::rmsnorm<bf16> rms(device);

    auto output = rms(input, weight);
    REQUIRE(output.numel() == input.numel());
    REQUIRE(output.dim() == input.dim());

    for (auto i = 0; i < input.dim(); i++) {
        REQUIRE(output.size(i) == input.size(i));
    }

    for (auto& n : output) {
        REQUIRE(n == 3.0);
    }
}


TEST_CASE("RMSNorm array of random numbers", "[kernel::rmsnorm]")
{
    auto input = full<bf16>({4, 3, 5, 7}, 1.0);
    auto weight = full<bf16>({7}, 3.0);

    metalchat::device device("metalchat.metallib");
    metalchat::rmsnorm<bf16> rms(device);

    auto output = rms(input, weight);
    REQUIRE(output.numel() == input.numel());
    REQUIRE(output.dim() == input.dim());

    for (auto i = 0; i < input.dim(); i++) {
        REQUIRE(output.size(i) == input.size(i));
    }

    for (auto b0 = 0; b0 < input.size(0); b0++) {
        for (auto b1 = 0; b1 < input.size(1); b1++) {
            for (auto b2 = 0; b2 < input.size(2); b2++) {
                bf16 rms = 0.0;
                for (auto i = 0; i < input.size(3); i++) {
                    rms += input[b0][b1][b2][i] * input[b0][b1][b2][i];
                }

                bf16 inv_mean = std::sqrt(rms / weight.size(0) + 1e-5);
                for (auto i = 0; i < input.size(3); i++) {
                    bf16 result = weight[i] * input[b0][b1][b2][i] * inv_mean;
                    REQUIRE(output[b0][b1][b2][i] == result);
                }
            }
        }
    }
}
