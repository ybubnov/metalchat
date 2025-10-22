#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/kernel/roll.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Roll function 3-dim", "[kernel::roll]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::roll<float> roll(gpu0);

    auto input = shared_tensor(rand<float>({2, 4, 5}));
    auto output = roll(input, 1, 1).get();

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == input.size(0));
    REQUIRE(output.size(1) == input.size(1));
    REQUIRE(output.size(2) == input.size(2));

    for (std::size_t b = 0; b < output.size(0); b++) {
        for (std::size_t s0 = 0; s0 < output.size(1); s0++) {
            std::size_t s1 = (s0 + 1) % output.size(1);

            for (std::size_t i = 0; i < output.size(2); i++) {
                REQUIRE((output[b, s0, i]) == (input[b, s1, i]));
            }
        }
    }
}


TEST_CASE("Roll function 4-dim", "[kernel::roll]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::roll<float> roll(gpu0);

    std::size_t bs = 2;
    std::size_t seq_len = 128;
    std::size_t kv_heads = 8;
    std::size_t head_dim = 64;

    auto input = shared_tensor(rand<float>({bs, seq_len, kv_heads, head_dim}));
    auto output = roll(input, 1, 1).get();

    REQUIRE(output.dim() == 4);
    REQUIRE(output.size(0) == bs);
    REQUIRE(output.size(1) == seq_len);
    REQUIRE(output.size(2) == kv_heads);
    REQUIRE(output.size(3) == head_dim);

    for (std::size_t b = 0; b < bs; b++) {
        for (std::size_t s0 = 0; s0 < seq_len; s0++) {
            std::size_t s1 = (s0 + 1) % seq_len;

            for (std::size_t i = 0; i < output.size(2); i++) {
                for (std::size_t j = 0; j < output.size(3); j++) {
                    REQUIRE((output[b, s0, i, j]) == (input[b, s1, i, j]));
                }
            }
        }
    }
}
