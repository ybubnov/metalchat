#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel/embedding.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Embedding batched", "[kernel::embedding]")
{
    metalchat::hardware_accelerator gpu0;
    kernel::embedding<float, 16, 128> emb(gpu0);

    auto input = shared_tensor(full<int32_t>({3, 4}, 0.0));
    input[0, 0] = 0;
    input[0, 1] = 1;
    input[0, 2] = 2;
    input[0, 3] = 3;
    input[1, 0] = 2;
    input[1, 1] = 4;
    input[1, 2] = 1;
    input[1, 3] = 0;
    input[2, 0] = 4;
    input[2, 1] = 3;
    input[2, 2] = 3;
    input[2, 3] = 2;

    auto weight = shared_tensor(rand<float>({5, 128256}));
    auto output = emb(input, weight).get();

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 3);
    REQUIRE(output.size(1) == 4);
    REQUIRE(output.size(2) == 128256);

    for (std::size_t i = 0; i < output.size(0); i++) {
        for (std::size_t j = 0; j < output.size(1); j++) {
            for (std::size_t k = 0; k < output.size(2); k++) {
                REQUIRE(output[i, j, k] == weight[input[i, j], k]);
            }
        }
    }
}


auto
compute_rope_freqs(
    std::size_t dim,
    std::size_t seq_len,
    float theta,
    std::size_t start_pos,
    hardware_accelerator& gpu
)
{
    auto f_cos = empty<float>({seq_len, dim / 2}, gpu.get_allocator());
    auto f_sin = empty<float>({seq_len, dim / 2}, gpu.get_allocator());

    std::vector<float> freqs(dim / 2);
    for (std::size_t j = 0; j < freqs.size(); j++) {
        freqs[j] = 1.0f / std::powf(theta, 2.0 * j / dim);
    }

    // scale_freqs(freqs);

    std::size_t end = start_pos + seq_len;
    for (std::size_t i = start_pos; i < end; i++) {
        std::size_t ii = i - start_pos;

        for (std::size_t j = 0; j < freqs.size(); j++) {
            float angle = float(i) * freqs[j];

            f_cos[ii, j] = std::cos(angle);
            f_sin[ii, j] = std::sin(angle);
        }
    }

    return std::make_tuple(std::move(f_cos), std::move(f_sin));
}


TEST_CASE("RoPE Frequencies", "[kernel::rope_freqs]")
{
    metalchat::hardware_accelerator gpu0;

    auto seq_len = std::size_t(1024);
    auto dim = 64;
    auto theta = float(500000.0);

    kernel::rope_freqs<float> rope_freqs(dim, seq_len, 500000.0, gpu0);

    auto [cos_f, sin_f] = rope_freqs(/*start_pos=*/100);
    auto freqs_cos = cos_f.get();
    auto freqs_sin = sin_f.get();

    auto [true_cos, true_sin] = compute_rope_freqs(dim, seq_len, theta, 100, gpu0);

    REQUIRE(true_cos.dim() == freqs_cos.dim());
    REQUIRE(true_sin.dim() == freqs_sin.dim());

    REQUIRE(true_cos.size(0) == freqs_cos.size(0));
    REQUIRE(true_cos.size(1) == freqs_cos.size(1));
    REQUIRE(true_sin.size(0) == freqs_sin.size(0));
    REQUIRE(true_sin.size(1) == freqs_sin.size(1));

    for (std::size_t i = 0; i < true_cos.size(0); i++) {
        for (std::size_t j = 0; j < true_cos.size(1); j++) {
            REQUIRE_THAT((true_cos[i, j]), Catch::Matchers::WithinAbs(freqs_cos[i, j], 0.0001));
            REQUIRE_THAT((true_sin[i, j]), Catch::Matchers::WithinAbs(freqs_sin[i, j], 0.0001));
        }
    }
}
