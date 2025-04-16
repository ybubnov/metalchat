#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel/multinomial.h>
#include <metalchat/tensor.h>


using namespace metalchat;


TEST_CASE("Multinomial generator", "[kernel::multinomial]")
{
    metalchat::device gpu0("metalchat.metallib");
    metalchat::multinomial<float> m(gpu0);

    auto input = shared_tensor(empty<float>({4, 5}));

    // Experiment probabilities.
    auto experiment_probs = std::vector<float>({0.1, 0.2, 0.1, 0.4, 0.2});

    // Cumulative probabilities.
    auto input_probs = std::vector<float>({0.1, 0.3, 0.4, 0.8, 1.0});
    for (std::size_t i = 0; i < input.size(0); i++) {
        std::copy(input_probs.begin(), input_probs.end(), input[i].begin());
    }

    std::size_t sample_size = 8192;
    auto output = m(input, sample_size).get();

    for (std::size_t i = 0; i < output.size(0); i++) {
        auto output_probs = std::vector<float>(input_probs.size(), 0.0f);
        for (std::size_t j = 0; j < output.size(1); j++) {
            output_probs[output[i, j]] += 1.0f;
        }

        for (std::size_t k = 0; k < input_probs.size(); k++) {
            output_probs[k] /= float(sample_size);
            CHECK_THAT(output_probs[k], Catch::Matchers::WithinAbs(experiment_probs[k], 0.02));
        }
    }
}
