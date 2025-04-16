#include <catch2/catch_test_macros.hpp>
#include <chrono>


#include <metalchat/bpe.h>
#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/kernel/sort.h>
#include <metalchat/llama.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Test make model", "[llama]")
{
    metalchat::bpe bpe("../Llama-3.2-1B/original/tokenizer.model");
    metalchat::device gpu0("metalchat.metallib", 16);

    // Load tensors in lambda, so that all resources are cleaned up after the load.
    auto m = [&] -> auto {
        safetensor_file tensors("../llama32.safetensors");
        return make_llama<bf16>(tensors, gpu0);
    }();

    auto input_text = std::string("I have a dog called");

    std::vector<int32_t> ids;
    bpe.encode(special_token::begin_text, ids);
    bpe.encode(input_text, ids);

    auto input0 = shared_tensor(to_tensor<int32_t>({1, ids.size()}, ids.begin(), ids.end()));
    auto logit0 = m(input0, 0);
    auto id = fn::top_p(logit0.flatten<2>(), bf16(0.6f), bf16(0.9), gpu0).get();

    std::cout << input_text;
    std::cout << bpe.decode(id[0, 0]);

    for (std::size_t i = input0.size(1); i < 52; i++) {
        // const auto start{std::chrono::steady_clock::now()};

        auto logits = m(id, i).flatten<2>();
        id = fn::top_p(logits, bf16(0.6f), bf16(0.9f), gpu0).get();

        // const auto finish{std::chrono::steady_clock::now()};
        // const std::chrono::duration<double> elapsed_seconds{finish - start};
        // std::cout << "t=" << elapsed_seconds << std::endl;

        std::cout << bpe.decode(id[0, 0]) << std::flush;
    }
    std::cout << std::endl;
}
