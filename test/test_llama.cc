#include <catch2/catch_test_macros.hpp>
#include <chrono>


#include <metalchat/accelerator.h>
#include <metalchat/bpe.h>
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
    metalchat::hardware_accelerator gpu0("metalchat.metallib", 16);

    hardware_heap_allocator<void> heap_alloc(
        gpu0.get_hardware_device(), std::size_t(512) * 1024 * 1024
    );
    gpu0.set_allocator(std::move(heap_alloc));

    safetensor_file tensors("../llama32.safetensors");
    auto m = llama::make_model<bf16>(tensors, gpu0);

    auto input_text = std::string("I have a dog called");

    std::vector<int32_t> ids;
    bpe.encode(special_token::begin_text, ids);
    bpe.encode(input_text, ids);

    auto input0 = shared_tensor(to_tensor<int32_t>({1, ids.size()}, ids.begin(), ids.end()));
    auto logit0 = m(input0, 0);
    auto id = top_p(logit0.flatten<2>(), bf16(0.6f), bf16(0.9), gpu0); //.get();

    std::cout << input_text;
    // std::cout << bpe.decode(id[0, 0]);
    std::vector<future_tensor<int32_t, 2>> outputs;
    outputs.push_back(id);

    for (std::size_t i = input0.size(1); i < 52; i++) {
        // const auto start{std::chrono::steady_clock::now()};

        auto logits = m(id, i).flatten<2>();
        id = top_p(logits, bf16(0.6f), bf16(0.9f), gpu0); //.get();

        // const auto finish{std::chrono::steady_clock::now()};
        // const std::chrono::duration<double> elapsed_seconds{finish - start};
        // std::cout << "t=" << elapsed_seconds << std::endl;

        // std::cout << bpe.decode(id[0, 0]) << std::flush;
        outputs.push_back(id);
    }
    for (auto& id : outputs) {
        std::cout << bpe.decode(id.get()[0, 0]) << std::flush;
    }
    std::cout << std::endl;
}
