#include <chrono>

#include <catch2/catch_test_macros.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/bpe.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/kernel/sort.h>
#include <metalchat/nn/llama.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Test make model", "[llama]")
{
    metalchat::byte_pair_encoder bpe("../Llama-3.2-1B/original/tokenizer.model");
    metalchat::hardware_accelerator gpu0;

    auto alloc1 = hardware_nocopy_allocator(gpu0.get_allocator(), gpu0.get_metal_device());
    auto alloc2 = hardware_resident_allocator(alloc1, gpu0.get_metal_device());

    gpu0.set_allocator(std::move(alloc2));

    auto options = nn::default_llama3_1b_options().max_seq_len(16);
    nn::llama3<bf16> m(options, gpu0);

    auto alloc = make_rebind_allocator<bf16>(gpu0.get_allocator());
    safetensor_document::load("../llama32.safetensors", m, alloc);

    auto heap_size = std::size_t(512) * 1024 * 1024;
    auto alloc3 = hardware_heap_allocator<void>(gpu0.get_metal_device(), heap_size);
    auto alloc4 = hardware_nocopy_allocator(alloc3, gpu0.get_metal_device());
    gpu0.set_allocator(std::move(alloc4));

    auto input_text = std::string("I have a dog called");

    std::vector<int32_t> ids;
    bpe.encode(special_token::begin_text, std::back_inserter(ids));
    bpe.encode(input_text, std::back_inserter(ids));

    auto input0 = shared_tensor(to_tensor<int32_t>({1, ids.size()}, ids.begin(), ids.end()));
    auto logit0 = m(input0, 0);
    auto id = top_p(logit0.flatten<2>(), bf16(0.6f), bf16(0.9), gpu0); //.get();

    std::cout << input_text;
    std::cout << bpe.decode(id.get()[0, 0]);
    std::vector<future_tensor<int32_t, 2>> outputs;
    // outputs.push_back(id);

    for (std::size_t i = input0.size(1); i < 64; i++) {
        // const auto start{std::chrono::steady_clock::now()};

        auto logits = m(id, i).flatten<2>();
        id = top_p(logits, bf16(0.6f), bf16(0.9f), gpu0); //.get();

        // const auto finish{std::chrono::steady_clock::now()};
        // const std::chrono::duration<double> elapsed_seconds{finish - start};
        // std::cout << "t=" << elapsed_seconds << std::endl;

        std::cout << bpe.decode(id.get()[0, 0]) << std::flush;
        // outputs.push_back(id);
    }
    for (auto& id : outputs) {
        std::cout << bpe.decode(id.get()[0, 0]) << std::flush;
    }
    std::cout << std::endl;
}
