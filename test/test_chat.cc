#include <catch2/catch_test_macros.hpp>

#include <metalchat/allocator.h>
#include <metalchat/chat.h>
#include <metalchat/nn.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Test chat", "[llama]")
{
    metalchat::byte_pair_encoder bpe("../Llama-3.2-1B-Instruct/original/tokenizer.model");
    metalchat::hardware_accelerator gpu0;

    metalchat::safetensor_file tensors("../llama32.safetensors");

    auto alloc1 = hardware_nocopy_allocator(gpu0.get_allocator(), gpu0.get_hardware_device());
    auto alloc2 = hardware_resident_allocator(alloc1, gpu0.get_hardware_device());

    gpu0.set_allocator(std::move(alloc2));
    auto options = nn::attention_options{
        .head_dim = 64,
        .n_heads = 32,
        .n_kv_heads = 8,
        .max_seq_len = 32,
        .rope_theta = 500000.0
    };

    nn::llama<bf16> m(16, options, gpu0);
    m.initialize(tensors, make_rebind_allocator<bf16>(gpu0.get_allocator()));

    auto transformer = language_transformer(std::move(m));
    auto agent = chat(std::move(transformer), std::move(bpe));

    agent.send(basic_message("system", "You are a helpful assistant"));
    agent.send(basic_message("user", "What is the capital of France?"));
    std::cout << agent.receive_text() << std::endl;

    agent.send(basic_message("user", "what is the capital of Belgium?"));
    std::cout << agent.receive_text() << std::endl;
}
