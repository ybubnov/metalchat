#include <catch2/catch_test_macros.hpp>

#include <metalchat/allocator.h>
#include <metalchat/chat.h>
#include <metalchat/nn.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Test chat", "[llama]")
{
    metalchat::bpe bpe("../Llama-3.2-1B/original/tokenizer.model");
    metalchat::hardware_accelerator gpu0("metalchat.metallib", 16);
    metalchat::safetensor_file tensors("../llama32.safetensors");

    auto alloc1 = hardware_nocopy_allocator(gpu0.get_allocator(), gpu0.get_hardware_device());
    auto alloc2 = hardware_resident_allocator(alloc1, gpu0.get_hardware_device());

    gpu0.set_allocator(std::move(alloc2));
    auto options = nn::attention_options{
        .head_dim = 64,
        .n_heads = 32,
        .n_kv_heads = 8,
        .max_seq_len = 1024,
        .rope_theta = 500000.0
    };

    nn::llama<bf16> m(16, options, gpu0);
    m.initialize(tensors, make_rebind_allocator<bf16>(gpu0.get_allocator()));

    auto transformer = llama_transformer(std::move(m), gpu0);
    transformer.print();
}
