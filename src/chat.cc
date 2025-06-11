#include <metalchat/chat.h>


namespace metalchat {


llama3_options
default_llama3_1b_options()
{
    return llama3_options()
        .head_dim(64)
        .n_heads(32)
        .n_kv_heads(8)
        .n_layers(16)
        .max_seq_len(1024)
        .rope_theta(500000.0f)
        .heap_size(std::size_t(512) * 1024 * 1024);
}


llama3_traits<dtype::bf16>::type
construct_llama3_1b(
    const std::filesystem::path& weights_path,
    const std::filesystem::path& tokens_path,
    std::optional<llama3_options> options_
)
{
    metalchat::bpe bpe(tokens_path);
    metalchat::safetensor_file weights(weights_path);

    metalchat::hardware_accelerator gpu0;
    auto alloc1 = hardware_nocopy_allocator(gpu0.get_allocator(), gpu0.get_metal_device());
    auto alloc2 = hardware_resident_allocator(alloc1, gpu0.get_metal_device());

    gpu0.set_allocator(std::move(alloc2));

    auto options = options_.value_or(default_llama3_1b_options());
    auto attention_options = nn::attention_options{
        .head_dim = options.head_dim(),
        .n_heads = options.n_heads(),
        .n_kv_heads = options.n_kv_heads(),
        .max_seq_len = options.max_seq_len(),
        .rope_theta = options.rope_theta()
    };

    nn::llama<dtype::bf16> m(options.n_layers(), attention_options, gpu0);
    m.initialize(weights, make_rebind_allocator<dtype::bf16>(gpu0.get_allocator()));

    auto alloc3 = hardware_heap_allocator<void>(gpu0.get_metal_device(), options.heap_size());
    auto alloc4 = hardware_nocopy_allocator(alloc3, gpu0.get_metal_device());

    gpu0.set_allocator(std::move(alloc4));

    auto transformer = language_transformer(std::move(m));
    return chat(std::move(transformer), bpe);
}


} // namespace metalchat
