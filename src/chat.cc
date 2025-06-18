#include <metalchat/chat.h>


namespace metalchat {


polymorphic_language_transformer::polymorphic_language_transformer(
    std::shared_ptr<basic_language_transformer> ptr
)
: _m_transformer(ptr)
{}


polymorphic_language_transformer::output_tensor
polymorphic_language_transformer::transform(input_tensor input, std::size_t start_pos)
{
    return _m_transformer->transform(input, start_pos);
}


hardware_accelerator
polymorphic_language_transformer::get_accelerator()
{
    return _m_transformer->get_accelerator();
}

void
polymorphic_chat::send(const basic_message& message)
{
    _m_chat.send(message);
}


basic_message
polymorphic_chat::receive()
{
    return _m_chat.receive();
}


std::string
polymorphic_chat::receive_text()
{
    return _m_chat.receive_text();
}


llama3_options
llama3_options::head_dim(std::optional<std::size_t> head_dim) const noexcept
{
    llama3_options o = *this;
    if (head_dim.has_value()) {
        o.set_head_dim(head_dim.value());
    }
    return o;
}


llama3_options
llama3_options::n_heads(std::optional<std::size_t> n_heads) const noexcept
{
    llama3_options o = *this;
    if (n_heads.has_value()) {
        o.set_n_heads(n_heads.value());
    }
    return o;
}


llama3_options
llama3_options::n_kv_heads(std::optional<std::size_t> n_kv_heads) const noexcept
{
    llama3_options o = *this;
    if (n_kv_heads.has_value()) {
        o.set_n_kv_heads(n_kv_heads.value());
    }
    return o;
}


llama3_options
llama3_options::n_layers(std::optional<std::size_t> n_layers) const noexcept
{
    llama3_options o = *this;
    if (n_layers.has_value()) {
        o.set_n_layers(n_layers.value());
    }
    return o;
}


llama3_options
llama3_options::max_seq_len(std::optional<std::size_t> max_seq_len) const noexcept
{
    llama3_options o = *this;
    if (max_seq_len.has_value()) {
        o.set_max_seq_len(max_seq_len.value());
    }
    return o;
}


llama3_options
llama3_options::heap_size(std::optional<std::size_t> heap_size) const noexcept
{
    llama3_options o = *this;
    if (heap_size.has_value()) {
        o.set_heap_size(heap_size.value());
    }
    return o;
}


llama3_options
llama3_options::rope_theta(std::optional<float> rope_theta) const noexcept
{
    llama3_options o = *this;
    if (rope_theta.has_value()) {
        o.set_rope_theta(rope_theta.value());
    }
    return o;
}


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


polymorphic_chat
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
    auto agent = polymorphic_chat(std::move(transformer), bpe);

    return agent;
}


polymorphic_chat
construct_llama3_1b(
    const std::string& weights_path,
    const std::string& tokens_path,
    std::optional<llama3_options> options
)
{
    auto weights_fs_path = std::filesystem::path(weights_path);
    auto tokens_fs_path = std::filesystem::path(tokens_path);

    return construct_llama3_1b(weights_fs_path, tokens_fs_path, options);
}


} // namespace metalchat
