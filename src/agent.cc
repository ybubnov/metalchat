#include <metalchat/agent.h>

#include "metal_impl.h"


namespace metalchat {


agent
make_llama3(
    const std::filesystem::path& weights_path,
    const std::filesystem::path& tokens_path,
    std::optional<nn::llama3_options> options_
)
{
    metalchat::bpe bpe(tokens_path);
    metalchat::hardware_accelerator gpu0;

    auto weights_file = std::make_shared<basic_memfile>(weights_path);
    weights_file->declare_mapped();
    metalchat::safetensor_document weights(weights_file, safetensor_openmode::in);

    // Move memory file to a resident set, do not copy the underlying memory.
    //
    // On some metal devices, a single buffer could be limited by a "max_buffer_size",
    // for such devices, allocate multiple containers that tile the underlying memory.
    auto alloc0 = hardware_nocopy_allocator(gpu0.get_allocator(), gpu0.get_metal_device());
    auto alloc1 = hardware_resident_allocator(alloc0, gpu0.get_metal_device());
    auto alloc2 = paginated_allocator_adapter(alloc1, gpu0.max_buffer_size());
    auto containers = alloc2.allocate(weights.data(), weights.sizes());

    // Allocate all subsequent tensors from the buffer allocator, it will maintain
    // correct offset from the buffer and does not cause any container allocations.
    //
    // For all other remaining allocations, use a generic allocator from accelerator,
    // which will be also placed into resident memory.
    auto alloc3 = hardware_nocopy_allocator(gpu0.get_allocator(), gpu0.get_metal_device());
    auto alloc4 = hardware_resident_allocator(alloc3, gpu0.get_metal_device());
    auto alloc5 = hardware_buffer_allocator(alloc4, containers);
    auto alloc6 = hardware_aliasing_allocator(alloc5, weights_file);
    gpu0.set_allocator(std::move(alloc6));

    auto options = options_.value_or(nn::default_llama3_1b_options());

    using value_type = dtype::bf16;
    using container_type = hardware_memory_container<value_type>;
    using transformer_type = nn::llama3<value_type, container_type>;

    auto transformer = transformer_type(options, gpu0);

    auto alloc = make_rebind_allocator<value_type>(gpu0.get_allocator());
    weights.load(transformer, alloc);

    auto alloc7 = hardware_heap_allocator<void>(gpu0.get_metal_device(), options.heap_size());
    auto alloc8 = hardware_nocopy_allocator(alloc7, gpu0.get_metal_device());

    gpu0.set_allocator(std::move(alloc8));

    return agent(std::move(transformer), bpe);
}


agent
make_llama3_compact(
    const std::filesystem::path& weights_path,
    const std::filesystem::path& tokens_path,
    std::optional<nn::llama3_options> options_
)
{
    metalchat::bpe bpe(tokens_path);
    metalchat::hardware_accelerator gpu0(64);

    auto alloc0 = hardware_resident_allocator(gpu0.get_allocator(), gpu0.get_metal_device());
    gpu0.set_allocator(std::move(alloc0));

    auto weights_file = std::make_shared<basic_memfile>(weights_path);

    auto options = options_.value_or(nn::default_llama3_1b_options());

    using value_type = dtype::bf16;
    using container_type = filebuf_memory_container<value_type>;
    using transformer_type = nn::llama3<value_type, container_type>;

    auto transformer = transformer_type(options, gpu0);

    auto alloc = filebuf_memory_allocator<value_type>();
    auto weights = safetensor_document(weights_file, safetensor_openmode::in);
    weights.load(transformer, alloc);

    auto alloc3 = hardware_heap_allocator<void>(gpu0.get_metal_device(), options.heap_size());
    auto alloc4 = hardware_nocopy_allocator(alloc3, gpu0.get_metal_device());

    gpu0.set_allocator(std::move(alloc4));

    return agent(std::move(transformer), bpe);
}


agent
make_llama3(
    const std::string& weights_path,
    const std::string& tokens_path,
    std::optional<nn::llama3_options> options
)
{
    auto weights_fs_path = std::filesystem::path(weights_path);
    auto tokens_fs_path = std::filesystem::path(tokens_path);

    return make_llama3(weights_fs_path, tokens_fs_path, options);
}


agent
make_llama3_compact(
    const std::string& weights_path,
    const std::string& tokens_path,
    std::optional<nn::llama3_options> options
)
{
    auto weights_fs_path = std::filesystem::path(weights_path);
    auto tokens_fs_path = std::filesystem::path(tokens_path);

    return make_llama3_compact(weights_fs_path, tokens_fs_path, options);
}


} // namespace metalchat
