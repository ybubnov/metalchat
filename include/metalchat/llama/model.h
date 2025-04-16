#pragma once

#include <format>
#include <optional>
#include <vector>

#include <metalchat/container.h>
#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/functional.h>
#include <metalchat/llama/transformer.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/safetensor.h>


namespace metalchat {
namespace llama {


template <typename T, contiguous_container Container> class model {
private:
    nn::embedding<T, hardware_memory_container<T>> _m_embedding;
    nn::rmsnorm<T, Container> _m_norm;
    nn::linear<T, hardware_memory_container<T>> _m_output;

    std::vector<transformer<T, Container>> _m_layers;
    std::reference_wrapper<device> _m_device;

    auto
    create_additive_causal_mask(const std::size_t size) const
    {
        std::optional<shared_tensor<T, 2, hardware_memory_container<T>>> mask;

        if (size > 1) {
            const T infinity = T(std::numeric_limits<float>::infinity());
            auto m = full<T>({size, size}, -infinity, _m_device.get().get_allocator());
            triu(m);

            mask = std::make_optional(std::move(m));
        }

        return mask;
    }

public:
    model(model&&) = default;

    model(
        nn::embedding<T, hardware_memory_container<T>>&& embedding,
        nn::rmsnorm<T, Container>&& norm,
        nn::linear<T, hardware_memory_container<T>>&& output,
        std::vector<transformer<T, Container>>&& layers,
        device& gpu
    )
    : _m_embedding(std::move(embedding)),
      _m_norm(std::move(norm)),
      _m_output(std::move(output)),
      _m_layers(std::move(layers)),
      _m_device(gpu)
    {}

    template <immutable_tensor2_t<int32_t> Input>
    auto
    operator()(Input input, std::size_t start_pos = 0)
    {
        const auto mask = create_additive_causal_mask(input.size(1));
        auto x = _m_embedding(input);

        for (auto& layer : _m_layers) {
            x = layer(x, mask, start_pos);
        }

        auto output = _m_norm(x);

        using s = indexing::slice;
        auto seqlen = output.size(1);
        output = output[s(), s(seqlen - 1, seqlen), s()];

        return _m_output(output);
    }
};


template <typename T>
auto
make_model(const metalchat::safetensor_file& tensors, device& gpu, std::size_t nlayers = 16)
{
    using allocator_type = rebind_hardware_allocator<T, device::allocator_type>;
    using container_type = allocator_type::container_type;

    auto alloc0 = allocator_type(gpu.get_allocator());
    auto alloc1 = hardware_nocopy_allocator(alloc0, gpu.get_hardware_device());
    auto alloc = hardware_resident_allocator(alloc1, gpu.get_hardware_device());


    auto input = shared_tensor(tensors["tok_embeddings.weight"].as<2>(alloc));
    nn::embedding embedding(input, gpu);
    nn::rmsnorm norm(tensors["norm.weight"].as<1>(alloc), gpu);

    auto options = llama::attention_options{
        .head_dim = 64,
        .n_heads = 32,
        .n_kv_heads = 8,
        .max_seq_len = 1024,
        .rope_theta = 500000.0
    };

    std::vector<llama::transformer<T, container_type>> layers;
    for (std::size_t i = 0; i < nlayers; i++) {
        const std::string layer_name = std::format("layers.{}.", i);

        llama::feed_forward ff(
            tensors[layer_name + "feed_forward.w1.weight"].as<2>(alloc),
            tensors[layer_name + "feed_forward.w2.weight"].as<2>(alloc),
            tensors[layer_name + "feed_forward.w3.weight"].as<2>(alloc), gpu
        );

        llama::attention attention(
            tensors[layer_name + "attention.wq.weight"].as<2>(alloc),
            tensors[layer_name + "attention.wk.weight"].as<2>(alloc),
            tensors[layer_name + "attention.wv.weight"].as<2>(alloc),
            tensors[layer_name + "attention.wo.weight"].as<2>(alloc), options, gpu
        );

        nn::rmsnorm attention_norm(tensors[layer_name + "attention_norm.weight"].as<1>(alloc), gpu);
        nn::rmsnorm ff_norm(tensors[layer_name + "ffn_norm.weight"].as<1>(alloc), gpu);

        llama::transformer<T, container_type> transformer(
            std::move(attention), std::move(attention_norm), std::move(ff), std::move(ff_norm), gpu
        );

        layers.push_back(std::move(transformer));
    }

    alloc.wire_memory();

    // Re-use the same linear module (or even a tensor) in both, embeddings computation
    // and in output layer computation.
    return llama::model(
        std::move(embedding), std::move(norm), nn::linear(input, gpu), std::move(layers), gpu
    );
}


} // namespace llama
} // namespace metalchat
