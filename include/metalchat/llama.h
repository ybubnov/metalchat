#pragma once

#include <format>
#include <vector>

#include <metalchat/dtype.h>
#include <metalchat/llama/model.h>
#include <metalchat/safetensor.h>


namespace metalchat {


template <typename T>
auto
make_llama(const metalchat::safetensor_file& tensors, device& device, std::size_t nlayers = 16)
{
    using container_type = weak_ref<T>;

    nn::embedding embedding(tensors["model.embed_tokens.weight"].as<T, 2>(), device);
    nn::rmsnorm norm(tensors["model.norm.weight"].as<T, 1>(), device);

    std::cout << embedding << std::endl;

    auto options = llama::attention_options{
        .head_dim = 64,
        .n_heads = 32,
        .n_kv_heads = 8,
        .max_seq_len = 1024,
        .rope_theta = 500000.0
    };

    std::vector<llama::transformer<T, container_type>> layers;
    for (std::size_t i = 0; i < nlayers; i++) {
        const std::string layer_name = std::format("model.layers.{}.", i);

        llama::feed_forward ff(
            tensors[layer_name + "mlp.gate_proj.weight"].as<T, 2>(),
            tensors[layer_name + "mlp.down_proj.weight"].as<T, 2>(),
            tensors[layer_name + "mlp.up_proj.weight"].as<T, 2>(), device
        );

        llama::attention attention(
            tensors[layer_name + "self_attn.q_proj.weight"].as<T, 2>(),
            tensors[layer_name + "self_attn.k_proj.weight"].as<T, 2>(),
            tensors[layer_name + "self_attn.v_proj.weight"].as<T, 2>(),
            tensors[layer_name + "self_attn.o_proj.weight"].as<T, 2>(), options, device
        );

        nn::rmsnorm attention_norm(
            tensors[layer_name + "input_layernorm.weight"].as<T, 1>(), device
        );
        nn::rmsnorm ff_norm(
            tensors[layer_name + "post_attention_layernorm.weight"].as<T, 1>(), device
        );

        llama::transformer<T, container_type> transformer(
            std::move(attention), std::move(attention_norm), std::move(ff), std::move(ff_norm),
            device
        );

        layers.push_back(std::move(transformer));
    }

    return llama::model(
        std::move(embedding), std::move(norm),
        nn::linear(tensors["model.embed_tokens.weight"].as<T, 2>(), device), std::move(layers)
    );
}


} // namespace metalchat
