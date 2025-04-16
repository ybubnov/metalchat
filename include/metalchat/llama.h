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

    nn::embedding embedding(tensors["tok_embeddings.weight"].as<T, 2>(device), device);
    nn::rmsnorm norm(tensors["norm.weight"].as<T, 1>(), device);

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
            tensors[layer_name + "feed_forward.w1.weight"].as<T, 2>(device),
            tensors[layer_name + "feed_forward.w2.weight"].as<T, 2>(device),
            tensors[layer_name + "feed_forward.w3.weight"].as<T, 2>(device), device
        );

        llama::attention attention(
            tensors[layer_name + "attention.wq.weight"].as<T, 2>(device),
            tensors[layer_name + "attention.wk.weight"].as<T, 2>(device),
            tensors[layer_name + "attention.wv.weight"].as<T, 2>(device),
            tensors[layer_name + "attention.wo.weight"].as<T, 2>(device), options, device
        );

        nn::rmsnorm attention_norm(
            tensors[layer_name + "attention_norm.weight"].as<T, 1>(), device
        );
        nn::rmsnorm ff_norm(tensors[layer_name + "ffn_norm.weight"].as<T, 1>(), device);

        llama::transformer<T, container_type> transformer(
            std::move(attention), std::move(attention_norm), std::move(ff), std::move(ff_norm),
            device
        );

        layers.push_back(std::move(transformer));
    }

    // Re-use the same linear module (or even a tensor) in both, embeddings computation
    // and in output layer computation.
    return llama::model(
        std::move(embedding), std::move(norm),
        nn::linear(tensors["tok_embeddings.weight"].as<T, 2>(device), device), std::move(layers)
    );
}


} // namespace metalchat
