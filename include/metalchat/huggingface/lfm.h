// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <memory>
#include <string_view>

#include <metalchat/format.h>
#include <metalchat/huggingface/llama.h>
#include <metalchat/nn/lfm.h>
#include <metalchat/safetensor.h>
#include <metalchat/text.h>


namespace metalchat {
namespace huggingface {


using namespace std::literals;


/// LFM2 options serializer for configurations distributed through HuggingFace repository.
struct lfm2_options_serializer {
    using value_type = nn::lfm2_options;

    nn::lfm2_options
    load(std::istream& is) const;

    void
    save(std::ostream& os, const nn::lfm2_options& options) const;
};


template <nn::mutable_layer Layer> class lfm2_safetensor_serializer {
public:
    using value_type = nn::indirect_layer<Layer>;

    void
    load(const safetensor_document& document, value_type& layer) const
    {
        auto doc = adapt(document);
        doc.load(layer);
    }

    void
    save(safetensor_document& document, const value_type& layer) const
    {
        document.save(layer);
    }

    safetensor_document
    adapt(const safetensor_document& document) const
    {
        const std::vector<std::pair<std::regex, std::string>> mapping = {
            {std::regex(R"(model\.(layers\.\d+)\.conv\.in_proj)"), "$1.attention.wq"},
            {std::regex(R"(model\.(layers\.\d+)\.conv\.out_proj)"), "$1.attention.wo"},
            {std::regex(R"(model\.(layers\.\d+)\.conv\.conv)"), "$1.attention.conv"},
            {std::regex(R"(model\.(layers\.\d+)\.operator_norm)"), "$1.attention_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.q_proj)"), "$1.attention.wq"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.q_layernorm)"), "$1.attention.q_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.k_proj)"), "$1.attention.wk"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.k_layernorm)"), "$1.attention.k_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.v_proj)"), "$1.attention.wv"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.out_proj)"), "$1.attention.wo"},
            {std::regex(R"(model\.(layers\.\d+))"), "$1"},
            {std::regex(R"(model.embedding_norm)"), "norm"},
            {std::regex(R"(model.embed_tokens)"), "tok_embeddings"}
        };

        auto doc = document.rename(mapping.begin(), mapping.end());
        doc.insert("output.weight", "tok_embeddings.weight");

        return doc;
    }
};


template <contiguous_container Container> struct lfm2_traits {
    using value_type = Container::value_type;
    using container_type = Container;

    using layer_type = nn::lfm2<value_type, Container>;
    using layer_serializer = lfm2_safetensor_serializer<layer_type>;

    using options_type = nn::lfm2_options;
    using options_serializer = lfm2_options_serializer;

    using tokenizer_type = text::byte_pair_encoder<char>;
    using tokenizer_loader = llama3_tokenizer_loader;

    static constexpr std::string_view tokenizer_location = "tokenizer.json";
    static constexpr std::string_view options_location = "config.json";
    static constexpr std::string_view transformer_location = "model.safetensors";
};


using lfm2 = lfm2_traits<hardware_memory_container<bf16>>;


} // namespace huggingface
} // namespace metalchat
