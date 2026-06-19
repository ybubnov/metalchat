// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/nn/gemma.h>
#include <metalchat/safetensor.h>
#include <metalchat/text.h>


namespace metalchat {
namespace huggingface {


/// Gemma3 options serializer for configurations distributed through HuggingFace repository.
struct gemma3_options_serializer {
    using value_type = nn::gemma3_options;

    nn::gemma3_options
    load(std::istream& is) const;

    void
    save(std::ostream& os, const nn::gemma3_options& options) const;
};


template <typename T, nn::mutable_layer Layer> class gemma3_safetensor_serializer {
public:
    using value_type = nn::indirect_layer<Layer>;

    /// Creates a new instance of a layer serializer with Gemma3 options.
    gemma3_safetensor_serializer(
        const nn::gemma3_options& options, hardware_accelerator& accelerator
    )
    : _M_options(options),
      _M_accelerator(accelerator)
    {}

    value_type
    load(safetensor_document& document)
    {
        value_type layer(_M_options, _M_accelerator);

        auto doc = adapt(document);
        doc.load(layer);

        return layer;
    }

    void
    save(safetensor_document& document, value_type layer)
    {
        document.save(layer);
    }

    safetensor_document
    adapt(const safetensor_document& document) const
    {
        const std::vector<std::pair<std::regex, std::string>> mapping = {
            {std::regex(R"(model\.(layers\.\d+)\.input_layernorm)"), "$1.attention_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.post_attention_layernorm)"),
             "$1.attention_post_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.pre_feedforward_layernorm)"), "$1.ffn_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.post_feedforward_layernorm)"), "$1.ffn_post_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.mlp\.gate_proj)"), "$1.feed_forward.w1"},
            {std::regex(R"(model\.(layers\.\d+)\.mlp\.down_proj)"), "$1.feed_forward.w2"},
            {std::regex(R"(model\.(layers\.\d+)\.mlp\.up_proj)"), "$1.feed_forward.w3"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.q_proj)"), "$1.attention.wq"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.q_norm)"), "$1.attention.q_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.k_proj)"), "$1.attention.wk"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.k_norm)"), "$1.attention.k_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.v_proj)"), "$1.attention.wv"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.o_proj)"), "$1.attention.wo"},
            {std::regex(R"(model.norm)"), "norm"},
            {std::regex(R"(model.embed_tokens)"), "tok_embeddings"},
        };

        auto doc = document.rename(mapping.begin(), mapping.end());
        doc.insert("output.weight", "tok_embeddings.weight");

        return doc;
    }

private:
    nn::gemma3_options _M_options;
    hardware_accelerator _M_accelerator;
};


struct gemma3_tokenizer_loader {
    using type = text::sentence_piece;

    type
    load(std::istream& is) const;

    type
    load(const std::filesystem::path& p) const;
};


template <contiguous_container Container> struct gemma3_traits {
    using value_type = Container::value_type;
    using container_type = Container;

    using layer_type = nn::gemma3<value_type, Container>;
    using layer_serializer = gemma3_safetensor_serializer<value_type, layer_type>;

    using options_type = nn::gemma3_options;
    using options_serializer = gemma3_options_serializer;

    using tokenizer_type = text::sentence_piece;
    using tokenizer_loader = gemma3_tokenizer_loader;
};


using gemma3 = gemma3_traits<hardware_memory_container<bf16>>;


} // namespace huggingface
} // namespace metalchat
