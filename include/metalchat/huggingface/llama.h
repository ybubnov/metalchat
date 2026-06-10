// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <istream>
#include <regex>
#include <vector>

#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/nn/llama.h>
#include <metalchat/quantization.h>
#include <metalchat/reference.h>
#include <metalchat/safetensor.h>
#include <metalchat/tensor/basic.h>
#include <metalchat/tensor/shared.h>
#include <metalchat/transformer.h>


namespace metalchat {
namespace huggingface {


/// Llama3 options serializer for configuration distributed through HuggingFace repository.
///
/// The HuggingFace configuration format differs from the format of reference Llama3
/// implementation, so this serializer performs necessary mapping of JSON fields internally.
struct llama3_options_serializer {
    using value_type = nn::llama3_options;

    nn::llama3_options
    load(std::istream& is) const;

    void
    save(std::ostream& os, const nn::llama3_options& options) const;
};


/// Safetensor serializer for Llama3 model distributed through HuggingFace repository.
///
/// \tparam T a type of the attention weights (Wq, Wk).
/// \tparam Layer a Llama3 implementation layer.
template <typename T, nn::mutable_layer Layer> class llama3_safetensor_serializer {
public:
    using value_type = nn::indirect_layer<Layer>;

    /// Creates a new instance of a layer serializer with Llama3 options.
    llama3_safetensor_serializer(
        const nn::llama3_options& options, hardware_accelerator& accelerator
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
        // TODO: remove tok_embeddings.weight tensor, permute back layers?
        document.save(layer);
    }

    /// Adapt HuggingFace's safetensor to the Meta Llama3 reference implementation.
    ///
    /// The Meta's reference implementation uses layer naming principle that differs from
    /// the layer naming in HuggingFace. This method performs re-mapping of layer names.
    ///
    /// The method creates a new safetensor document that carries shallow copies of the
    /// containers from the specified document. Original object could be safely destroyed
    /// thereafter.
    ///
    /// \param document Llama3 model weights distributed through HuggingFace.
    safetensor_document
    adapt(const safetensor_document& document) const
    {
        const std::vector<std::pair<std::regex, std::string>> mapping = {
            {std::regex(R"(model\.(layers\.\d+)\.input_layernorm)"), "$1.attention_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.post_attention_layernorm)"), "$1.ffn_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.mlp\.gate_proj)"), "$1.feed_forward.w1"},
            {std::regex(R"(model\.(layers\.\d+)\.mlp\.down_proj)"), "$1.feed_forward.w2"},
            {std::regex(R"(model\.(layers\.\d+)\.mlp\.up_proj)"), "$1.feed_forward.w3"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.q_proj)"), "$1.attention.wq"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.k_proj)"), "$1.attention.wk"},
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
    nn::llama3_options _M_options;
    hardware_accelerator _M_accelerator;
};


/// Llama3 serializer with QLoRa quantization distributed through HuggingFace repo.
///
/// The serializer replaces linear layers with \ref quantization::lora_linear, embedding layer
/// with \ref quantization::lora_embedding and output layer with \ref quantization::linear.
///
/// These layers perform on-the-flyte dequantization, which increases compute time due to the need
/// to reconstruct original weights.
///
/// \tparam T a type of the dequantized weights.
/// \tparam Layer a Llama3 implementation layer.
template <typename T, nn::mutable_layer Layer> class llama3_qlora_safetensor_serializer {
public:
    using value_type = nn::indirect_layer<Layer>;

    llama3_qlora_safetensor_serializer(
        const nn::llama3_options& options, hardware_accelerator& accelerator
    )
    : _M_options(options),
      _M_accelerator(accelerator)
    {}

    value_type
    load(safetensor_document& document)
    {
        value_type layer(_M_options, _M_accelerator);
        adapt(layer);
        document.load(layer);
        return layer;
    }

    void
    save(safetensor_document& document, value_type layer)
    {
        document.save(layer);
    }

    /// Adapt the Llama3 model before loading weights. Performs in-place replacement of linear
    /// and embedding layers.
    void
    adapt(value_type layer)
    {
        auto is_basic_linear = nn::layer_common_with<nn::basic_linear<T>>();
        auto is_basic_embedding = nn::layer_common_with<nn::basic_embedding<T>>();
        auto is_output = nn::layer_match_all(is_basic_linear, nn::layer_match_name("output"));

        using QLinear = quantization::linear<T>;
        using QLoraEmbedding = quantization::lora_embedding<T>;
        using QLoraLinear = quantization::lora_linear<T>;

        nn::indirect_layer<QLinear> linear(_M_accelerator);
        nn::indirect_layer<QLoraEmbedding> embedding(_M_accelerator);

        nn::replace_layer(layer, is_basic_linear, [&] {
            return nn::indirect_layer<QLoraLinear>(2.0, 32, _M_accelerator);
        });
        nn::replace_layer(layer, is_basic_embedding, embedding);
        nn::replace_layer(layer, is_output, linear);
    }

private:
    nn::llama3_options _M_options;
    hardware_accelerator _M_accelerator;
};


/// Llama3 tokenizer loader for a model distributed through HuggingFace repository.
///
/// The Meta's reference implementation distributes the tokenizer model in a tiktoken format,
/// while HuggingFace maintain it's own JSON-based tokenizer format. This loader performs
/// adaptation of HuggingFace JSON format into the MetalChat implementation of the tokenizer.
///
/// Note, it does not implement all features available in HuggingFace's tokenizer format, rather
/// queries necessary tokens of data from the `tokenizer.json` file in order to replicate the
/// original tiktoken format.
struct llama3_tokenizer_loader {
    using type = text::byte_pair_encoder<text::regexp>;

    /// Load the tokenizer from the specified input stream.
    ///
    /// \param is An input stream containing a JSON-encoded tokenizer model (HuggingFace format).
    type
    load(std::istream& is) const;

    /// Load the tokenizer from the specified local file.
    ///
    /// \param p A path to the JSON-encoded tokenizer model (HuggingFace format).
    type
    load(const std::filesystem::path& p) const;
};


template <contiguous_container Container> struct llama3_traits {
    using value_type = Container::value_type;
    using container_type = Container;

    using layer_type = nn::llama3<value_type, Container>;
    using layer_serializer = llama3_safetensor_serializer<value_type, layer_type>;

    using options_type = nn::llama3_options;
    using options_serializer = llama3_options_serializer;

    using tokenizer_type = text::byte_pair_encoder<text::regexp>;
    using tokenizer_loader = llama3_tokenizer_loader;

    static constexpr std::string_view tokenizer_location = "tokenizer.json";
    static constexpr std::string_view options_location = "config.json";
    static constexpr std::string_view transformer_location = "model.safetensors";
};


// TODO: Replace llama3_options with llama3_qlora_options.
template <contiguous_container Container> struct llama3_qlora_traits {
    using value_type = Container::value_type;
    using container_type = Container;

    using layer_type = nn::llama3<value_type, Container>;
    using layer_serializer = llama3_qlora_safetensor_serializer<value_type, layer_type>;

    using options_type = nn::llama3_options;
    using options_serializer = llama3_options_serializer;

    using tokenizer_type = text::byte_pair_encoder<text::regexp>;
    using tokenizer_loader = reference::llama3_tokenizer_loader;
};


using llama3 = llama3_traits<hardware_memory_container<bf16>>;
using llama3_qlora = llama3_qlora_traits<hardware_memory_container<bf16>>;


} // namespace huggingface
} // namespace metalchat
