// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <istream>
#include <regex>
#include <vector>

#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/nn.h>
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
template <typename T, nn::layer Layer> class llama3_safetensor_serializer {
public:
    using value_type = nn::indirect_layer<Layer>;

    /// Creates a new instance of a layer serializer with the Llama3 options.
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

        adapt(layer);
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
        safetensor_document doc;

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

        for (auto it = document.begin(); it != document.end(); ++it) {
            auto st = *it;
            auto name = st.name();

            for (const auto& [re, replacement] : mapping) {
                name = std::regex_replace(name, re, replacement);
            }

            auto sizes = st.sizes();
            auto shape = std::vector<std::size_t>(sizes.begin(), sizes.end());

            doc.insert(safetensor(name, st.dtype(), shape, st.container_ptr()));
        }

        doc.insert("output.weight", "tok_embeddings.weight");
        return doc;
    }

    /// Perform permutation of the attention heads within Wq and Wk layers so that the order
    /// of elements as in the Meta's reference implementation.
    ///
    /// The Meta's reference implementation of attention layer differs from HuggingFace's
    /// implementation. Specifically, the attention heads are permuted. This layer adaptor
    /// performs a permutation to the shape expected in the reference implementation.
    ///
    /// The side-effect of this adaptor is increase of a memory required to launch the model,
    /// since after permutations weight tensors become discontiguous and their usage requires
    /// copying them.
    ///
    /// \param layer a layer to adapt to the reference implementation.
    void
    adapt(value_type& layer)
    {
        const std::vector<std::pair<std::regex, std::size_t>> permutations = {
            {std::regex(R"(layers\.(\d+)\.attention\.wk\.weight)"), _M_options.n_kv_heads()},
            {std::regex(R"(layers\.(\d+)\.attention\.wq\.weight)"), _M_options.n_heads()},
        };

        // Create a typed container, duplicate accessor attributes (strides, sizes, and offsets);
        // and use the same container. After permutations, override the original container with
        // the resulting container.
        auto permute_attention = [&](nn::named_parameter param) {
            for (auto& [re, n_heads] : permutations) {
                if (std::regex_match(param.path, re)) {
                    permute_attention_heads(param.ptr, n_heads);
                    break;
                }
            }
        };

        layer.apply(permute_attention);
    }

private:
    using tensor_pointer = std::shared_ptr<basic_tensor>;

    void
    permute_attention_heads(tensor_pointer ptr, std::size_t n_heads)
    {
        using container_type = hardware_memory_container<T>;
        using tensor_type = tensor<T, 2, container_type>;

        auto weight = shared_tensor(tensor_type());
        tensor_accessor::resize(*ptr, weight.accessor(), ptr->dimensions());
        weight.set_container(ptr->container_ptr());

        weight = permute_attention_heads(weight, n_heads);
        ptr->set_container(weight.container_ptr());
    }

    template <immutable_tensor2_t<T> Input> requires std::default_initializable<Input>
    auto
    permute_attention_heads(const Input& input, std::size_t n_heads)
    {
        std::size_t size = input.sizes().front();
        std::size_t attention_heads = size / n_heads / 2;

        // Transposition of the dimension 1 and 2 results in a discontiguous container layout,
        // therefore we need to copy elements row-wise (by the last dimension).
        //
        // This implementation performs transposition on-the-fly, inserts rows into the necessary
        // positions given the strides of the input and output tensors.
        tensor_accessor input_layout({n_heads, 2, attention_heads});
        tensor_accessor output_layout({n_heads, attention_heads, 2});

        std::vector<Input> tensors(size);
        for (std::size_t input_index = 0; input_index < size; input_index++) {
            auto i = input_index / input_layout.stride(0);
            auto rem = input_index % input_layout.stride(0);
            auto j = rem / input_layout.stride(1);
            auto k = rem % input_layout.stride(1);

            const auto output_index = i * output_layout.stride(0) + k * output_layout.stride(1) + j;
            tensors[output_index] = input.narrow(0, input_index, 1);
        }

        auto output = concatenate<T>(tensors.begin(), tensors.end(), 0, _M_accelerator);
        return output.get();
    }

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
template <typename T, nn::layer Layer> class llama3_qlora_safetensor_serializer {
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
