// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <istream>

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


/// Document adaptor (safetensors) for Llama3 model distributed through HuggingFace repository.
///
/// The Meta's reference implementation uses layer naming principle that differs from the layer
/// naming in HuggingFace. This implementation performs re-mapping of layer names.
///
/// This document adaptor creates a new safetensor document that carries shallow copies of the
/// containers from the specified document. Original object could be safely destroyed thereafter.
struct llama3_document_adaptor {
    safetensor_document
    adapt(const safetensor_document& document) const;
};


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


/// Layer adaptor for Llama3 model distributed through HuggingFace repository.
///
/// The Meta's reference implementation of attention layer differs from HuggingFace's
/// implementation. Specifically, the attention heads are permuted. This layer adaptor performs
/// a permutation to the shape expected in the reference implementation.
///
/// The side-effect of this adaptor is increase of a memory required to launch the model, since
/// after permutations weight tensors become discontiguous and their usage requires copying them.
///
/// \tparam T a type of the attention weights (Wq, Wk).
template <typename T> struct llama3_layer_adaptor {
    /// Creates a new instance of a layer adaptor with the Llama3 options.
    llama3_layer_adaptor(nn::llama3_options options)
    : _M_options(options)
    {}

    void
    adapt_pre(nn::indirect_layer<nn::basic_layer> layer) const
    {}

    /// Performs permutation of the attention heads within Wq and Wk layers so that the order
    /// of elements as in the Meta's reference implementation.
    ///
    /// \param layer a layer to adapt to the reference implementation.
    void
    adapt_post(nn::indirect_layer<nn::basic_layer> layer) const
    {
        const std::vector<std::pair<std::regex, std::size_t>> permutations = {
            {std::regex(R"(layers\.(\d+)\.attention\.wk\.weight)"), _M_options.n_kv_heads()},
            {std::regex(R"(layers\.(\d+)\.attention\.wq\.weight)"), _M_options.n_heads()},
        };

        auto& accelerator = layer.accelerator();

        // Create a typed container, duplicate accessor attributes (strides, sizes, and offsets);
        // and use the same container. After permutations, override the original container with
        // the resulting container.
        auto permute_attention = [&](nn::named_parameter param) {
            for (auto& [re, n_heads] : permutations) {
                if (std::regex_match(param.path, re)) {
                    permute_attention_heads(param.ptr, n_heads, accelerator);
                    break;
                }
            }
        };

        layer.apply(permute_attention);
    }

private:
    using tensor_pointer = std::shared_ptr<basic_tensor>;

    void
    permute_attention_heads(
        tensor_pointer ptr, std::size_t n_heads, hardware_accelerator& accelerator
    ) const
    {
        using container_type = hardware_memory_container<T>;
        using tensor_type = tensor<T, 2, container_type>;

        auto weight = shared_tensor(tensor_type());
        tensor_accessor::resize(*ptr, weight.accessor(), ptr->dimensions());
        weight.set_container(ptr->container_ptr());

        weight = permute_attention_heads(weight, n_heads, accelerator);
        ptr->set_container(weight.container_ptr());
    }

    template <immutable_tensor2_t<T> Input> requires std::default_initializable<Input>
    auto
    permute_attention_heads(
        const Input& input, std::size_t n_heads, hardware_accelerator& accelerator
    ) const
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

        auto output = concatenate<T>(tensors.begin(), tensors.end(), 0, accelerator);
        return output.get();
    }

    nn::llama3_options _M_options;
};


/// Layer adaptor for Llama3 model with QLoRa quantization distributed through HuggingFace repo.
///
/// The adaptor replaces linear layers with \ref quantization::lora_linear, embedding layer with
/// \ref quantization::lora_embedding and output layer with \ref quantization::linear.
///
/// These layers perform on-the-flyte dequantization, which increases compute time due to the need
/// to reconstruct original weights.
///
/// \tparam T a type of the dequantized weights.
template <typename T> struct llama3_qlora_layer_adaptor {
    llama3_qlora_layer_adaptor(nn::llama3_options options)
    : _M_options(options)
    {}

    /// Adapt the Llama3 model before loading weights. Performs in-place replacement of linear
    /// and embedding layers.
    void
    adapt_pre(nn::indirect_layer<nn::basic_layer> layer) const
    {
        auto is_basic_linear = nn::layer_common_with<nn::basic_linear<T>>();
        auto is_basic_embedding = nn::layer_common_with<nn::basic_embedding<T>>();
        auto is_output = nn::layer_all(is_basic_linear, nn::layer_name_match("output"));

        using QLinear = quantization::linear<T>;
        using QLoraEmbedding = quantization::lora_embedding<T>;
        using QLoraLinear = quantization::lora_linear<T>;

        auto& accelerator = layer.accelerator();
        nn::indirect_layer<QLinear> linear(accelerator);
        nn::indirect_layer<QLoraEmbedding> embedding(accelerator);

        nn::replace_layer(layer, is_basic_linear, [&] {
            return nn::indirect_layer<QLoraLinear>(2.0, 32, accelerator);
        });
        nn::replace_layer(layer, is_basic_embedding, embedding);
        nn::replace_layer(layer, is_output, linear);
    }

    void
    adapt_post(nn::indirect_layer<nn::basic_layer> layer) const
    {}

private:
    nn::llama3_options _M_options;
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


template <typename T, contiguous_container Container> struct llama3_traits {
    using value_type = T;
    using layer_type = nn::llama3<T, Container>;
    using layer_adaptor = llama3_layer_adaptor<T>;
    using options_type = nn::llama3_options;
    using options_serializer = llama3_options_serializer;
    using container_type = Container;
    using document_adaptor = llama3_document_adaptor;
    using tokenizer_type = text::byte_pair_encoder<text::regexp>;
    using tokenizer_loader = llama3_tokenizer_loader;

    static constexpr std::string_view tokenizer_location = "tokenizer.json";
    static constexpr std::string_view options_location = "config.json";
    static constexpr std::string_view transformer_location = "model.safetensors";
};


/// TODO: Replace llama3_options with llama3_qlora_options.
template <typename T, contiguous_container Container> struct llama3_qlora_traits {
    using value_type = T;
    using layer_type = nn::llama3<T, Container>;
    using layer_adaptor = llama3_qlora_layer_adaptor<T>;
    using options_type = nn::llama3_options;
    using options_serializer = llama3_options_serializer;
    using container_type = Container;
    using document_adaptor = noop_document_adaptor;
    using tokenizer_type = text::byte_pair_encoder<text::regexp>;
    using tokenizer_loader = reference::llama3_tokenizer_loader;
};


using llama3 = llama3_traits<bf16, hardware_memory_container<bf16>>;
using llama3_qlora = llama3_qlora_traits<bf16, hardware_memory_container<bf16>>;


} // namespace huggingface
} // namespace metalchat
