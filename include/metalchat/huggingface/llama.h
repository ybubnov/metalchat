// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/autoloader.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/nn.h>
#include <metalchat/safetensor.h>
#include <metalchat/tensor/basic.h>
#include <metalchat/tensor/shared.h>


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

    /// Performs permutation of the attention heads within Wq and Wk layers so that the order
    /// of elements as in the Meta's reference implementation.
    ///
    /// \param layer a layer to adapt to the reference implementation.
    void
    adapt(nn::indirect_layer<nn::basic_layer> layer) const
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


template <typename T = bf16, contiguous_container Container = hardware_memory_container<T>>
struct llama3_traits {
    using value_type = T;
    using layer_type = nn::llama3<T, Container>;
    using layer_adaptor = llama3_layer_adaptor<T>;
    using options_type = nn::llama3_options;
    using container_type = Container;

    using document_adaptor = llama3_document_adaptor;
};


using llama3_autoloader = autoloader<llama3_traits<bf16>>;


} // namespace huggingface
} // namespace metalchat
