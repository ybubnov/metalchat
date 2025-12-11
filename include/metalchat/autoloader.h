// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/nn.h>
#include <metalchat/safetensor.h>
#include <metalchat/tensor/concept.h>


namespace metalchat {


template <typename Layer> class transformer {
public:
    using index_type = std::int32_t;
    using value_type = Layer::value_type;

    using layer_type = Layer;
    using layer_pointer = nn::indirect_layer<layer_type>;

    using sampler_type = nn::basic_sampler<value_type>;
    using sampler_pointer = std::shared_ptr<sampler_type>;

    transformer(const layer_pointer& layer, const sampler_pointer& sampler)
    : _M_layer(layer),
      _M_sampler(sampler)
    {}

    transformer(const layer_pointer& layer)
    : _M_layer(layer),
      _M_sampler(std::make_shared<nn::nucleus_sampler<value_type>>())
    {}

    void
    set_sampler(const sampler_pointer& sampler)
    {
        _M_sampler = sampler;
    }

    const sampler_pointer&
    get_sampler() const
    {
        return _M_sampler;
    }

    const layer_pointer&
    get_layer() const
    {
        return _M_layer;
    }

    layer_pointer&
    get_layer()
    {
        return _M_layer;
    }

    template <immutable_tensor2_t<index_type> Input>
    auto
    transform(Input input, std::size_t start_pos = 0)
    {
        auto& accelerator = _M_layer.accelerator();
        auto logits = _M_layer(input, start_pos);
        return _M_sampler->sample(logits.template flatten<2>(), accelerator);
    }

private:
    nn::indirect_layer<layer_type> _M_layer;
    sampler_pointer _M_sampler;
};


struct llama3_reference_traits {
    using layer_type = nn::llama3<bf16>;
    using options_type = nn::llama3_options;

    struct document_resolver {
        std::filesystem::path
        resolve(const std::filesystem::path& p) const;
    };

    // The original implementation of Llama 3.2 shares the weight of token embeddings
    // and the output layer, use a shared tensor in order to reduce memory footprint.
    //
    // This adaptor implement \ref safetensor_document_adaptor concept and creates an
    // alias between output and embedding layers.
    struct document_adaptor {
        void
        adapt(safetensor_document& document) const;
    };
};


struct llama3_huggingface_traits {
    using layer_type = nn::llama3<bf16>;
    using options_type = nn::llama3_options;

    struct document_adaptor {
        void
        adapt(safetensor_document& document) const;
    };
};


template <typename TransformerTraits>
concept transformer_traits = requires {
    typename TransformerTraits::layer_type;
    typename TransformerTraits::options_type;
    typename TransformerTraits::document_adaptor;

    safetensor_document_adaptor<typename TransformerTraits::document_adaptor>;
};


/// ```c++
/// using Transformer = metalchat::llama3_huggingface_traits;
/// using Autoloader = metalchat::autoloader<Transformer>;
///
/// hardware_accelerator accelerator;
/// Autoloader autoloader("Llama-3.1-1B-Instruct");
/// auto layer = autoloader.load(accelerator);
/// ```
template <transformer_traits TransformerTraits = llama3_reference_traits> struct autoloader {
    using layer_type = TransformerTraits::layer_type;
    using options_type = TransformerTraits::options_type;
    using document_adaptor_type = TransformerTraits::document_adaptor;
    using document_resolver_type = TransformerTraits::document_resolver;

    autoloader(const std::filesystem::path& local_path)
    : _M_local_path(local_path)
    {}

    transformer<layer_type>
    load(options_type options, hardware_accelerator& accelerator) const
    {
        nn::indirect_layer<layer_type> layer(options, accelerator);

        auto document_adaptor = document_adaptor_type();
        auto document_resolver = document_resolver_type();
        auto document_path = document_resolver.resolve(_M_local_path);
        auto document = safetensor_document::open(document_path, accelerator);

        document_adaptor.adapt(document);
        document.load(layer);

        return transformer<layer_type>(layer);
    }

    transformer<layer_type>
    load(hardware_accelerator& accelerator) const;

private:
    std::filesystem::path _M_local_path;
};


using reference_autoloader = autoloader<llama3_reference_traits>;
using huggingface_autoloader = autoloader<llama3_huggingface_traits>;


} // namespace metalchat
