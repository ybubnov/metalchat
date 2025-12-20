// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <fstream>

#include <metalchat/allocator.h>
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


template <typename T = bf16, contiguous_container Container = hardware_memory_container<T>>
struct llama3_reference_traits {
    using value_type = T;
    using layer_type = nn::llama3<T, Container>;
    using options_type = nn::llama3_options;
    using container_type = Container;

    // The original implementation of Llama 3.2 shares the weight of token embeddings
    // and the output layer, use a shared tensor in order to reduce memory footprint.
    //
    // This adaptor implement \ref safetensor_document_adaptor concept and creates an
    // alias between output and embedding layers.
    struct document_adaptor {
        safetensor_document
        adapt(const safetensor_document& document) const
        {
            auto doc = document;
            doc.insert("output.weight", "tok_embeddings.weight");
            return doc;
        }
    };
};


template <typename TransformerTraits>
concept transformer_traits = requires {
    typename TransformerTraits::layer_type;
    typename TransformerTraits::options_type;
    typename TransformerTraits::document_adaptor;
    typename TransformerTraits::container_type;

    contiguous_container<typename TransformerTraits::container_type>;
    safetensor_document_adaptor<typename TransformerTraits::document_adaptor>;
};


template <transformer_traits TransformerTraits> struct autoloader {
    using layer_type = TransformerTraits::layer_type;
    using options_type = TransformerTraits::options_type;
    using container_type = TransformerTraits::container_type;
    using document_adaptor_type = TransformerTraits::document_adaptor;

    autoloader(
        const std::filesystem::path& repo_path,
        const std::string& safetensor_filename,
        hardware_accelerator accelerator
    )
    : _M_local_path(repo_path / safetensor_filename),
      _M_accelerator(accelerator)
    {}

    autoloader(const std::filesystem::path& repo_path, hardware_accelerator accelerator)
    : autoloader(repo_path, "model.safetensors", accelerator)
    {}

    autoloader(const std::filesystem::path& repo_path)
    : autoloader(repo_path, hardware_accelerator())
    {}

    transformer<layer_type>
    load(const options_type& options)
    {
        nn::indirect_layer<layer_type> layer(options, _M_accelerator);

        auto document_adaptor = document_adaptor_type();
        auto document = safetensor_document::open(_M_local_path, _M_accelerator);

        document = document_adaptor.adapt(document);
        document.load(layer);

        return transformer<layer_type>(layer);
    }

    template <allocator_t<void> Allocator>
    transformer<layer_type>
    load(const options_type& options, Allocator alloc = Allocator())
    {
        nn::indirect_layer<layer_type> layer(options, _M_accelerator);

        auto document_stream = std::ifstream(_M_local_path, std::ios::binary);
        auto document_adaptor = document_adaptor_type();
        auto document = safetensor_document::open(document_stream, alloc);

        document = document_adaptor.adapt(document);
        document.load(layer);

        return transformer<layer_type>(layer);
    }

private:
    std::filesystem::path _M_local_path;
    hardware_accelerator _M_accelerator;
};


using reference_autoloader = autoloader<llama3_reference_traits<bf16>>;


} // namespace metalchat
