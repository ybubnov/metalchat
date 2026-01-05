// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <fstream>
#include <istream>
#include <string_view>

#include <metalchat/allocator.h>
#include <metalchat/nn.h>
#include <metalchat/safetensor.h>
#include <metalchat/tensor/concept.h>
#include <metalchat/text.h>


namespace metalchat {


/// A layer adaptor that is used by repository implementations to prepare the model for the usage.
///
/// Depending on the layer type and type of model distribution, the adaptor could be used to
/// (1) map weight names from one implementation to another, (2) rebuild a model to load
/// quantized implementation (QLoRa), etc.
template <typename Adaptor>
concept indirect_layer_adaptor = requires(std::remove_reference_t<Adaptor> const a) {
    /// An `adapt_pre` method is called, immediately after model creation and before loading
    /// weights from the file.
    { a.adapt_pre(nn::indirect_layer<nn::basic_layer>()) } -> std::same_as<void>;

    /// An `adapt_post` method is called after loading weights from file.
    { a.adapt_post(nn::indirect_layer<nn::basic_layer>()) } -> std::same_as<void>;
};


/// An implementation of \ref indirect_layer_adaptor concept that does nothing. Use it to declare
/// transformer traits (\ref transformer_traits) that do not perform any layer adaptation.
template <typename LayerOptions> struct noop_layer_adaptor {
    noop_layer_adaptor(LayerOptions) {}

    /// The implementation does not modify the specified layer.
    void
    adapt_pre(nn::indirect_layer<nn::basic_layer>) const
    {}

    /// The implementation does not modify the specified layer.
    void
    adapt_post(nn::indirect_layer<nn::basic_layer>) const
    {}
};


template <typename Loader>
concept tokenizer_loader = requires(std::remove_reference_t<Loader> const l, std::istream& is) {
    typename Loader::type;

    { l.load(is) } -> std::same_as<typename Loader::type>;
};


/// The requirements for a transformer declaration.
template <typename Traits>
concept transformer_traits = requires {
    typename Traits::layer_type;
    typename Traits::layer_adaptor;
    typename Traits::options_type;
    typename Traits::options_loader;
    typename Traits::document_adaptor;
    typename Traits::container_type;

    requires nn::layer<typename Traits::layer_type>;
    requires contiguous_container<typename Traits::container_type>;
    requires indirect_layer_adaptor<typename Traits::layer_adaptor>;
    requires safetensor_document_adaptor<typename Traits::document_adaptor>;
};


/// Requirement of the `transformer_location` constant expression presence.
///
/// This concept is used in the repository implementation enabling methods for
/// retrieval of transformer instances from a default location within a repository.
template <typename Traits>
concept has_transformer_location = requires {
    Traits::transformer_location;

    requires std::same_as<decltype(Traits::transformer_location), std::string_view const>;
};

/// Requirement of the `options_location` constant expression presence.
///
/// This concept is used in the repository implementation enabling methods for
/// retrieval of option instances from a default location within a repository.
template <typename Traits>
concept has_options_location = requires {
    Traits::options_location;

    requires std::same_as<decltype(Traits::options_location), std::string_view const>;
};


/// Requirement of the `tokenizer_location` constant expression presence.
///
/// This concept is used in the repository implementation enabling methods for
/// retrieval of tokenizer instances from a default location within a repository.
template <typename Traits>
concept has_tokenizer_location = requires {
    Traits::tokenizer_location;

    requires std::same_as<decltype(Traits::tokenizer_location), std::string_view const>;
};


class basic_transformer {
public:
    using index_type = int32_t;
    using tensor_type = future_tensor<index_type, 2>;

    virtual tensor_type
    transform(tensor_type, std::size_t start_pos) = 0;

    virtual hardware_accelerator&
    accelerator() = 0;
};


template <typename Transformer> class transformer_wrapper : public basic_transformer {
public:
    transformer_wrapper(Transformer&& transformer)
    : _M_transformer(std::move(transformer))
    {}

    transformer_wrapper(const Transformer& transformer)
    : _M_transformer(transformer)
    {}

    tensor_type
    transform(tensor_type input, std::size_t start_pos)
    {
        return _M_transformer.transform(input, start_pos);
    }

    hardware_accelerator&
    accelerator()
    {
        return _M_transformer.get_layer().accelerator();
    }

private:
    Transformer _M_transformer;
};


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


} // namespace metalchat
