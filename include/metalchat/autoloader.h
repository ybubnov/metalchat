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


/// A layer adaptor that is used by autoloader to prepare the model for the usage.
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


/// An implementation of \ref indirect_layer_adaptor concept that does nothing.
template <typename LayerOptions> struct noop_layer_adaptor {
    noop_layer_adaptor(LayerOptions) {}

    void
    adapt_pre(nn::indirect_layer<nn::basic_layer>) const
    {}

    void
    adapt_post(nn::indirect_layer<nn::basic_layer>) const
    {}
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


template <typename Traits>
concept transformer_traits = requires {
    typename Traits::layer_type;
    typename Traits::layer_adaptor;
    typename Traits::options_type;
    typename Traits::document_adaptor;
    typename Traits::container_type;

    requires contiguous_container<typename Traits::container_type>;
    requires indirect_layer_adaptor<typename Traits::layer_adaptor>;
    requires safetensor_document_adaptor<typename Traits::document_adaptor>;
};


/// A type that supports creating transformers from the traits.
template <transformer_traits TransformerTraits> struct autoloader {
    using layer_type = TransformerTraits::layer_type;
    using layer_adaptor_type = TransformerTraits::layer_adaptor;
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
        nn::indirect_layer<nn::basic_layer> layer_base(layer.get());

        layer_adaptor_type layer_adaptor(options);
        layer_adaptor.adapt_pre(layer_base);

        auto document_adaptor = document_adaptor_type();
        auto document = safetensor_document::open(_M_local_path, _M_accelerator);

        document = document_adaptor.adapt(document);
        document.load(layer);

        layer_adaptor.adapt_post(layer_base);
        return transformer<layer_type>(layer);
    }

    template <allocator_t<void> Allocator>
    transformer<layer_type>
    load(const options_type& options, Allocator alloc = Allocator())
    {
        nn::indirect_layer<layer_type> layer(options, _M_accelerator);
        nn::indirect_layer<nn::basic_layer> layer_base(layer.get());

        layer_adaptor_type layer_adaptor(options);
        layer_adaptor.adapt_pre(layer_base);

        auto document_stream = std::ifstream(_M_local_path, std::ios::binary);
        auto document_adaptor = document_adaptor_type();
        auto document = safetensor_document::open(document_stream, alloc);

        document = document_adaptor.adapt(document);
        document.load(layer);

        layer_adaptor.adapt_post(layer_base);
        return transformer<layer_type>(layer);
    }

private:
    std::filesystem::path _M_local_path;
    hardware_accelerator _M_accelerator;
};


namespace reference {


struct llama3_document_adaptor {
    safetensor_document
    adapt(const safetensor_document& document) const;
};


struct llama3_options_loader {
    nn::llama3_options
    load(std::istream&) const;
};


/// Reference implementation of the Llama3 tokenizer.
///
/// This loader implements loading of a tokenizer model in a reference (tiktoken) format. It
/// expects that `load` methods receives a file in a tiktoken format.
struct llama3_tokenizer_loader {
    using tokenizer_type = text::byte_pair_encoder<text::regexp>;

    /// A regular expression string that is used to split the input text into tokens.
    // clang-format off
    static constexpr std::string_view default_regex =
        (R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|)"
         R"([^\r\n\p{L}\p{N}]?\p{L}+|)"
         R"(\p{N}{1,3}|)"
         R"( ?[^\s\p{L}\p{N}]+[\r\n]*|)"
         R"(\s*[\r\n]+|)"
         R"(\s+(?!\S)|)"
         R"(\s+)");
    // clang-format on

    /// Load a tokenizer from the input stream.
    ///
    /// \param is An input stream containing tokenizer model (tiktoken format).
    /// \param token_regex A regular expression used to split a string into tokens.
    tokenizer_type
    load(std::istream& is, const std::string& token_regex) const;

    /// Load a tokenizer from the local file.
    ///
    /// \param p A path to the file containing tokenizer model (tiktoken format).
    /// \param token_regex A regular expression used to split a string into tokens.
    tokenizer_type
    load(const std::filesystem::path& p, const std::string& token_regex) const;

    /// Load a tokenizer from the input stream.
    ///
    /// The implementation uses a \ref default_regex to split sentence into tokens.
    ///
    /// See also \ref load(std::istream&, const std::string&) const.
    tokenizer_type
    load(std::istream& is) const;

    /// Load a tokenizer from the local file.
    ///
    /// The implementation uses a \ref default_regex to split sentence into tokens.
    ///
    /// See also \ref load(const std::filesystem::path&, const std::string&) const.
    tokenizer_type
    load(const std::filesystem::path& p) const;

    static void
    insert_control_tokens(tokenizer_type& bpe);
};


text::byte_pair_encoder<text::regexp>
make_tokenizer(const std::filesystem::path& local_path);


template <typename T = bf16, contiguous_container Container = hardware_memory_container<T>>
struct llama3_traits {
    using value_type = T;
    using layer_type = nn::llama3<T, Container>;
    using options_type = nn::llama3_options;
    using container_type = Container;

    /// The original implementation of Llama 3.2 shares the weight of token embeddings
    /// and the output layer, use a shared tensor in order to reduce memory footprint.
    ///
    /// This adaptor implement \ref safetensor_document_adaptor concept and creates an
    /// alias between output and embedding layers.
    using document_adaptor = llama3_document_adaptor;

    /// The original implementation does not require adaptation of the layer, so this
    /// layer adaptor does nothing.
    using layer_adaptor = noop_layer_adaptor<options_type>;
};


using llama3_autoloader = autoloader<llama3_traits<bf16>>;


} // namespace reference


} // namespace metalchat
