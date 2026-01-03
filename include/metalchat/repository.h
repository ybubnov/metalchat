// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>
#include <string>

#include <metalchat/accelerator.h>
#include <metalchat/transformer.h>


namespace metalchat {


/// A filesystem-based read-only repository used to retrieve language transformer building blocks
/// (layer options, layer, and string tokenizer).
template <transformer_traits TransformerTraits> struct filesystem_repository {
    using layer_type = TransformerTraits::layer_type;
    using layer_adaptor_type = TransformerTraits::layer_adaptor;
    using options_type = TransformerTraits::options_type;
    using options_loader = TransformerTraits::options_loader;
    using tokenizer_type = TransformerTraits::tokenizer_type;
    using tokenizer_loader = TransformerTraits::tokenizer_loader;
    using container_type = TransformerTraits::container_type;
    using document_adaptor_type = TransformerTraits::document_adaptor;

    using transformer_type = transformer<layer_type>;

    filesystem_repository(const std::filesystem::path& repo_path, hardware_accelerator accelerator)
    : _M_repo_path(repo_path),
      _M_accelerator(accelerator)
    {}

    filesystem_repository(const std::filesystem::path& repo_path)
    : filesystem_repository(repo_path, hardware_accelerator())
    {}

    options_type
    retrieve_options(const std::filesystem::path& p) const
    {
        auto options_path = _M_repo_path / p;
        auto options_stream = std::ifstream(options_path);
        if (!options_stream.is_open()) {
            throw std::invalid_argument(std::format(
                "filesystem_repository: failed opening file '{}'", options_path.string()
            ));
        }

        options_loader loader;
        return loader.load(options_stream);
    }

    options_type
    retrieve_options() const requires has_options_location<TransformerTraits>
    {
        const std::filesystem::path p(TransformerTraits::options_location);
        return retrieve_options(p);
    }

    tokenizer_type
    retrieve_tokenizer(const std::filesystem::path& p) const
    {
        auto tokenizer_path = _M_repo_path / p;
        auto tokenizer_stream = std::ifstream(tokenizer_path);
        if (!tokenizer_stream.is_open()) {
            throw std::invalid_argument(std::format(
                "filesystem_repository: failed opening file '{}'", tokenizer_path.string()
            ));
        }

        tokenizer_loader loader;
        return loader.load(tokenizer_stream);
    }

    tokenizer_type
    retrieve_tokenizer() const requires has_tokenizer_location<TransformerTraits>
    {
        const std::filesystem::path p(typename TransformerTraits::tokenizer_location);
        return retrieve_tokenizer(p);
    }

    transformer_type
    retrieve_transformer(const std::filesystem::path& p, const options_type& options)
    {
        nn::indirect_layer<layer_type> layer(options, _M_accelerator);
        nn::indirect_layer<nn::basic_layer> layer_base(layer.get());

        layer_adaptor_type layer_adaptor(options);
        layer_adaptor.adapt_pre(layer_base);

        auto document_adaptor = document_adaptor_type();
        auto document = safetensor_document::open(_M_repo_path / p, _M_accelerator);

        document = document_adaptor.adapt(document);
        document.load(layer);

        layer_adaptor.adapt_post(layer_base);
        return transformer_type(layer);
    }

    transformer_type
    retrieve_transformer(const options_type& options)
        requires has_transformer_location<TransformerTraits>
    {
        const std::filesystem::path p(typename TransformerTraits::transformer_location);
        return retrieve_transformer(p);
    }

    transformer_type
    retrieve_transformer()
    {
        return retrieve_transformer(retrieve_options());
    }

    template <allocator_t<void> Allocator>
    transformer_type
    retrieve_transformer(
        const std::filesystem::path& p, const options_type& options, Allocator alloc = Allocator()
    )
    {
        nn::indirect_layer<layer_type> layer(options, _M_accelerator);
        nn::indirect_layer<nn::basic_layer> layer_base(layer.get());

        layer_adaptor_type layer_adaptor(options);
        layer_adaptor.adapt_pre(layer_base);

        auto document_path = _M_repo_path / p;
        auto document_stream = std::ifstream(document_path, std::ios::binary);
        if (!document_stream.is_open()) {
            throw std::invalid_argument(std::format(
                "filesystem_repository: failed opening file '{}'", document_path.string()
            ));
        }
        auto document_adaptor = document_adaptor_type();
        auto document = safetensor_document::open(document_stream, alloc);

        document = document_adaptor.adapt(document);
        document.load(layer);

        layer_adaptor.adapt_post(layer_base);
        return transformer<layer_type>(layer);
    }

    template <allocator_t<void> Allocator>
    transformer_type
    retrieve_transformer(const options_type& options, Allocator alloc = Allocator())
        requires has_transformer_location<TransformerTraits>
    {
        const std::filesystem::path p(typename TransformerTraits::transformer_location);
        return retrieve_transformer(p, options, alloc);
    }

    template <allocator_t<void> Allocator>
    transformer_type
    retrieve_transformer(Allocator alloc = Allocator())
    {
        return retrieve_transformer(retrieve_options(), alloc);
    }

private:
    std::filesystem::path _M_repo_path;
    hardware_accelerator _M_accelerator;
};


} // namespace metalchat
