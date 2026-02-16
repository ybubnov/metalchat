// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>

#include <metalchat/accelerator.h>
#include <metalchat/safetensor.h>
#include <metalchat/transformer.h>


namespace metalchat {


/// A filesystem-based read-only repository used to retrieve language transformer building blocks
/// (layer options, layer, and string tokenizer).
///
/// \tparam Transformer a transformer specification.
/// \tparam Document a document format type.
template <language_transformer Transformer, typename Document = safetensor_document>
struct filesystem_repository {
    using layer_type = Transformer::layer_type;
    using layer_serializer = Transformer::layer_serializer;
    using options_type = Transformer::options_type;
    using options_serializer = Transformer::options_serializer;
    using tokenizer_type = Transformer::tokenizer_type;
    using tokenizer_loader = Transformer::tokenizer_loader;
    using container_type = Transformer::container_type;

    using document_type = Document;
    using transformer_type = transformer<layer_type>;

    filesystem_repository(const std::filesystem::path& repo_path, hardware_accelerator accelerator)
    : _M_repo_path(repo_path),
      _M_accelerator(accelerator)
    {}

    filesystem_repository(const std::filesystem::path& repo_path)
    : filesystem_repository(repo_path, hardware_accelerator())
    {}

    const std::filesystem::path&
    path() const
    {
        return _M_repo_path;
    }

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

        options_serializer serializer;
        return serializer.load(options_stream);
    }

    options_type
    retrieve_options() const requires has_options_location<Transformer>
    {
        const std::filesystem::path p(Transformer::options_location);
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
    retrieve_tokenizer() const requires has_tokenizer_location<Transformer>
    {
        const std::filesystem::path p(Transformer::tokenizer_location);
        return retrieve_tokenizer(p);
    }

    transformer_type
    retrieve_transformer(const std::filesystem::path& p, const options_type& options)
    {
        layer_serializer serializer(options, _M_accelerator);

        auto document_path = _M_repo_path / p;
        auto document = document_type::open(document_path, _M_accelerator);
        auto layer = serializer.load(document);

        return transformer_type(layer);
    }

    transformer_type
    retrieve_transformer(const options_type& options) requires has_transformer_location<Transformer>
    {
        const std::filesystem::path p(Transformer::transformer_location);
        return retrieve_transformer(p, options);
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
        layer_serializer serializer(options, _M_accelerator);

        auto document_path = _M_repo_path / p;
        auto document_stream = std::ifstream(document_path, std::ios::binary);
        if (!document_stream.is_open()) {
            throw std::invalid_argument(std::format(
                "filesystem_repository: failed opening file '{}'", document_path.string()
            ));
        }

        auto document = document_type::open(document_stream, alloc);
        auto layer = serializer.load(document);

        return transformer<layer_type>(layer);
    }

    template <allocator_t<void> Allocator>
    transformer_type
    retrieve_transformer(const options_type& options, Allocator alloc = Allocator())
        requires has_transformer_location<Transformer>
    {
        const std::filesystem::path p(Transformer::transformer_location);
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


template <typename FileSystem>
concept readonly_filesystem =
    requires(FileSystem const fs, const std::string& filename, std::ostream& output) {
        { fs.read(filename, output) } -> std::same_as<void>;
        { fs.exists(filename) } -> std::same_as<bool>;
    };


/// A repository that dynamically retrieves transformers from HuggingFace repository.
///
/// The implementation does not assume transport used to access HuggingFace repository,
/// therefore users must provide a necessary implementation and authentication of requests.
///
/// \tparam Transformer transformer specification.
/// \tparam FileSystem a read-only file access system used to download the transformer.
template <language_transformer Transformer, readonly_filesystem FileSystem>
struct huggingface_repository {
    using layer_type = Transformer::layer_type;
    using layer_serializer = Transformer::layer_serializer;
    using options_type = Transformer::options_type;
    using options_serializer = Transformer::options_serializer;
    using tokenizer_type = Transformer::tokenizer_type;
    using tokenizer_loader = Transformer::tokenizer_loader;
    using container_type = Transformer::container_type;

    using transformer_type = transformer<layer_type>;

    huggingface_repository(
        const std::string& id,
        const std::string& revision,
        const std::filesystem::path& p,
        FileSystem fs = FileSystem()
    )
    : _M_id(id),
      _M_revision(revision),
      _M_fs(fs),
      _M_repo(p)
    {}

    huggingface_repository(
        const std::string& id, const std::filesystem::path& p, FileSystem fs = FileSystem()
    )
    : huggingface_repository(id, resolve_revision(id, fs), p, fs)
    {}

    void
    clone() const
    {
        clone_file("config.json");
        clone_file("tokenizer.json");

        const std::string index_filename("model.safetensors.index.json");
        const std::filesystem::path index_filepath(_M_repo.path() / index_filename);

        if (!exists(index_filename)) {
            clone_file("model.safetensors");
            return;
        }

        clone_file(index_filename);

        std::ifstream index_file(index_filepath, std::ios::binary);
        std::unordered_set<std::string> filenames;

        auto index = safetensor_index::open(index_file);
        for (const auto& [_, filename] : index.weight_map) {
            if (!filenames.contains(filename)) {
                clone_file(filename);
            }
            filenames.insert(filename);
        }
    }

    tokenizer_type
    retrieve_tokenizer() const
    {
        return _M_repo.retrieve_tokenizer("tokenizer.json");
    }

    options_type
    retrieve_options() const
    {
        return _M_repo.retrieve_options("config.json");
    }

    transformer_type
    retrieve_transformer()
    {
        return retrieve_transformer(retrieve_options());
    }

    transformer_type
    retrieve_transformer(const options_type& options)
    {
        return _M_repo.retrieve_transformer("model.safetensors", options);
    }

private:
    void
    clone_file(const std::string& filename) const
    {
        std::filesystem::create_directories(_M_repo.path());
        const auto filepath = _M_repo.path() / filename;

        std::ofstream filestream(filepath, std::ios::trunc | std::ios::binary);
        _M_fs.read(link_to(filename), filestream);
    }

    bool
    exists(const std::string& filename) const
    {
        return _M_fs.exists(link_to(filename));
    }

    std::string
    link_to(const std::string& resource) const
    {
        return std::format("resolve/{}/{}", _M_revision, resource);
    }

    static std::string
    resolve_revision(const std::string& id, const FileSystem& fs)
    {
        // TODO: Resolve the model revision using /api/models/:model_id.
        return "main";
    }

    std::string _M_id;
    std::string _M_revision;
    FileSystem _M_fs;

    // Tensors in the public HuggingFace repositories are stored in multiple
    // formats, but one of the most common and supported by HuggingFace infrastructure
    // is safetensors format.
    filesystem_repository<Transformer, safetensor_document> _M_repo;
};


} // namespace metalchat
