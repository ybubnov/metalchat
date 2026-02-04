// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/repository.h>

#include "command.h"
#include "manifest.h"


namespace metalchat {
namespace runtime {


struct model_info {
    manifest manifest;
    std::filesystem::path path;
};


class model_provider {
public:
    /// The default location of model data within a root path.
    static constexpr std::string_view default_path = "models";

    using ManifestFile = tomlfile<manifest>;

    model_provider(const std::filesystem::path& p);

    bool
    exists(const std::string& id) const;

    /// Find a model in a repository and return it's manifest. When the model
    /// does not exist in a repository, method throws an exception.
    model_info
    find(const std::string& id) const;

    template <typename UnaryPred>
    std::optional<model_info>
    find_if(UnaryPred p) const
    {
        for (auto const& entry : std::filesystem::directory_iterator(_M_path)) {
            if (!std::filesystem::is_directory(entry)) {
                continue;
            }

            auto model = find(entry.path().filename().string());
            if (p(model)) {
                return model;
            }
        }
        return std::nullopt;
    }

    /// Remove model from the repository by the given identifier. When the model
    /// does not exist in a repository, method throws an exception.
    void
    remove(const std::string& id);

    void
    insert(const manifest&);

private:
    std::filesystem::path
    resolve_path(const std::string& id) const;

    std::filesystem::path _M_path;
};


template <language_transformer Transformer> class scoped_repository_adapter {
public:
    using layer_type = Transformer::layer_type;
    using layer_adaptor_type = Transformer::layer_adaptor;
    using options_type = Transformer::options_type;
    using options_serializer = Transformer::options_serializer;
    using tokenizer_type = Transformer::tokenizer_type;
    using tokenizer_loader = Transformer::tokenizer_loader;
    using container_type = Transformer::container_type;
    using document_adaptor_type = Transformer::document_adaptor;

    using transformer_type = transformer<layer_type>;

    scoped_repository_adapter(const std::filesystem::path& root_path, const manifest& m)
    : _M_repo(root_path),
      _M_manifest(m)
    {}

    options_type
    retrieve_options() const
    {
        using TransformerTraits = transformer_traits<Transformer>;

        auto options = _M_repo.retrieve_options();
        if (_M_manifest.options) {
            auto manifest_options = _M_manifest.options.value();
            auto first = manifest_options.begin();
            auto last = manifest_options.end();
            options = TransformerTraits::merge_options(first, last, options);
        }

        return options;
    }

    tokenizer_type
    retrieve_tokenizer() const
    {
        return _M_repo.retrieve_tokenizer();
    }

    transformer_type
    retrieve_transformer() const
    {
        return _M_repo.retrieve_transformer(retrieve_options());
    }

private:
    filesystem_repository<Transformer> _M_repo;
    manifest _M_manifest;
};


class model_command : public basic_command {
public:
    model_command(basic_command& parent);

    void
    pull(const command_context&);

    void
    list(const command_context&);

    void
    remove(const command_context&);

private:
    parser_type _M_pull;
    parser_type _M_list;
    parser_type _M_remove;

    std::string _M_repository;
    std::string _M_partitioning;
    std::string _M_arch;
    std::string _M_variant;
    std::string _M_id;
};


} // namespace runtime
} // namespace metalchat
