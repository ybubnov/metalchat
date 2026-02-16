// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/nn.h>
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


/// A nucleus sampler adapter for the \ref nn::nucleus_sampler implementation that parses
/// configuration options from the key-value mapping (extracted from the manifest file).
template <typename T> struct nucleus_sampler_traits {
    using value_type = nn::basic_sampler<T>;
    using pointer = std::shared_ptr<value_type>;

    static constexpr float default_temperature = 0.6f;
    static constexpr float default_probability = 0.9f;
    static constexpr float default_k = 40;

    /// Extract a value of the specified option key from the options section, throws
    /// on a type mismatch.
    template <typename U>
    static std::optional<U>
    get(const options_section& options, const std::string& key)
    {
        auto option_it = options.find(key);
        if (option_it == options.end()) {
            return std::nullopt;
        }
        auto& option_value = option_it->second;
        if (!std::holds_alternative<U>(option_value)) {
            throw std::invalid_argument(std::format("error: sampling option type '{}'", key));
        }

        return std::get<U>(option_value);
    }

    static pointer
    construct(const options_section& options)
    {
        auto temp = get<float>(options, "temperature").value_or(default_temperature);
        auto prob = get<float>(options, "probability").value_or(default_probability);
        auto k = get<int>(options, "k").value_or(default_k);

        using sampler_type = nn::sequential_sampler<T>;
        sampler_type sampler(
            {std::make_shared<nn::topk_sampler<T>>(k),
             std::make_shared<nn::nucleus_sampler<T>>(temp, prob),
             std::make_shared<nn::multinomial_sampler<T>>(/*sample_size=*/1)}
        );

        return std::make_shared<sampler_type>(std::move(sampler));
    }
};


/// A container of the supported sampling strategies available through manifest file.
template <typename T> class sampler_container {
public:
    using value_type = nn::basic_sampler<T>;
    using pointer = std::shared_ptr<value_type>;
    using constructor_type = std::function<pointer(options_section)>;

    sampler_container()
    : _M_values({{"nucleus", nucleus_sampler_traits<T>::construct}})
    {}

    pointer
    find(const options_section& options) const
    {
        auto type = options.find("type");
        if (type == options.end()) {
            throw std::runtime_error("error: sampler section is missing a key 'type'");
        }
        if (!std::holds_alternative<std::string>(type->second)) {
            throw std::runtime_error("error: sampler type must by a string");
        }

        auto sampler_type = std::get<std::string>(type->second);
        auto sampler = _M_values.find(sampler_type);
        if (sampler == _M_values.end()) {
            throw std::runtime_error(std::format("sampler: type '{}' not registered", sampler_type)
            );
        }

        auto& constructor = sampler->second;
        return constructor(options);
    }

private:
    std::unordered_map<std::string, constructor_type> _M_values;
};


template <language_transformer Transformer> class scoped_repository_adapter {
public:
    using container_type = Transformer::container_type;
    using layer_type = Transformer::layer_type;
    using layer_serializer = Transformer::layer_serializer;
    using options_type = Transformer::options_type;
    using options_serializer = Transformer::options_serializer;
    using tokenizer_type = Transformer::tokenizer_type;
    using tokenizer_loader = Transformer::tokenizer_loader;

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

        auto inference = _M_manifest.inference.value_or(inference_section{});
        if (inference.max_sequence_length) {
            options = options.max_seq_len(inference.max_sequence_length.value());
        }

        return options;
    }

    tokenizer_type
    retrieve_tokenizer() const
    {
        return _M_repo.retrieve_tokenizer();
    }

    transformer_type
    retrieve_transformer()
    {
        auto transformer = _M_repo.retrieve_transformer(retrieve_options());
        auto inference = _M_manifest.inference.value_or(inference_section{});

        if (inference.sampling) {
            using value_type = transformer_type::value_type;
            sampler_container<value_type> samplers;

            auto sampling_options = inference.sampling.value();
            auto sampler = samplers.find(sampling_options);
            transformer.set_sampler(sampler);
        }

        return transformer;
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
