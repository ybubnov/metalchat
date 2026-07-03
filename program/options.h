// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/huggingface/gemma.h>
#include <metalchat/huggingface/llama.h>

#include "command.h"
#include "model.h"


namespace metalchat {
namespace runtime {


struct optionkind {
    static std::string integer;
    static std::string boolean;
    static std::string floating;
    static std::string string;
};


class options_command : public basic_command {
public:
    options_command(basic_command& parent);

    void
    get(const command_context&) const;

    void
    set(const command_context&) const;

    void
    unset(const command_context&) const;

    void
    list(const command_context&) const;

private:
    struct option {
        std::string scope;
        std::string name;
        std::string value;
    };

    template <typename Transformer>
    static std::optional<std::string>
    get(const model_info& model, const std::string& option_name)
    {
        using TransformerTraits = transformer_traits<Transformer>;

        std::optional<std::string> option_value;
        auto option_iterator = function_output_iterator([&](auto option) {
            if (option.first == option_name) {
                option_value = option.second;
            }
        });

        scoped_repository_adapter<Transformer> repo(model.path, model.manifest);
        TransformerTraits::iter_options(repo.retrieve_options(), option_iterator);

        return option_value;
    }

    template <typename Transformer>
    static void
    list(const model_info& model, const command_scope& scope, std::vector<option>& options)
    {
        // Insert the options into the runtime options container, so that it is possible
        // to sort the values in a container and print options sorted by scope.
        auto back_inserter = function_output_iterator([&](auto option) {
            auto option_scope_name = context_scope::string(scope);
            if (!model.manifest.get_option(option.first)) {
                option_scope_name = context_scope::string(context_scope::model);
            }
            options.emplace_back(option_scope_name, option.first, option.second);
        });

        using TransformerTraits = transformer_traits<Transformer>;

        scoped_repository_adapter<Transformer> repo(model.path, model.manifest);
        TransformerTraits::iter_options(repo.retrieve_options(), back_inserter);
    }

    template <std::size_t... Indices>
    void
    register_supported_architechtures(std::index_sequence<Indices...>)
    {
        (register_supported_architechture<Indices>(), ...);
    }

    template <std::size_t Index>
    void
    register_supported_architechture()
    {
        using Arch = architecture_typeinfo<Index>;
        using Transformer = Arch::transformer_type;

        _M_option_getters.insert_or_assign(std::string(Arch::name), &get<Transformer>);
        _M_option_listers.insert_or_assign(std::string(Arch::name), &list<Transformer>);
    }

    parser_type _M_get;
    parser_type _M_set;
    parser_type _M_unset;
    parser_type _M_list;

    std::string _M_name;
    std::string _M_value;
    std::string _M_type;

    using OptionGetter = std::function<decltype(get<huggingface::llama3>)>;
    using OptionLister = std::function<decltype(list<huggingface::llama3>)>;

    std::unordered_map<std::string, OptionGetter> _M_option_getters;
    std::unordered_map<std::string, OptionLister> _M_option_listers;
};


} // namespace runtime
} // namespace metalchat
