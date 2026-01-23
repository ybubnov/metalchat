// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

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

    /// Update manifest of an existing model.
    void
    update(const model_info&);

    void
    insert(const manifest&);

private:
    std::filesystem::path
    resolve_path(const std::string& id) const;

    std::filesystem::path _M_path;
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
