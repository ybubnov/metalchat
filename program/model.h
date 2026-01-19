// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <toml.hpp>

#include "command.h"
#include "digest.h"
#include "http.h"


namespace metalchat {
namespace runtime {


struct architecture {
    static std::string llama3;
};


struct partitioning {
    static std::string consolidated;
    static std::string sharded;
};


struct variant {
    static std::string huggingface;
};


struct model {
    std::string repository;
    std::string variant;
    std::string architecture;
    std::string partitioning;

    std::string
    id() const
    {
        url u(repository);
        u.push_query("variant", variant);
        u.push_query("architecture", architecture);
        u.push_query("partitioning", partitioning);

        return sha1(u);
    }
};


struct manifest {
    model model;
};


} // namespace runtime
} // namespace metalchat


TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(
    metalchat::runtime::model, variant, repository, architecture, partitioning
);
TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(metalchat::runtime::manifest, model);


namespace metalchat {
namespace runtime {


class model_command : public basic_command {
public:
    static constexpr std::string_view default_path = "models";
    static constexpr std::string_view manifest_name = "manifest.toml";

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
