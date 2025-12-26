// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_map>

#include <CLI/CLI.hpp>

#include "hash.h"


namespace metalchat {
namespace program {


struct architecture {
    static constexpr std::string_view llama3x2_1b = "llama3.2:1b";
    static constexpr std::string_view llama3x2_3b = "llama3.2:3b";
};


/// Implementation differs between various sources of models distributions. Specifically
/// Llama3.2 has different weight names in original implementation and in huggingface.
///
/// There is no intention supporting all implementations of the model implementation, it's
/// only serves a purpose to support some of the most popular ones.
struct implementation {
    static constexpr std::string_view reference = "reference";
    static constexpr std::string_view huggingface = "huggingface";
};


struct chat_create_options {
    std::optional<std::string> name = std::nullopt;
    std::optional<std::string> system_prompt = std::nullopt;
    std::string model = "";
    std::string arch = "";
    std::string impl = "";
};


class chat {
public:
    using model_version = std::tuple<std::string, std::string>;

    static constexpr std::string_view default_model =
        "huggingface.co/meta-llama/Llama-3.2-1B-Instruct";

    /// This mapping holds the all supported combinations of the models. If model is not
    /// within this list, then attempt to create such a model fails.
    static const std::unordered_map<model_version, int> models;

    chat(const chat_create_options&);
};


class chat_command {
public:
    chat_command(CLI::App& app);

    void
    create();

private:
    chat_create_options _M_create_options;
};


} // namespace program
} // namespace metalchat
