// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/text.h>


namespace metalchat {
namespace huggingface {


struct llama3_tokenizer_loader {
    using tokenizer_type = text::byte_pair_encoder<text::regexp>;

    tokenizer_type
    load(std::istream& is) const;

    tokenizer_type
    load(const std::filesystem::path& p) const;
};


} // namespace huggingface
} // namespace metalchat
