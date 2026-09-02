// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <memory>
#include <string_view>

#include <metalchat/format.h>
#include <metalchat/nn/lfm.h>
#include <metalchat/safetensor.h>
#include <metalchat/text.h>


namespace metalchat {
namespace huggingface {


using namespace std::literals;


/// LFM2 options serializer for configurations distributed through HuggingFace repository.
struct lfm2_options_serializer {
    using value_type = nn::lfm2_options;

    nn::lfm2_options
    load(std::istream& is) const;

    void
    save(std::ostream& os, const nn::lf2_options& options) const;
};


struct lfm2_tokenizer_loader {
    using type = text::byte_pair_encoder<char>;

    type
    load(std::istream& is) const;

    type
    load(const std::filesystem::path& p) const;
};


} // namespace huggingface
} // namespace metalchat
