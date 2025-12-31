// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/text.h>


namespace metalchat {
namespace huggingface {


/// Llama3 tokenizer loader for a model distributed through HuggingFace repository.
///
/// The Meta's reference implementation distributes the tokenizer model in a tiktoken format,
/// while HuggingFace maintain it's own JSON-based tokenizer format. This loader performs
/// adaptation of HuggingFace JSON format into the MetalChat implementation of the tokenizer.
///
/// Note, it does not implement all features available in HuggingFace's tokenizer format, rather
/// queries necessary tokens of data from the `tokenizer.json` file in order to replicate the
/// original tiktoken format.
struct llama3_tokenizer_loader {
    using tokenizer_type = text::byte_pair_encoder<text::regexp>;

    /// Load the tokenizer from the specified input stream.
    ///
    /// \param is An input stream containing a JSON-encoded tokenizer model (HuggingFace format).
    tokenizer_type
    load(std::istream& is) const;

    /// Load the tokenizer from the specified local file.
    ///
    /// \param p A path to the JSON-encoded tokenizer model (HuggingFace format).
    tokenizer_type
    load(const std::filesystem::path& p) const;
};


} // namespace huggingface
} // namespace metalchat
