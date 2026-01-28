// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <istream>
#include <string_view>

#include <metalchat/nn.h>
#include <metalchat/safetensor.h>
#include <metalchat/tensor/concept.h>
#include <metalchat/text.h>
#include <metalchat/transformer.h>


namespace metalchat {
namespace reference {


/// The reference implementation of Llama 3.2 shares the weight of token embeddings and the output
/// layer, use a shared tensor in order to reduce memory footprint.
///
/// This adaptor implement \ref safetensor_document_adaptor concept and creates an  alias between
/// output and embedding layers. The rest of the tensors remains unchanged.
struct llama3_document_adaptor {
    safetensor_document
    adapt(const safetensor_document& document) const;
};


/// The reference Llama3.2 options loader. This serializers provides support of loading
/// and saving LLama3 options from a Meta Llama JSON format.
struct llama3_options_serializer {
    using value_type = nn::llama3_options;

    value_type
    load(std::istream&) const;

    void
    save(std::ostream& os, const value_type&) const;
};


/// The reference implementation of the Llama3 tokenizer.
///
/// This loader implements loading of a tokenizer model in a reference (tiktoken) format. It
/// expects that `load` methods receives a file in a tiktoken format.
struct llama3_tokenizer_loader {
    using type = text::byte_pair_encoder<text::regexp>;

    /// A regular expression string that is used to split the input text into tokens.
    // clang-format off
    static constexpr std::string_view default_regex =
        (R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|)"
         R"([^\r\n\p{L}\p{N}]?\p{L}+|)"
         R"(\p{N}{1,3}|)"
         R"( ?[^\s\p{L}\p{N}]+[\r\n]*|)"
         R"(\s*[\r\n]+|)"
         R"(\s+(?!\S)|)"
         R"(\s+)");
    // clang-format on

    /// Load a tokenizer from the input stream.
    ///
    /// \param is An input stream containing tokenizer model (tiktoken format).
    /// \param token_regex A regular expression used to split a string into tokens.
    type
    load(std::istream& is, const std::string& token_regex) const;

    /// Load a tokenizer from the local file.
    ///
    /// \param p A path to the file containing tokenizer model (tiktoken format).
    /// \param token_regex A regular expression used to split a string into tokens.
    type
    load(const std::filesystem::path& p, const std::string& token_regex) const;

    /// Load a tokenizer from the input stream.
    ///
    /// The implementation uses a \ref default_regex to split sentence into tokens.
    ///
    /// See also \ref load(std::istream&, const std::string&) const.
    type
    load(std::istream& is) const;

    /// Load a tokenizer from the local file.
    ///
    /// The implementation uses a \ref default_regex to split sentence into tokens.
    ///
    /// See also \ref load(const std::filesystem::path&, const std::string&) const.
    type
    load(const std::filesystem::path& p) const;

    static void
    insert_control_tokens(type& bpe);
};


template <typename T, contiguous_container Container> struct llama3_traits {
    using value_type = T;
    using options_type = nn::llama3_options;
    using options_serializer = llama3_options_serializer;
    using layer_type = nn::llama3<T, Container>;
    using layer_adaptor = noop_layer_adaptor<options_type>;
    using container_type = Container;
    using document_adaptor = llama3_document_adaptor;
    using tokenizer_type = text::byte_pair_encoder<text::regexp>;
    using tokenizer_loader = llama3_tokenizer_loader;
};


using llama3 = llama3_traits<bf16, hardware_memory_container<bf16>>;


} // namespace reference
} // namespace metalchat
