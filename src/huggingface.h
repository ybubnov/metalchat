// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <map>
#include <variant>
#include <vector>

#include <jsoncons/json.hpp>


namespace metalchat {
namespace huggingface {
namespace detail {


/// Partial view of the HuggingFace's model of the LLM configuration. The original model
/// supports more parameters. But MetalChat has no use of that, therefore only supported
/// subset of options is defined in this structure.
struct options {
    std::size_t head_dim;
    std::size_t num_hidden_layers;
    std::size_t num_attention_heads;
    std::size_t num_key_value_heads;
    float rms_norm_eps;
    float rope_theta;
};


struct special_token {
    std::string id;
    std::string content;
    bool single_word;
    bool lstrip;
    bool rstrip;
    bool normalized;
    bool special;
};


struct split_pattern {
    std::string Regex;
};


struct split_tokenizer {
    std::string type;
    std::string behavior;
    split_pattern pattern;
    bool invert;
};


struct bytelevel_tokenizer {
    std::string type;
    bool add_prefix_space;
    bool trim_offsets;
    bool use_regex;
};


struct sequence_tokenizer {
    std::string type;
    std::vector<std::variant<split_tokenizer, bytelevel_tokenizer>> pretokenizers;
};


struct bpe {
    std::string type;
    bool fuse_unk;
    bool byte_fallback;
    bool ignore_merges;
    std::map<std::string, int32_t> vocab;
    std::vector<std::string> merges;
};


/// The format of the HuggingFace's tokenizer model.
///
/// The purpose of this model is to adapt HuggingFace's tokenizer to the MetalChat tokenizer,
/// therefore some of the fields are missing, since they won't be supported by the MetalChat
/// anyway. Here it lists only the parameters could be used for creating Llama3 tokenizer.
///
/// Effectively, HuggingFace supports various tokenization models, here we expect only BPE model.
struct tokenizer {
    std::string version;
    std::vector<special_token> added_tokens;
    sequence_tokenizer pre_tokenizer;
    bpe model;
};


} // namespace detail
} // namespace huggingface
} // namespace metalchat


namespace hf = metalchat::huggingface::detail;


// clang-format off
JSONCONS_ALL_MEMBER_TRAITS(
    hf::options,
    head_dim,
    num_hidden_layers,
    num_attention_heads,
    num_key_value_heads,
    rms_norm_eps,
    rope_theta
);
JSONCONS_ALL_MEMBER_TRAITS(hf::special_token, id, content, single_word, lstrip, rstrip, normalized, special);
JSONCONS_ALL_MEMBER_TRAITS(hf::split_pattern, Regex);
JSONCONS_ALL_MEMBER_TRAITS(hf::split_tokenizer, type, behavior, pattern, invert);
JSONCONS_ALL_MEMBER_TRAITS(hf::bytelevel_tokenizer, type, add_prefix_space, trim_offsets, use_regex);
JSONCONS_ALL_MEMBER_TRAITS(hf::sequence_tokenizer, type, pretokenizers);
JSONCONS_ALL_MEMBER_TRAITS(hf::bpe, fuse_unk, byte_fallback, ignore_merges, vocab, merges);
JSONCONS_ALL_MEMBER_TRAITS(hf::tokenizer, version, added_tokens, pre_tokenizer, model);
// clang-format on
