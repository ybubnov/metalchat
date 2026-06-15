// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <jsoncons/json.hpp>

#include <metalchat/huggingface/llama.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/reference.h>
#include <metalchat/tensor/accessor.h>

#include "huggingface.h"


namespace metalchat {
namespace nn {


llama3_options
default_llama3_1b_options()
{
    return llama3_options{
        .head_dim = 64,
        .n_heads = 32,
        .n_kv_heads = 8,
        .n_layers = 16,
        .max_seq_len = 1024,
        .rope_theta = 500000.0f,
        .norm_eps = 1e-5f,
    };
}


} // namespace nn


namespace huggingface {


nn::llama3_options
llama3_options_serializer::load(std::istream& is) const
{
    using options_type = metalchat::huggingface::detail::llama3_options;
    auto options = jsoncons::decode_json<options_type>(is);

    return nn::llama3_options{
        .head_dim = options.head_dim,
        .n_heads = options.num_attention_heads,
        .n_kv_heads = options.num_key_value_heads,
        .n_layers = options.num_hidden_layers,
        .max_seq_len = 1024,
        .rope_theta = options.rope_theta,
        .norm_eps = options.rms_norm_eps
    };
}


void
llama3_options_serializer::save(std::ostream& os, const nn::llama3_options& options) const
{
    using options_type = metalchat::huggingface::detail::llama3_options;

    auto hf_options = options_type{
        .head_dim = options.head_dim,
        .num_hidden_layers = options.n_layers,
        .num_attention_heads = options.n_heads,
        .num_key_value_heads = options.n_kv_heads,
        .rms_norm_eps = options.norm_eps,
        .rope_theta = options.rope_theta
    };

    // Ensure that output JSON follows output stream float precision format.
    auto encode_options = jsoncons::json_options()
                              .float_format(jsoncons::float_chars_format::general)
                              .precision(os.precision());

    jsoncons::encode_json<options_type>(hf_options, os, encode_options);
}


llama3_tokenizer_loader::type
llama3_tokenizer_loader::load(std::istream& is) const
{
    namespace hf = metalchat::huggingface::detail;
    auto model_file = jsoncons::decode_json<hf::tokenizer>(is);
    std::string token_regex;

    if (const auto seq = std::get_if<hf::sequence_tokenizer>(&model_file.pre_tokenizer)) {
        for (const auto tokenizer : seq->pretokenizers) {
            if (const auto split = std::get_if<hf::split_tokenizer>(&tokenizer)) {
                token_regex = split->pattern.Regex;
                break;
            }
        }
    }

    if (token_regex.empty()) {
        throw std::runtime_error(
            "llama3_tokenizer_loader::load: the JSON encoding does not provide "
            "an input sequence regular expression"
        );
    }

    text::gpt2_codec codec;
    llama3_tokenizer_loader::type tokenizer(token_regex);

    for (const auto& [value, key] : model_file.model.vocab) {
        auto val = codec.decode(value);
        tokenizer.insert(val, key, text::token::regular);
    }

    using loader_type = metalchat::reference::llama3_tokenizer_loader;
    loader_type::insert_control_tokens(tokenizer);
    return tokenizer;
}


llama3_tokenizer_loader::type
llama3_tokenizer_loader::load(const std::filesystem::path& p) const
{
    std::ifstream file(p, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        throw std::invalid_argument(
            std::format("llama3_tokenizer_loader: failed opening file '{}'", p.string())
        );
    }

    return load(file);
}


} // namespace huggingface
} // namespace metalchat
