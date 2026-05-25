// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <jsoncons/json.hpp>

#include <metalchat/huggingface/llama.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/nn/options.h>
#include <metalchat/reference.h>
#include <metalchat/tensor/accessor.h>

#include "huggingface.h"


namespace metalchat {
namespace nn {


llama3_options
default_llama3_1b_options()
{
    return llama3_options()
        .head_dim(64)
        .n_heads(32)
        .n_kv_heads(8)
        .n_layers(16)
        .max_seq_len(1024)
        .rope_theta(500000.0f)
        .norm_eps(1e-5);
}


llama3_options::llama3_options()
: _M_max_seq_len(1024)
{}


llama3_options
llama3_options::head_dim(std::size_t value) const noexcept
{
    llama3_options o = *this;
    o._M_head_dim = value;
    return o;
}


std::size_t
llama3_options::head_dim() const noexcept
{
    return _M_head_dim;
}


llama3_options
llama3_options::n_heads(std::size_t value) const noexcept
{
    llama3_options o = *this;
    o._M_n_heads = value;
    return o;
}


std::size_t
llama3_options::n_heads() const noexcept
{
    return _M_n_heads;
}


llama3_options
llama3_options::n_kv_heads(std::size_t value) const noexcept
{
    llama3_options o = *this;
    o._M_n_kv_heads = value;
    return o;
}


std::size_t
llama3_options::n_kv_heads() const noexcept
{
    return _M_n_kv_heads;
}


llama3_options
llama3_options::n_layers(std::size_t value) const noexcept
{
    llama3_options o = *this;
    o._M_n_layers = value;
    return o;
}


std::size_t
llama3_options::n_layers() const noexcept
{
    return _M_n_layers;
}


llama3_options
llama3_options::max_seq_len(std::size_t value) const noexcept
{
    llama3_options o = *this;
    o._M_max_seq_len = value;
    return o;
}


std::size_t
llama3_options::max_seq_len() const noexcept
{
    return _M_max_seq_len;
}


llama3_options
llama3_options::rope_theta(float value) const noexcept
{
    llama3_options o = *this;
    o._M_rope_theta = value;
    return o;
}


float
llama3_options::rope_theta() const noexcept
{
    return _M_rope_theta;
}


llama3_options
llama3_options::norm_eps(float value) const noexcept
{
    llama3_options o = *this;
    o._M_norm_eps = value;
    return o;
}


float
llama3_options::norm_eps() const noexcept
{
    return _M_norm_eps;
}


} // namespace nn


namespace huggingface {


nn::llama3_options
llama3_options_serializer::load(std::istream& is) const
{
    using options_type = metalchat::huggingface::detail::options;
    auto options = jsoncons::decode_json<options_type>(is);

    return nn::llama3_options()
        .head_dim(options.head_dim)
        .n_layers(options.num_hidden_layers)
        .n_heads(options.num_attention_heads)
        .n_kv_heads(options.num_key_value_heads)
        .rope_theta(options.rope_theta)
        .norm_eps(options.rms_norm_eps);
}


void
llama3_options_serializer::save(std::ostream& os, const nn::llama3_options& options) const
{
    using options_type = metalchat::huggingface::detail::options;

    auto hf_options = options_type{
        .head_dim = options.head_dim(),
        .num_hidden_layers = options.n_layers(),
        .num_attention_heads = options.n_heads(),
        .num_key_value_heads = options.n_kv_heads(),
        .rms_norm_eps = options.norm_eps(),
        .rope_theta = options.rope_theta()
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
    using model_type = metalchat::huggingface::detail::tokenizer;
    using split_tokenizer_type = metalchat::huggingface::detail::split_tokenizer;
    using loader_type = metalchat::reference::llama3_tokenizer_loader;

    auto model_file = jsoncons::decode_json<model_type>(is);
    std::string token_regex;

    for (const auto& tokenizer : model_file.pre_tokenizer.pretokenizers) {
        if (const auto split = std::get_if<split_tokenizer_type>(&tokenizer)) {
            token_regex = split->pattern.Regex;
            break;
        }
    }

    text::gpt2_codec codec;
    llama3_tokenizer_loader::type tokenizer(token_regex);

    for (const auto& [value, key] : model_file.model.vocab) {
        auto val = codec.decode(value);
        tokenizer.insert(val, key, text::token::regular);
    }

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
