// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <codecvt>
#include <locale>

#include <jsoncons/json.hpp>

#include <metalchat/huggingface/gemma.h>

#include "huggingface.h"


namespace metalchat {
namespace huggingface {


nn::gemma3_options
gemma3_options_serializer::load(std::istream& is) const
{
    using options_type = metalchat::huggingface::detail::gemma3_options;
    auto options = jsoncons::decode_json<options_type>(is);

    auto sliding_stride =
        options._sliding_window_pattern.value_or(options._sliding_window_pattern.value_or(0));

    return nn::gemma3_options{
        .head_dim = options.head_dim,
        .hidden_dim = options.hidden_size,
        .n_heads = options.num_attention_heads,
        .n_kv_heads = options.num_key_value_heads,
        .n_layers = options.num_hidden_layers,
        .max_seq_len = 1024,
        .sliding_window = options.sliding_window,
        .sliding_stride = sliding_stride,
        .attn_scale = options.query_pre_attn_scalar,
        .rope_theta = options.rope_theta,
        .rope_sliding_theta = options.rope_local_base_freq,
        .norm_eps = options.rms_norm_eps,
    };
}


void
gemma3_options_serializer::save(std::ostream& os, const nn::gemma3_options& options) const
{
    using options_type = metalchat::huggingface::detail::gemma3_options;

    auto hf_options = options_type{
        .head_dim = options.head_dim,
        .hidden_size = options.hidden_dim,
        .num_hidden_layers = options.n_layers,
        .num_attention_heads = options.n_heads,
        .num_key_value_heads = options.n_kv_heads,
        .sliding_window = options.sliding_window,
        .sliding_window_pattern = options.sliding_stride,
        .query_pre_attn_scalar = options.attn_scale,
        .rms_norm_eps = options.norm_eps,
        .rope_theta = options.rope_theta,
        .rope_local_base_freq = options.rope_sliding_theta
    };

    auto encode_options = jsoncons::json_options()
                              .float_format(jsoncons::float_chars_format::general)
                              .precision(os.precision());

    jsoncons::encode_json<options_type>(hf_options, os, encode_options);
}


gemma3_tokenizer_loader::type
gemma3_tokenizer_loader::load(std::istream& is) const
{
    using model_type = metalchat::huggingface::detail::tokenizer;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    using codecvt_type = std::codecvt_utf8<char32_t>;
    using convert_type = std::wstring_convert<codecvt_type, char32_t>;
    convert_type convert;
#pragma clang diagnostic pop

    auto model_file = jsoncons::decode_json<model_type>(is);
    gemma3_tokenizer_loader::type tokenizer;

    for (const auto& [value, key] : model_file.model.vocab) {
        tokenizer.insert(convert.from_bytes(value), key, text::token::regular);
    }
    for (const auto& token : model_file.added_tokens) {
        tokenizer.insert(convert.from_bytes(token.content), token.id, text::tokenkind(token.id));
    }

    return tokenizer;
}

gemma3_tokenizer_loader::type
gemma3_tokenizer_loader::load(const std::filesystem::path& p) const
{
    std::ifstream file(p, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        throw std::invalid_argument(
            std::format("gemma3_tokenizer_loader: failed opening file '{}'", p.string())
        );
    }

    return load(file);
}


} // namespace huggingface
} // namespace metalchat
