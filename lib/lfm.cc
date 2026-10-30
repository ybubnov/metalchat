// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <jsoncons/json.hpp>

#include <metalchat/huggingface/lfm.h>

#include "huggingface.h"


namespace metalchat {
namespace huggingface {


nn::lfm2_options
lfm2_options_serializer::load(std::istream& is) const
{
    using options_type = metalchat::huggingface::detail::lfm2_options;
    auto options = jsoncons::decode_json<options_type>(is);

    std::vector<nn::attentionkind> attentions;
    for (const auto& layer_type : options.layer_types) {
        attentions.push_back(
            layer_type == "full_attention" ? nn::attention::multihead : nn::attention::recurrent
        );
    }

    return nn::lfm2_options{
        .head_dim = options.hidden_size / options.num_attention_heads,
        .hidden_dim = options.hidden_size,
        .n_heads = options.num_attention_heads,
        .n_kv_heads = options.num_key_value_heads,
        .kernel_size = options.conv_L_cache,
        .max_seq_len = 1024,
        .max_batch_size = 1,
        .rope_theta = options.rope_parameters.rope_theta,
        .norm_eps = options.norm_eps,
        .attentions = attentions
    };
}


void
lfm2_options_serializer::save(std::ostream& os, const nn::lfm2_options& options) const
{}


} // namespace huggingface
} // namespace metalchat
