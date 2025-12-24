// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <jsoncons/json.hpp>

#include <metalchat/autoloader.h>
#include <metalchat/nn/options.h>


namespace metalchat {
namespace detail {

struct llama3_reference_options {
    std::size_t dim;
    std::size_t n_layers;
    std::size_t n_heads;
    std::size_t n_kv_heads;
    std::size_t vocab_size;
    float ffn_dim_multiplier;
    std::size_t multiple_of;
    float norm_eps;
    float rope_theta;
    bool use_scaled_rope;
};


} // namespace detail
} // namespace metalchat


JSONCONS_ALL_MEMBER_TRAITS(
    metalchat::detail::llama3_reference_options,
    dim,
    n_layers,
    n_heads,
    n_kv_heads,
    vocab_size,
    ffn_dim_multiplier,
    multiple_of,
    norm_eps,
    rope_theta,
    use_scaled_rope
);


namespace metalchat {
namespace reference {


safetensor_document
llama3_document_adaptor::adapt(const safetensor_document& document) const
{
    auto doc = document;
    doc.insert("output.weight", "tok_embeddings.weight");
    return doc;
}


nn::llama3_options
llama3_options_loader::load(std::istream& is) const
{
    using options_type = detail::llama3_reference_options;
    auto options_value = jsoncons::decode_json<options_type>(is);

    auto options = nn::llama3_options()
                       .head_dim(options_value.dim / options_value.n_heads)
                       .n_layers(options_value.n_layers)
                       .n_heads(options_value.n_heads)
                       .n_kv_heads(options_value.n_kv_heads)
                       .rope_theta(options_value.rope_theta)
                       .norm_eps(options_value.norm_eps);

    return options;
}


} // namespace reference
} // namespace metalchat
