// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <jsoncons/json.hpp>

#include <metalchat/nn/options.h>
#include <metalchat/reference.h>


namespace metalchat {
namespace detail {

struct llama3_reference_options {
    std::size_t dim;
    std::size_t n_layers;
    std::size_t n_heads;
    std::size_t n_kv_heads;
    std::size_t vocab_size;
    double ffn_dim_multiplier;
    std::size_t multiple_of;
    double norm_eps;
    double rope_theta;
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


nn::llama3_options
llama3_options_serializer::load(std::istream& is) const
{
    using options_type = metalchat::detail::llama3_reference_options;
    auto options = jsoncons::decode_json<options_type>(is);

    return nn::llama3_options()
        .head_dim(options.dim / options.n_heads)
        .n_layers(options.n_layers)
        .n_heads(options.n_heads)
        .n_kv_heads(options.n_kv_heads)
        .rope_theta(options.rope_theta)
        .norm_eps(options.norm_eps);
}


void
llama3_options_serializer::save(std::ostream&, const nn::llama3_options&) const
{
    throw std::runtime_error("not implemented");
}


llama3_tokenizer_loader::type
llama3_tokenizer_loader::load(std::istream& is, const std::string& token_regex) const
{
    type tokenizer(is, token_regex);
    insert_control_tokens(tokenizer);
    return tokenizer;
}


llama3_tokenizer_loader::type
llama3_tokenizer_loader::load(const std::filesystem::path& p, const std::string& token_regex) const
{
    std::ifstream file(p, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        throw std::invalid_argument(
            std::format("llama3_tokenizer_loader: failed opening file '{}'", p.string())
        );
    }

    return load(file, token_regex);
}


llama3_tokenizer_loader::type
llama3_tokenizer_loader::load(std::istream& is) const
{
    return load(is, std::string(default_regex));
}


llama3_tokenizer_loader::type
llama3_tokenizer_loader::load(const std::filesystem::path& p) const
{
    return load(p, std::string(default_regex));
}


void
llama3_tokenizer_loader::insert_control_tokens(type& tokenizer)
{
    tokenizer.insert_back("<|begin_of_text|>", text::token::begin_text);
    tokenizer.insert_back("<|end_of_text|>", text::token::end_text);
    tokenizer.insert_back(text::make_reserved_token(0), text::token::reserved);
    tokenizer.insert_back(text::make_reserved_token(1), text::token::reserved);
    tokenizer.insert_back("<|finetune_right_pad_id|>", text::token::finetune_right_pad);
    tokenizer.insert_back(text::make_reserved_token(2), text::token::reserved);
    tokenizer.insert_back("<|start_header_id|>", text::token::begin_header);
    tokenizer.insert_back("<|end_header_id|>", text::token::end_header);
    tokenizer.insert_back("<|eom_id|>", text::token::end_message);
    tokenizer.insert_back("<|eot_id|>", text::token::end_turn);
    tokenizer.insert_back("<|python_tag|>", text::token::ipython);
}


} // namespace reference
} // namespace metalchat
