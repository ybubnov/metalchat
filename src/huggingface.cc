#include <map>
#include <regex>
#include <variant>

#include <jsoncons/json.hpp>

#include <metalchat/huggingface.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/reference.h>
#include <metalchat/tensor/accessor.h>


namespace metalchat {
namespace detail {
namespace hf {


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


} // namespace hf
} // namespace detail
} // namespace metalchat


using namespace metalchat::detail;


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


namespace metalchat {
namespace huggingface {


nn::llama3_options
llama3_options_serializer::load(std::istream& is) const
{
    using options_type = metalchat::detail::hf::options;
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
    using options_type = metalchat::detail::hf::options;

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
    using model_type = metalchat::detail::hf::tokenizer;
    using split_tokenizer_type = metalchat::detail::hf::split_tokenizer;
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
