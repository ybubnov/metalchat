// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <memory>
#include <string_view>

#include <metalchat/format.h>
#include <metalchat/nn/gemma.h>
#include <metalchat/safetensor.h>
#include <metalchat/text.h>


namespace metalchat {
namespace huggingface {


using namespace std::literals;


/// Gemma3 options serializer for configurations distributed through HuggingFace repository.
struct gemma3_options_serializer {
    using value_type = nn::gemma3_options;

    nn::gemma3_options
    load(std::istream& is) const;

    void
    save(std::ostream& os, const nn::gemma3_options& options) const;
};


template <typename T, nn::mutable_layer Layer> class gemma3_safetensor_serializer {
public:
    using value_type = nn::indirect_layer<Layer>;

    /// Creates a new instance of a layer serializer with Gemma3 options.
    gemma3_safetensor_serializer(
        const nn::gemma3_options& options, hardware_accelerator& accelerator
    )
    : _M_options(options),
      _M_accelerator(accelerator)
    {}

    value_type
    load(safetensor_document& document)
    {
        value_type layer(_M_options, _M_accelerator);

        auto doc = adapt(document);
        doc.load(layer);

        return layer;
    }

    void
    save(safetensor_document& document, value_type layer)
    {
        document.save(layer);
    }

    safetensor_document
    adapt(const safetensor_document& document) const
    {
        const std::vector<std::pair<std::regex, std::string>> mapping = {
            {std::regex(R"(model\.(layers\.\d+)\.input_layernorm)"), "$1.attention_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.post_attention_layernorm)"),
             "$1.attention_post_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.pre_feedforward_layernorm)"), "$1.ffn_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.post_feedforward_layernorm)"), "$1.ffn_post_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.mlp\.gate_proj)"), "$1.feed_forward.w1"},
            {std::regex(R"(model\.(layers\.\d+)\.mlp\.down_proj)"), "$1.feed_forward.w2"},
            {std::regex(R"(model\.(layers\.\d+)\.mlp\.up_proj)"), "$1.feed_forward.w3"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.q_proj)"), "$1.attention.wq"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.q_norm)"), "$1.attention.q_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.k_proj)"), "$1.attention.wk"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.k_norm)"), "$1.attention.k_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.v_proj)"), "$1.attention.wv"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.o_proj)"), "$1.attention.wo"},
            {std::regex(R"(model.norm)"), "norm"},
            {std::regex(R"(model.embed_tokens)"), "tok_embeddings"},
        };

        auto doc = document.rename(mapping.begin(), mapping.end());
        doc.insert("output.weight", "tok_embeddings.weight");

        return doc;
    }

private:
    nn::gemma3_options _M_options;
    hardware_accelerator _M_accelerator;
};


struct gemma3_tokenizer_loader {
    using type = text::unicode_tokenizer_adaptor<text::sentence_piece>;

    type
    load(std::istream& is) const;

    type
    load(const std::filesystem::path& p) const;
};


struct gemma3_prompt {
    static constexpr std::string_view pad = "<pad>";
    static constexpr std::string_view eos = "<eos>";
    static constexpr std::string_view bos = "<bos>";
    static constexpr std::string_view unk = "<unk>";
    static constexpr std::string_view mask = "<mask>";
    static constexpr std::string_view begin_turn = "<start_of_turn>";
    static constexpr std::string_view end_turn = "<end_of_turn>";
};


template <typename Tokenizer> class gemma3_formatter : public basic_formatter<int32_t, char> {
public:
    static constexpr auto default_roles = std::make_tuple(
        std::make_pair(role::system, "user"sv),
        std::make_pair(role::request, "user"sv),
        std::make_pair(role::response, "model"sv),
        std::make_pair(role::result, "user"sv)
    );

    using index_type = int32_t;
    using char_type = char;

    using formatter_type = basic_formatter<index_type, char_type>;
    using scanner_type = basic_token_scanner<index_type>;
    using scanner_pointer = std::shared_ptr<scanner_type>;

    using prompt_type = gemma3_prompt;
    using tokenizer_type = Tokenizer;
    using tokenizer_traits = text::tokenizer_traits<tokenizer_type>;

    using istream_type = formatter_type::istream_type;
    using ostream_type = formatter_type::ostream_type;
    using message_type = formatter_type::message_type;

    gemma3_formatter(const Tokenizer& tokenizer)
    : _M_tokenizer(tokenizer),
      _M_scanner(nullptr),
      _M_first(false),
      _M_format_roles()
    {
        constexpr auto default_roles_size = std::tuple_size_v<decltype(default_roles)>;
        register_default_roles(std::make_index_sequence<default_roles_size>{});

        std::vector<index_type> terminals;
        auto terminals_iterator = std::back_inserter(terminals);

        tokenizer_traits::encode(_M_tokenizer, prompt_type::end_turn, terminals_iterator);
        tokenizer_traits::encode(_M_tokenizer, prompt_type::eos, terminals_iterator);

        auto scanner = make_default_scanner(terminals.cbegin(), terminals.cend());
        _M_scanner = std::make_shared<decltype(scanner)>(std::move(scanner));
    }

    message_type
    parse(istream_type& is)
    {
        std::basic_stringstream<char_type> content_stream;
        parse(is, content_stream);

        return message(role::response, content_stream.str());
    }

    void
    parse(istream_type& is, std::basic_ostream<char_type>& os)
    {
        std::istreambuf_iterator<index_type> input(is);
        std::ostreambuf_iterator<char_type> output(os);

        parse(input, output);
    }

    template <std::input_iterator InputIt, std::output_iterator<char_type> OutputIt>
    void
    parse(InputIt input, OutputIt output)
    {
        _M_scanner->reset();
        for (auto token = *input; _M_scanner->scan(token); token = *++input) {
            auto str = tokenizer_traits::decode(_M_tokenizer, token);
            std::copy(str.cbegin(), str.cend(), output);
        }
        ++input;
    }

    void
    format(const message_type& message, ostream_type& os)
    {
        std::ostreambuf_iterator<index_type> output(os);
        // Begin formatting with the begin-of-text token.
        if (!_M_first) {
            tokenizer_traits::encode(_M_tokenizer, gemma3_prompt::bos, output);
            _M_first = true;
        }

        format_header(message.role(), output);
        format_content(message.content(), output);
    }

private:
    template <typename OutputIt>
    void
    format_header(rolekind role, OutputIt output) const
    {
        if (auto it = _M_format_roles.find(role); it != _M_format_roles.end()) {
            tokenizer_traits::encode(_M_tokenizer, it->second, output);
            tokenizer_traits::encode(_M_tokenizer, "\n", output);
        } else {
            throw std::runtime_error(std::format("gemma3_formatter: role {} not found", role));
        }
    }

    template <typename OutputIt>
    void
    format_content(const std::basic_string<char_type>& content, OutputIt output) const
    {
        if (!content.empty()) {
            tokenizer_traits::encode(_M_tokenizer, content, output);
            tokenizer_traits::encode(_M_tokenizer, gemma3_prompt::end_turn, output);
            tokenizer_traits::encode(_M_tokenizer, "\n", output);
        }
    }

    template <std::size_t... Indices>
    void
    register_default_roles(std::index_sequence<Indices...>)
    {
        (register_default_role<Indices>(), ...);
    }

    template <std::size_t Index>
    void
    register_default_role()
    {
        auto default_role = std::get<Index>(default_roles);
        auto role = default_role.first;
        auto role_synonym = std::string(default_role.second);

        _M_format_roles.insert_or_assign(role, role_synonym);
    }

    tokenizer_type _M_tokenizer;
    scanner_pointer _M_scanner;
    bool _M_first;

    std::unordered_map<rolekind, std::string> _M_format_roles;
};


template <contiguous_container Container> struct gemma3_traits {
    using value_type = Container::value_type;
    using container_type = Container;

    using layer_type = nn::gemma3<value_type, Container>;
    using layer_serializer = gemma3_safetensor_serializer<value_type, layer_type>;

    using options_type = nn::gemma3_options;
    using options_serializer = gemma3_options_serializer;

    using tokenizer_type = text::unicode_tokenizer_adaptor<text::sentence_piece>;
    using tokenizer_loader = gemma3_tokenizer_loader;
    using formatter_type = gemma3_formatter<tokenizer_type>;

    static constexpr std::string_view tokenizer_location = "tokenizer.json";
    static constexpr std::string_view options_location = "config.json";
    static constexpr std::string_view transformer_location = "model.safetensors";
};


using gemma3 = gemma3_traits<hardware_memory_container<bf16>>;


} // namespace huggingface
} // namespace metalchat
