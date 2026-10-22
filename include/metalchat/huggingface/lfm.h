// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <memory>
#include <string_view>

#include <metalchat/format.h>
#include <metalchat/huggingface/llama.h>
#include <metalchat/nn/lfm.h>
#include <metalchat/safetensor.h>
#include <metalchat/text.h>


namespace metalchat {
namespace huggingface {


using namespace std::literals;


/// LFM2 options serializer for configurations distributed through HuggingFace repository.
struct lfm2_options_serializer {
    using value_type = nn::lfm2_options;

    nn::lfm2_options
    load(std::istream& is) const;

    void
    save(std::ostream& os, const nn::lfm2_options& options) const;
};


template <nn::mutable_layer Layer> class lfm2_safetensor_serializer {
public:
    using value_type = nn::indirect_layer<Layer>;

    void
    load(const safetensor_document& document, value_type& layer) const
    {
        auto doc = adapt(document);
        doc.load(layer);
    }

    void
    save(safetensor_document& document, const value_type& layer) const
    {
        document.save(layer);
    }

    safetensor_document
    adapt(const safetensor_document& document) const
    {
        const std::vector<std::pair<std::regex, std::string>> mapping = {
            {std::regex(R"(model\.(layers\.\d+)\.conv\.in_proj)"), "$1.attention.wq"},
            {std::regex(R"(model\.(layers\.\d+)\.conv\.out_proj)"), "$1.attention.wo"},
            {std::regex(R"(model\.(layers\.\d+)\.conv\.conv)"), "$1.attention.conv"},
            {std::regex(R"(model\.(layers\.\d+)\.operator_norm)"), "$1.attention_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.q_proj)"), "$1.attention.wq"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.q_layernorm)"), "$1.attention.q_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.k_proj)"), "$1.attention.wk"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.k_layernorm)"), "$1.attention.k_norm"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.v_proj)"), "$1.attention.wv"},
            {std::regex(R"(model\.(layers\.\d+)\.self_attn\.out_proj)"), "$1.attention.wo"},
            {std::regex(R"(model\.(layers\.\d+))"), "$1"},
            {std::regex(R"(model.embedding_norm)"), "norm"},
            {std::regex(R"(model.embed_tokens)"), "tok_embeddings"}
        };

        auto doc = document.rename(mapping.begin(), mapping.end());
        doc.insert("output.weight", "tok_embeddings.weight");

        return doc;
    }
};


struct lfm2_prompt {
    static constexpr std::string_view start_of_text = "<|startoftext|>";
    static constexpr std::string_view end_of_text = "<|endoftext|>";
    static constexpr std::string_view message_start = "<|im_start|>";
    static constexpr std::string_view message_end = "<|im_end|>";
    static constexpr std::string_view tool_list_start = "<|tool_list_start|>";
    static constexpr std::string_view tool_list_end = "<|tool_list_end|>";
    static constexpr std::string_view tool_call_start = "<|tool_call_start|>";
    static constexpr std::string_view tool_call_end = "<|tool_call_end|>";
};


template <typename Tokenizer> class lfm2_formatter : public basic_formatter<int32_t, char> {
public:
    static constexpr auto default_roles = std::make_tuple(
        std::make_pair(role::system, "system"sv),
        std::make_pair(role::request, "user"sv),
        std::make_pair(role::response, "assistant"sv),
        std::make_pair(role::result, "tool"sv)
    );

    using index_type = int32_t;
    using char_type = char;

    using formatter_type = basic_formatter<index_type, char_type>;
    using scanner_type = basic_token_scanner<index_type>;
    using scanner_pointer = std::shared_ptr<scanner_type>;

    using prompt_type = lfm2_prompt;
    using tokenizer_type = Tokenizer;
    using tokenizer_traits = text::tokenizer_traits<tokenizer_type>;

    using istream_type = formatter_type::istream_type;
    using ostream_type = formatter_type::ostream_type;
    using message_type = formatter_type::message_type;

    lfm2_formatter(const Tokenizer& tokenizer)
    : basic_formatter(),
      _M_tokenizer(tokenizer),
      _M_scanner(nullptr),
      _M_first(false)
    {
        register_message_roles(default_roles);

        std::vector<index_type> terminals;
        auto terminals_iterator = std::back_inserter(terminals);

        tokenizer_traits::encode(_M_tokenizer, prompt_type::message_end, terminals_iterator);
        tokenizer_traits::encode(_M_tokenizer, prompt_type::end_of_text, terminals_iterator);

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
        std::istreambuf_iterator<index_type, token_traits<index_type>> input(is);
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
        std::ostreambuf_iterator<index_type, token_traits<index_type>> output(os);
        // Begin formatting with the begin-of-text token.
        if (!_M_first) {
            tokenizer_traits::encode(_M_tokenizer, prompt_type::start_of_text, output);
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
        tokenizer_traits::encode(_M_tokenizer, prompt_type::message_start, output);
        tokenizer_traits::encode(_M_tokenizer, message_role(role), output);
        tokenizer_traits::encode(_M_tokenizer, "\n", output);
    }

    template <typename OutputIt>
    void
    format_content(const std::basic_string<char_type>& content, OutputIt output) const
    {
        if (!content.empty()) {
            tokenizer_traits::encode(_M_tokenizer, content, output);
            tokenizer_traits::encode(_M_tokenizer, prompt_type::message_end, output);
            tokenizer_traits::encode(_M_tokenizer, "\n", output);
        }
    }

    tokenizer_type _M_tokenizer;
    scanner_pointer _M_scanner;
    bool _M_first;
};


template <contiguous_container Container> struct lfm2_traits {
    using value_type = Container::value_type;
    using container_type = Container;

    using layer_type = nn::lfm2<value_type, Container>;
    using layer_serializer = lfm2_safetensor_serializer<layer_type>;

    using options_type = nn::lfm2_options;
    using options_serializer = lfm2_options_serializer;

    using tokenizer_type = text::byte_pair_encoder<char>;
    using tokenizer_loader = llama3_tokenizer_loader;
    using formatter_type = lfm2_formatter<tokenizer_type>;

    static constexpr std::string_view tokenizer_location = "tokenizer.json";
    static constexpr std::string_view options_location = "config.json";
    static constexpr std::string_view transformer_location = "model.safetensors";
};


using lfm2 = lfm2_traits<hardware_memory_container<bf16>>;


} // namespace huggingface
} // namespace metalchat
