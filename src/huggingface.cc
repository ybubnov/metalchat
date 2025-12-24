#include <regex>

#include <metalchat/huggingface/llama.h>
#include <metalchat/kernel/copy.h>
#include <metalchat/tensor/accessor.h>


namespace metalchat {
namespace huggingface {


safetensor_document
llama3_document_adaptor::adapt(const safetensor_document& document) const
{
    safetensor_document doc;

    const std::vector<std::pair<std::regex, std::string>> mapping = {
        {std::regex(R"(model\.(layers\.\d+)\.input_layernorm)"), "$1.attention_norm"},
        {std::regex(R"(model\.(layers\.\d+)\.post_attention_layernorm)"), "$1.ffn_norm"},
        {std::regex(R"(model\.(layers\.\d+)\.mlp\.gate_proj)"), "$1.feed_forward.w1"},
        {std::regex(R"(model\.(layers\.\d+)\.mlp\.down_proj)"), "$1.feed_forward.w2"},
        {std::regex(R"(model\.(layers\.\d+)\.mlp\.up_proj)"), "$1.feed_forward.w3"},
        {std::regex(R"(model\.(layers\.\d+)\.self_attn\.q_proj)"), "$1.attention.wq"},
        {std::regex(R"(model\.(layers\.\d+)\.self_attn\.k_proj)"), "$1.attention.wk"},
        {std::regex(R"(model\.(layers\.\d+)\.self_attn\.v_proj)"), "$1.attention.wv"},
        {std::regex(R"(model\.(layers\.\d+)\.self_attn\.o_proj)"), "$1.attention.wo"},
        {std::regex(R"(model.norm)"), "norm"},
        {std::regex(R"(model.embed_tokens)"), "tok_embeddings"},
    };

    for (auto it = document.begin(); it != document.end(); ++it) {
        auto st = *it;
        auto name = st.name();

        for (const auto& [re, replacement] : mapping) {
            name = std::regex_replace(name, re, replacement);
        }

        auto sizes = st.sizes();
        auto shape = std::vector<std::size_t>(sizes.begin(), sizes.end());

        doc.insert(safetensor(name, st.dtype(), shape, st.container_ptr()));
    }

    doc.insert("output.weight", "tok_embeddings.weight");
    return doc;
}


} // namespace huggingface
} // namespace metalchat
