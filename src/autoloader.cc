#include <metalchat/autoloader.h>


namespace metalchat {


void
llama3_reference_traits::document_adaptor::adapt(safetensor_document& document) const
{
    document.insert("output.weight", "tok_embeddings.weight");
}


std::filesystem::path
llama3_reference_traits::document_resolver::resolve(const std::filesystem::path& path) const
{
    return path / "model.safetensors";
}


void
llama3_huggingface_traits::document_adaptor::adapt(safetensor_document& document) const
{}


} // namespace metalchat
