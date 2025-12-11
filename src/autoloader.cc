#include <metalchat/autoloader.h>


namespace metalchat {


void
llama3_reference_traits::document_adaptor::adapt(safetensor_document& document) const
{
    document.insert("output.weight", "tok_embeddings.weight");
}


void
llama3_huggingface_traits::document_adaptor::adapt(safetensor_document& document) const
{}


} // namespace metalchat
