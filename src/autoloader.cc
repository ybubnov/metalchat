#include <metalchat/autoloader.h>


namespace metalchat {


void
llama3_traits::reference_document_adaptor::adapt(safetensor_document& document) const
{
    document.insert("output.weight", "tok_embeddings.weight");
}


} // namespace metalchat
