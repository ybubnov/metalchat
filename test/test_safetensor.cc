#include <catch2/catch_test_macros.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/bpe.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/llama.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Test model load", "[safetensor]")
{
    metalchat::bpe bpe("../Llama-3.2-1B/original/tokenizer.model");
    metalchat::hardware_accelerator gpu0("metalchat.metallib", 16);
    // metalchat::safetensor_file tensors("../llama32.safetensors");

    // auto m = llama::make_model<bf16>(tensors, gpu0);
    auto options = llama::attention_options{
        .head_dim = 64,
        .n_heads = 32,
        .n_kv_heads = 8,
        .max_seq_len = 1024,
        .rope_theta = 500000.0
    };

    llama::model<bf16> m(16, options, gpu0);
    auto params = m.get_parameters();

    for (auto [name, param] : params) {
        std::cout << name << ": (" << param.sizes() << ")" << std::endl;
    }
}
