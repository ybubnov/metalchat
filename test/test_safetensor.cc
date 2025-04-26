#include <catch2/catch_test_macros.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/llama.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Test model load", "[safetensor]")
{
    metalchat::hardware_accelerator gpu0("metalchat.metallib", 16);
    metalchat::safetensor_file tensors("../llama32.safetensors");

    auto options = llama::attention_options{
        .head_dim = 64,
        .n_heads = 32,
        .n_kv_heads = 8,
        .max_seq_len = 1024,
        .rope_theta = 500000.0
    };

    auto alloc0 = gpu0.get_allocator();
    auto alloc1 = make_rebind_allocator<bf16>(alloc0);

    llama::model<bf16> m(16, options, gpu0);
    m.initialize(tensors, alloc1);

    auto params = m.get_parameters();

    REQUIRE(params.size() == 179);
    for (auto [name, param] : params) {
        REQUIRE(param.numel() > 0);
    }
}
