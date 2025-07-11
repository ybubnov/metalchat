#include <catch2/catch_test_macros.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
#include <metalchat/nn.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Test model load", "[safetensor]")
{
    metalchat::hardware_accelerator gpu0(16);

    auto options = nn::attention_options{
        .head_dim = 64,
        .n_heads = 32,
        .n_kv_heads = 8,
        .max_seq_len = 1024,
        .rope_theta = 500000.0
    };

    auto alloc0 = gpu0.get_allocator();
    auto alloc1 = make_rebind_allocator<bf16>(alloc0);

    auto tensors = safetensor_document::load("../llama32.safetensors", alloc1);

    nn::llama<bf16> m(16, options, gpu0);
    m.initialize(tensors);

    auto params = m.get_parameters();

    REQUIRE(params.size() == 147);
    for (auto [name, param] : params) {
        REQUIRE(param.numel() > 0);
    }
}
