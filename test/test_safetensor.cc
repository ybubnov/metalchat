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

    auto alloc0 = gpu0.get_allocator();
    auto alloc1 = make_rebind_allocator<bf16>(alloc0);

    auto tensors = safetensor_document::load("../llama32.safetensors", alloc1);

    nn::llama3<bf16> m(nn::default_llama3_1b_options(), gpu0);
    m.initialize(tensors);

    auto params = m.get_parameters();

    REQUIRE(params.size() == 179);
    for (auto [name, param] : params) {
        REQUIRE(param->numel() > 0);
    }
}
