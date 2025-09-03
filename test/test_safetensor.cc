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

    nn::llama3<bf16> m(nn::default_llama3_1b_options(), gpu0);
    safetensor_document::load("../llama32.safetensors", m, alloc1);

    auto params = m.get_parameters();

    REQUIRE(params.size() == 179);
    for (auto [name, param] : params) {
        REQUIRE(param->numel() > 0);
    }
}


TEST_CASE("Test read/write/read", "[safetensor]")
{
    metalchat::hardware_accelerator gpu0(16);

    auto alloc0 = gpu0.get_allocator();
    auto alloc1 = make_rebind_allocator<bf16>(alloc0);

    nn::llama3<bf16> m(nn::default_llama3_1b_options(), gpu0);
    safetensor_document::load("../llama32.safetensors", m, alloc1);

    safetensor_document::save("../llama32-copy.safetensors", m);
    std::cout << "saved" << std::endl;

    nn::llama3<bf16> m1(nn::default_llama3_1b_options(), gpu0);
    safetensor_document::load("../llama32-copy.safetensors", m1, alloc1);

    auto params = m1.get_parameters();

    REQUIRE(params.size() == 179);
    for (auto [name, param] : params) {
        REQUIRE(param->numel() > 0);
    }
}
