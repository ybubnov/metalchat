// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>

#include <metalchat/nn/transformer.h>
#include <metalchat/quantization.h>


using namespace metalchat;


TEST_CASE("Test replace QLora linear", "[quantization]")
{
    hardware_accelerator gpu0;
    nn::feed_forward<float> input_layer(gpu0);

    auto params_before = input_layer.get_parameters();
    REQUIRE(params_before.size() == 3);

    using BasicLinear = nn::basic_linear<float>;
    using QLoraLinear = quantization::qlora_linear<float>;

    quantization::replace<BasicLinear>(input_layer, QLoraLinear(1.0, 32, gpu0));

    auto params_after = input_layer.get_parameters();
    REQUIRE(params_after.size() == 12);
}


TEST_CASE("Test QLora adaptor", "[quantization]")
{
    hardware_accelerator gpu0;
    quantization::qlora_adaptor<float> adaptor(gpu0);

    adaptor.set_parameter("A.weight", rand<float>({16, 2048}, gpu0));
    adaptor.set_parameter("B.weight", rand<float>({512, 16}, gpu0));

    auto input = rand<float>({1, 19, 2048}, gpu0);
    auto output = adaptor(input).get();

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 19);
    REQUIRE(output.size(2) == 512);
}
