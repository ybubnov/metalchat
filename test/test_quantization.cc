// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>

#include <metalchat/nn/transformer.h>
#include <metalchat/quantization.h>


using namespace metalchat;


TEST_CASE("Test replace QLora linear", "[layer]")
{
    hardware_accelerator gpu0;
    nn::feed_forward<float> input_layer(gpu0);

    auto params_before = input_layer.get_parameters();
    REQUIRE(params_before.size() == 3);

    using BasicLinear = nn::basic_linear<float>;
    using QLoraLinear = quantization::qlora_linear<float>;

    quantization::replace<BasicLinear>(input_layer, QLoraLinear(1.0, gpu0));

    auto params_after = input_layer.get_parameters();
    REQUIRE(params_after.size() == 12);
}
