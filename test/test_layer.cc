// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/nn.h>


using namespace metalchat;


TEST_CASE("Test layer copy assignment", "[layer]")
{
    using Linear = nn::linear<float>;

    hardware_accelerator gpu0;
    nn::indirect_layer<Linear> linear0(gpu0);
    auto linear1 = linear0;

    auto& weight = linear1.parameter("weight");
    REQUIRE(weight.dimensions() == 2);
    REQUIRE(weight.size(0) == 0);
    REQUIRE(weight.size(1) == 0);
}


TEST_CASE("Test linear layer bias", "[layer]")
{
    using Linear = nn::linear<float>;

    hardware_accelerator gpu0;
    nn::indirect_layer<Linear> linear(3, 4, /*bias=*/true, gpu0);

    auto& weight = linear.parameter("weight");
    REQUIRE(weight.dimensions() == 2);
    REQUIRE(weight.size(0) == 4);
    REQUIRE(weight.size(1) == 3);

    auto& bias = linear.parameter("bias");
    REQUIRE(bias.dimensions() == 1);
    REQUIRE(bias.size(0) == 4);

    auto output = linear(rand<float>({8, 3})).get();
    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == 8);
    REQUIRE(output.size(1) == 4);
}


TEST_CASE("Test layer parameters", "[layer]")
{
    using Linear = nn::linear<float, random_memory_container<float>>;

    hardware_accelerator gpu0(16);
    nn::indirect_layer<Linear> linear(gpu0);

    linear.set_parameter("weight", full<float>({3, 5}, 4.0));

    auto& weight = linear.parameter("weight");
    REQUIRE(weight.dimensions() == 2);
    REQUIRE(weight.size(0) == 3);
    REQUIRE(weight.size(1) == 5);

    auto output = linear(shared_tensor(full<float>({10, 5}, 2.0))).get();
    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == 10);
    REQUIRE(output.size(1) == 3);

    auto params = linear.parameters();
    REQUIRE(params.size() == 1);
}


TEST_CASE("Test recurse parameter query", "[layer]")
{
    using Linear = nn::linear<float, hardware_memory_container<float>>;

    struct test_layer : public nn::basic_layer {
        nn::indirect_layer<Linear> linear1;
        nn::indirect_layer<Linear> linear2;

        test_layer(hardware_accelerator gpu)
        : nn::basic_layer(gpu)
        {
            linear1 = register_layer<Linear>("layer1", 3, 4);
            linear2 = register_layer<Linear>("layer2", 4, 5);
        }
    };

    struct test_layer_outer : public nn::basic_layer {
        nn::indirect_layer<test_layer> inner;
        nn::indirect_layer<Linear> linear0;

        test_layer_outer(hardware_accelerator gpu)
        : nn::basic_layer(gpu)
        {
            inner = register_layer<test_layer>("inner");
            linear0 = register_layer<Linear>("linear0", 1, 2);
        }
    };

    hardware_accelerator gpu0;
    nn::indirect_layer<test_layer_outer> tl(gpu0);

    auto& param1 = tl.parameter("inner.layer1.weight");
    REQUIRE(param1.dimensions() == 2);
    REQUIRE(param1.size(0) == 4);
    REQUIRE(param1.size(1) == 3);

    auto& param0 = tl.parameter("linear0.weight");
    REQUIRE(param0.dimensions() == 2);
    REQUIRE(param0.size(0) == 2);
    REQUIRE(param0.size(1) == 1);

    auto match_not_registered = Catch::Matchers::ContainsSubstring("is not registered");

    REQUIRE_THROWS_WITH(tl.parameter("inner.linear3.weight"), match_not_registered);
    REQUIRE_THROWS_WITH(tl.parameter("inner.linear1"), match_not_registered);
    REQUIRE_THROWS_WITH(tl.parameter("."), match_not_registered);
    REQUIRE_THROWS_WITH(tl.parameter("inner....."), match_not_registered);
    REQUIRE_THROWS_WITH(tl.parameter(""), match_not_registered);
}


TEST_CASE("Test layers traversal", "[layer]")
{
    using linear = nn::linear<float, hardware_memory_container<float>>;

    struct test_layer : public nn::basic_layer {
        using LinearArray = nn::layer_array<linear>;
        nn::indirect_layer<LinearArray> layers;

        test_layer(std::size_t size, hardware_accelerator gpu)
        : nn::basic_layer(gpu)
        {
            layers = register_layer<LinearArray>("layers");

            for (std::size_t i = 0; i < size; i++) {
                layers->emplace_back(10, 3, gpu);
            }
        }
    };

    hardware_accelerator gpu0;
    nn::indirect_layer<test_layer> layer(10, gpu0);

    using layer_ptr = test_layer::layer_pointer;
    std::vector<layer_ptr> layers;

    layer.apply([&](nn::named_layer layer) { layers.push_back(layer.ptr); });

    REQUIRE(layers.size() == 11);
}
