// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/nn.h>


using namespace metalchat;


TEST_CASE("Test layer parameters", "[layer]")
{
    using Linear = nn::linear<float, random_memory_container<float>>;

    hardware_accelerator gpu0(16);
    nn::indirect_layer<Linear> linear(gpu0);

    linear.set_parameter("weight", full<float>({3, 5}, 4.0));

    auto weight = linear.get_parameter("weight");
    REQUIRE(weight->dimensions() == 2);
    REQUIRE(weight->size(0) == 3);
    REQUIRE(weight->size(1) == 5);

    auto output = linear(shared_tensor(full<float>({10, 5}, 2.0))).get();
    REQUIRE(output.dim() == 2);
    REQUIRE(output.size(0) == 10);
    REQUIRE(output.size(1) == 3);

    auto params = linear.get_parameters();
    REQUIRE(params.size() == 1);
}


TEST_CASE("Test recurse parameter query", "[layer]")
{
    using Linear = nn::linear<float, random_memory_container<float>>;

    struct test_layer : public nn::basic_layer {
        nn::indirect_layer<Linear> linear1;
        nn::indirect_layer<Linear> linear2;

        test_layer(hardware_accelerator gpu)
        : nn::basic_layer(gpu)
        {
            linear1 = register_layer<Linear>("layer1", full<float>({3, 4}, 3.0));
            linear2 = register_layer<Linear>("layer2", full<float>({4, 5}, 4.0));
        }
    };

    struct test_layer_outer : public nn::basic_layer {
        nn::indirect_layer<test_layer> inner;
        nn::indirect_layer<Linear> linear0;

        test_layer_outer(hardware_accelerator gpu)
        : nn::basic_layer(gpu)
        {
            inner = register_layer<test_layer>("inner");
            linear0 = register_layer<Linear>("linear0", full<float>({1, 2}, 5.0));
        }
    };

    hardware_accelerator gpu0;
    nn::indirect_layer<test_layer_outer> tl(gpu0);

    auto param = tl.get_parameter("inner.layer1.weight");
    REQUIRE(param->dimensions() == 2);
    REQUIRE(param->size(0) == 3);
    REQUIRE(param->size(1) == 4);

    param = tl.get_parameter("linear0.weight");
    REQUIRE(param->dimensions() == 2);
    REQUIRE(param->size(0) == 1);
    REQUIRE(param->size(1) == 2);

    auto match_not_registered = Catch::Matchers::ContainsSubstring("is not registered");

    REQUIRE_THROWS_WITH(tl.get_parameter("inner.linear3.weight"), match_not_registered);
    REQUIRE_THROWS_WITH(tl.get_parameter("inner.linear1"), match_not_registered);
    REQUIRE_THROWS_WITH(tl.get_parameter("."), match_not_registered);
    REQUIRE_THROWS_WITH(tl.get_parameter("inner....."), match_not_registered);
    REQUIRE_THROWS_WITH(tl.get_parameter(""), match_not_registered);
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
