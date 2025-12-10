// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>

#include <metalchat/nn/llama.h>
#include <metalchat/nn/transformer.h>
#include <metalchat/quantization.h>
#include <metalchat/text/bpe.h>

#include "metalchat/testing.h"


using namespace metalchat;


TEST_CASE("Test replace QLoRa linear", "[quantization]")
{
    using FeedForward = nn::feed_forward<float>;
    using BasicLinear = nn::basic_linear<float>;
    using QLoraLinear = quantization::qlora_linear<float>;

    hardware_accelerator gpu0;

    nn::indirect_layer<FeedForward> input_layer(gpu0);

    auto params_before = input_layer.get_parameters();
    REQUIRE(params_before.size() == 3);

    quantization::replace<BasicLinear>(input_layer, [&] {
        return nn::indirect_layer<QLoraLinear>(1.0, 32, gpu0);
    });

    auto params_after = input_layer.get_parameters();
    REQUIRE(params_after.size() == 12);
}


TEST_CASE("Test QLoRa adaptor", "[quantization]")
{
    using QLoraAdaptor = quantization::qlora_adaptor<float>;

    hardware_accelerator gpu0;
    nn::indirect_layer<QLoraAdaptor> adaptor(gpu0);

    adaptor.set_parameter("A.weight", rand<float>({16, 2048}, gpu0));
    adaptor.set_parameter("B.weight", rand<float>({512, 16}, gpu0));

    auto input = rand<float>({1, 19, 2048}, gpu0);
    auto output = adaptor(input).get();

    REQUIRE(output.dim() == 3);
    REQUIRE(output.size(0) == 1);
    REQUIRE(output.size(1) == 19);
    REQUIRE(output.size(2) == 512);
}


TEST_CASE("Test QLoRA inference", "[quantization]")
{
    using LLama3 = nn::llama3<bf16>;

    hardware_accelerator gpu0(1);
    nn::indirect_layer<LLama3> model(nn::default_llama3_1b_options(), gpu0);

    using BasicLinear = nn::basic_linear<bf16>;
    using BasicEmbedding = nn::basic_embedding<bf16>;
    using QLinear = quantization::linear<bf16>;
    using QLoraLinear = quantization::qlora_linear<bf16>;
    using QLoraEmbedding = quantization::qlora_embedding<bf16>;

    quantization::replace<BasicLinear>(model, [&] {
        return nn::indirect_layer<QLoraLinear>(2.0, 32, gpu0);
    });
    quantization::replace<BasicEmbedding>(model, [&] {
        return nn::indirect_layer<QLoraEmbedding>(gpu0);
    });

    auto replace = [&](nn::named_layer layer) {
        auto layer_ptr = dynamic_pointer_cast<BasicLinear>(layer.ptr);
        if (layer.name == "output" && layer_ptr != nullptr) {
            auto& layer_parent = model.get_parent_layer(layer.path);
            layer_parent.register_layer<QLinear>(layer.name);
        }
    };
    model.apply(replace);

    auto bpe_path = test_fixture_path() / "llama3.2:1b-qlora" / "tokenizer.model";
    auto model_path = test_fixture_path() / "llama3.2:1b-qlora" / "model.safetensors";

    metalchat::text::bpe bpe(bpe_path);
    safetensor_document::load(model_path, model);

    auto heap_size = std::size_t(1024) * 1024 * 1024;
    auto alloc0 = hardware_heap_allocator<void>(gpu0.get_metal_device(), heap_size);
    auto alloc1 = nocopy_allocator(alloc0, gpu0.get_metal_device());
    gpu0.set_allocator(std::move(alloc1));

    auto input_text = std::string("I have a dog called");

    std::vector<int32_t> ids;
    bpe.encode(text::special_token::begin_text, std::back_inserter(ids));
    bpe.encode(input_text, std::back_inserter(ids));

    auto input0 = shared_tensor(to_tensor<int32_t>({1, ids.size()}, ids.begin(), ids.end()));
    auto logit0 = model(input0, 0);
    auto id = top_p(logit0.flatten<2>(), bf16(0.6f), bf16(0.9), gpu0);

    std::cout << input_text;
    std::cout << bpe.decode(id.get()[0, 0]);

    // for (std::size_t i = input0.size(1); i < 16; i++) {
    //     auto logits = model(id, i).flatten<2>();
    //     id = top_p(logits, bf16(0.6f), bf16(0.9f), gpu0);

    //    std::cout << bpe.decode(id.get()[0, 0]) << std::flush;
    //}
}
