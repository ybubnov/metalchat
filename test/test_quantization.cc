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


TEST_CASE("Test QLoRa adaptor", "[quantization]")
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


TEST_CASE("Test QLoRA inference", "[quantization]")
{
    hardware_accelerator gpu0;
    nn::llama3<bf16> model(nn::default_llama3_1b_options(), gpu0);

    using BasicLinear = nn::basic_linear<bf16>;
    using BasicEmbedding = nn::basic_embedding<bf16>;
    using QLoraLinear = quantization::qlora_linear<bf16>;
    using QLoraEmbedding = quantization::qlora_embedding<bf16>;

    quantization::replace<BasicLinear>(model, QLoraLinear(2.0, 32, gpu0));
    std::cout << "1111111111111111" << std::endl;
    quantization::replace<BasicEmbedding>(model, QLoraEmbedding(gpu0));
    std::cout << "2222222222222222" << std::endl;

    auto replace = [&](nn::named_layer layer) {
        auto layer_ptr = dynamic_pointer_cast<BasicLinear>(layer.ptr);
        if (layer.name == "output" && layer_ptr != nullptr) {
            *layer.ptr = quantization::linear<bf16>(gpu0);
        }
    };
    model.apply(replace);

    auto bpe_path = test_fixture_path() / "llama3.2:1b-qlora" / "tokenizer.model";
    auto model_path = test_fixture_path() / "llama3.2:1b-qlora" / "model.safetensors";

    metalchat::text::bpe bpe(bpe_path);
    safetensor_document::load(model_path, model);

    auto alloc = gpu0.get_allocator();
    gpu0.set_allocator(nocopy_allocator(alloc, gpu0.get_metal_device()));

    auto input_text = std::string("I have a dog called");

    std::vector<int32_t> ids;
    bpe.encode(text::special_token::begin_text, std::back_inserter(ids));
    bpe.encode(input_text, std::back_inserter(ids));

    auto input0 = shared_tensor(to_tensor<int32_t>({1, ids.size()}, ids.begin(), ids.end()));
    auto logit0 = model(input0, 0);
    std::cout << "@@@@@@@@@@@@@@@@@@@@" << std::endl;
    auto id = top_p(logit0.flatten<2>(), bf16(0.6f), bf16(0.9), gpu0);

    std::cout << input_text;
    std::cout << bpe.decode(id.get()[0, 0]);
    std::vector<future_tensor<int32_t, 2>> outputs;

    for (std::size_t i = input0.size(1); i < 64; i++) {
        auto logits = model(id, i).flatten<2>();
        id = top_p(logits, bf16(0.6f), bf16(0.9f), gpu0);

        std::cout << bpe.decode(id.get()[0, 0]) << std::flush;
    }
}
