// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/autoloader.h>
#include <metalchat/functional.h>
#include <metalchat/kernel/sort.h>
#include <metalchat/nn/llama.h>
#include <metalchat/tensor.h>
#include <metalchat/text.h>

#include "metalchat/testing.h"

using namespace metalchat;


TEST_CASE("Test make model", "[llama]")
{
    auto bpe_path = test_fixture_path() / "llama3.2:1b-instruct" / "original" / "tokenizer.model";
    auto model_path = test_fixture_path() / "llama3.2:1b-instruct" / "model.safetensors";

    metalchat::text::byte_pair_encoder bpe(bpe_path);
    metalchat::hardware_accelerator gpu0(1);

    using LLama3 = nn::llama3<bf16>;
    auto options = nn::default_llama3_1b_options().max_seq_len(16);
    nn::indirect_layer<LLama3> m(options, gpu0);

    auto document = safetensor_document::open(model_path, gpu0);
    auto document_adaptor = llama3_reference_traits<bf16>::document_adaptor();
    document_adaptor.adapt(document);
    document.load(m);

    auto heap_size = std::size_t(512) * 1024 * 1024;
    auto alloc0 = hardware_heap_allocator<void>(gpu0.get_metal_device(), heap_size);
    auto alloc1 = nocopy_allocator(alloc0, gpu0.get_metal_device());
    gpu0.set_allocator(std::move(alloc1));

    auto input_text = std::string("I have a dog called");

    std::vector<int32_t> ids;
    bpe.encode(text::special_token::begin_text, std::back_inserter(ids));
    bpe.encode(input_text, std::back_inserter(ids));

    auto input0 = shared_tensor(to_tensor<int32_t>({1, ids.size()}, ids.begin(), ids.end()));
    auto logit0 = m(input0, 0);
    auto id = top_p(logit0.flatten<2>(), bf16(0.6f), bf16(0.9), gpu0);

    std::cout << input_text;
    std::cout << bpe.decode(id.get()[0, 0]);
    std::vector<future_tensor<int32_t, 2>> outputs;

    for (std::size_t i = input0.size(1); i < 64; i++) {
        auto logits = m(id, i).flatten<2>();
        id = top_p(logits, bf16(0.6f), bf16(0.9f), gpu0);

        // outputs.push_back(id);
        std::cout << bpe.decode(id.get()[0, 0]) << std::flush;
    }

    // for (auto& o : outputs) {
    //     std::cout << bpe.decode(o.get()[0, 0]) << std::flush;
    // }

    std::cout << std::endl;
}
