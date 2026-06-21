// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/functional.h>
#include <metalchat/huggingface/llama.h>
#include <metalchat/repository.h>
#include <metalchat/tensor.h>
#include <metalchat/text.h>

#include "metalchat/testing.h"

using namespace metalchat;


TEST_CASE("Test Llama3 implementation", "[llama][integration]")
{
    auto repo_path = test_fixture_path() / "meta-llama/Llama-3.2-1B-Instruct";

    metalchat::hardware_accelerator gpu0;
    filesystem_repository<huggingface::llama3> repository(repo_path, gpu0);

    auto options = nn::default_llama3_1b_options();
    options.max_seq_len = 16;

    auto transformer = repository.retrieve_transformer("model.safetensors", options);
    auto tokenizer = repository.retrieve_tokenizer("tokenizer.json");

    auto heap_size = std::size_t(512) * 1024 * 1024;
    auto alloc0 = hardware_heap_allocator<void>(gpu0.get_metal_device(), heap_size);
    auto alloc1 = nocopy_allocator(alloc0, gpu0.get_metal_device());
    gpu0.set_allocator(std::move(alloc1));

    auto input_text = std::string("I have a dog called");

    std::vector<int32_t> ids;
    auto output = std::back_inserter(ids);
    tokenizer.encode(text::token::begin_text, output);
    tokenizer.encode(input_text, output);


    auto input0 = shared_tensor(to_tensor<int32_t>({1, ids.size()}, ids.begin(), ids.end()));
    auto id = transformer.transform(input0);

    std::cout << input_text;
    std::cout << tokenizer.decode(id.get()[0, 0]);

    for (std::size_t i = input0.size(1); i < 64; i++) {
        id = transformer.transform(id, i);
        std::cout << tokenizer.decode(id.get()[0, 0]) << std::flush;
    }
}
