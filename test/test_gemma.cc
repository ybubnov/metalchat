// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <codecvt>
#include <locale>

#include <catch2/catch_test_macros.hpp>

#include <metalchat/huggingface/gemma.h>
#include <metalchat/nn/gemma.h>
#include <metalchat/repository.h>

#include "metalchat/testing.h"

using namespace metalchat;


TEST_CASE("Test gemma3", "[gemma][integration]")
{
    auto repo_path = test_fixture_path() / "google/gemma-3-270m-it";

    hardware_accelerator gpu0;
    filesystem_repository<huggingface::gemma3> repository(repo_path, gpu0);

    auto options = repository.retrieve_options("config.json");
    auto tokenizer = repository.retrieve_tokenizer("tokenizer.json");

    std::vector<int32_t> ids;
    tokenizer.encode(text::token::begin_text, std::back_inserter(ids));
    tokenizer.encode(UR"(I have a dog called)", std::back_inserter(ids));

    auto transformer = repository.retrieve_transformer("model.safetensors", options);

    auto heap_size = std::size_t(512) * 1024 * 1024;
    auto alloc0 = hardware_heap_allocator<void>(gpu0.get_metal_device(), heap_size);
    auto alloc1 = nocopy_allocator(alloc0, gpu0.get_metal_device());
    gpu0.set_allocator(std::move(alloc1));

    auto input0 = shared_tensor(to_tensor<int32_t>({1, ids.size()}, ids.begin(), ids.end()));
    auto id = transformer.transform(input0);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    using convert_type = std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>;
    convert_type convert;
#pragma clang diagnostic pop

    std::cout << "I have a dog called";
    std::cout << convert.to_bytes(tokenizer.decode(id.get()[0, 0]));

    for (std::size_t i = input0.size(1); i < 32; i++) {
        id = transformer.transform(id, i);
        std::cout << convert.to_bytes(tokenizer.decode(id.get()[0, 0])) << std::flush;
    }
}
