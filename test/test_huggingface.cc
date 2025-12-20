// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>

#include <metalchat/huggingface.h>

#include "metalchat/testing.h"

using namespace metalchat;


TEST_CASE("Test llama3 huggingface model adaptor", "[huggingface]")
{
    hardware_accelerator gpu0;
    auto repo_path = test_fixture_path() / "meta-llama/Llama-3.2-1B-Instruct";
    auto document_path = repo_path / "model.safetensors";
    auto document_adaptor = huggingface::llama3_document_adaptor();
    auto document = safetensor_document::open(document_path, gpu0);

    document = document_adaptor.adapt(document);
    for (auto it = document.begin(); it != document.end(); ++it) {
        auto st = *it;
        REQUIRE(!st.name().starts_with("model"));
    }

    REQUIRE(std::distance(document.begin(), document.end()) == 147);
}
