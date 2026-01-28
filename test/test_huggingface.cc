// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <metalchat/huggingface.h>

#include "metalchat/testing.h"

using namespace metalchat;
using namespace Catch::Matchers;


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


TEST_CASE("Test llama3 options serializer", "[huggingface]")
{
    // Some parameter are removed from the HuggingFace's configuration for compactness.
    const std::string options_json = R"({
      "attention_bias": false,
      "attention_dropout": 0.0,
      "head_dim": 64,
      "hidden_act": "silu",
      "hidden_size": 2048,
      "initializer_range": 0.02,
      "intermediate_size": 8192,
      "max_position_embeddings": 131072,
      "mlp_bias": false,
      "model_type": "llama",
      "num_attention_heads": 32,
      "num_hidden_layers": 16,
      "num_key_value_heads": 8,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-05,
      "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
      },
      "rope_theta": 500000.0,
      "use_cache": true,
      "vocab_size": 128256
    })";

    std::stringstream input(options_json);

    huggingface::llama3_options_serializer serializer;
    auto options = serializer.load(input);

    REQUIRE(options.head_dim() == 64);
    REQUIRE(options.n_layers() == 16);
    REQUIRE(options.n_heads() == 32);
    REQUIRE(options.n_kv_heads() == 8);
    REQUIRE(options.max_seq_len() == 1024);

    REQUIRE_THAT(options.rope_theta(), WithinRel(500000.0, 0.01));
    REQUIRE_THAT(options.norm_eps(), WithinRel(1e-5, 0.01));
}


TEST_CASE("Test llama3 tokenizer loader", "[huggingface]")
{
    const std::string tokenizer_json = R"({
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
          {
            "type": "Split",
            "pattern": {
              "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
            },
            "behavior": "Isolated",
            "invert": false
          },
          {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
        ]
      },
      "model": {
        "type": "BPE",
        "dropout": null,
        "unk_token": null,
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "byte_fallback": false,
        "ignore_merges": true,
        "vocab": {"!": 0, "\"": 1, "#": 2, "$": 3, "%": 4},
        "merges": []
      }
    })";

    std::stringstream input(tokenizer_json);

    huggingface::llama3_tokenizer_loader loader;
    auto tokenizer = loader.load(input);

    REQUIRE(tokenizer.size() == 16);
    REQUIRE(tokenizer.decode(4) == "%");
    REQUIRE(tokenizer.encode("#")[0] == 2);
}
