// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sstream>

#include <metalchat/reference.h>


using namespace metalchat;
using namespace Catch::Matchers;


TEST_CASE("Test llama3 options loader", "[reference]")
{
    const std::string options_json = R"({
      "dim": 2048,
      "n_layers": 16,
      "n_heads": 32,
      "n_kv_heads": 8,
      "vocab_size": 128256,
      "ffn_dim_multiplier": 1.5,
      "multiple_of": 256,
      "norm_eps": 1e-05,
      "rope_theta": 500000.0,
      "use_scaled_rope": true
    })";

    std::stringstream input(options_json);

    reference::llama3_options_loader loader;
    auto options = loader.load(input);

    REQUIRE(options.head_dim() == 64);
    REQUIRE(options.n_layers() == 16);
    REQUIRE(options.n_heads() == 32);
    REQUIRE(options.n_kv_heads() == 8);
    REQUIRE(options.max_seq_len() == 1024);

    REQUIRE_THAT(options.rope_theta(), WithinRel(500000.0, 0.01));
    REQUIRE_THAT(options.norm_eps(), WithinRel(1e-5, 0.01));
}
