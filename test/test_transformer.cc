// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <map>

#include <catch2/catch_test_macros.hpp>

#include <metalchat/huggingface.h>
#include <metalchat/transformer.h>


using namespace metalchat;


TEST_CASE("Test transformer options merging", "[transformer]")
{
    using Transformer = huggingface::llama3;
    using TransformerTraits = transformer_traits<Transformer>;

    using options_key = std::string;
    using options_value = std::variant<bool, int, float, std::string>;
    auto options = std::map<options_key, options_value>(
        {{"rope_theta", 40000.0f}, {"some.unknown.field", true}}
    );

    auto options_in = nn::llama3_options().rope_theta(20000.0);
    auto options_out = TransformerTraits::merge_options(options.begin(), options.end(), options_in);

    REQUIRE(options_out.rope_theta() == 40000.0);
}
