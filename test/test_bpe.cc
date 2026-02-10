// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <metalchat/reference.h>
#include <metalchat/repository.h>
#include <metalchat/text/gpt.h>

#include "metalchat/testing.h"


using namespace metalchat;


auto
make_tokenizer()
{
    auto repo_path = test_fixture_path() / "meta-llama/Llama-3.2-1B-Instruct/original";
    auto repository = filesystem_repository<reference::llama3>(repo_path);
    return repository.retrieve_tokenizer("tokenizer.model");
}


TEST_CASE("Test GPT-2 codec", "[gpt2]")
{
    text::gpt2_codec codec;
    auto output = codec.encode("    Hello  \x80");
    auto input = codec.decode(output);
    REQUIRE(output == "ĠĠĠĠHelloĠĠĢ");
    REQUIRE(input == "    Hello  \x80");
}


TEST_CASE("TEST GPT-2 to Reference", "[gpt2][integration]")
{
    auto tokenizer = make_tokenizer();
    text::gpt2_codec codec;

    auto str = tokenizer.decode(125579);
    REQUIRE(str == " استاندارد");

    auto output = codec.encode(str);
    REQUIRE(output == "ĠØ§Ø³ØªØ§ÙĨØ¯Ø§Ø±Ø¯");
    auto input = codec.decode(output);
    REQUIRE(input == " استاندارد");
}


TEST_CASE("Test BPE encode and decode", "[bpe][integration]")
{
    auto tokenizer = make_tokenizer();

    auto ids = tokenizer.encode("This is a test sentence.");
    REQUIRE(ids.size(0) == 6);

    std::vector<int32_t> actual(ids.begin(), ids.end());
    std::vector<int32_t> expect = {2028, 374, 264, 1296, 11914, 13};
    REQUIRE_THAT(actual, Catch::Matchers::Equals(expect));

    auto str = tokenizer.decode(ids.data_ptr(), ids.data_ptr() + ids.size(0));
    REQUIRE(str == "This is a test sentence.");
}


TEST_CASE("Encode pairs with byte merge", "[bpe][integration]")
{
    auto tokenizer = make_tokenizer();
    auto ids = tokenizer.encode("And his name is John Cena.");

    REQUIRE(ids.size(0) == 7);
    std::vector<int32_t> actual(ids.begin(), ids.end());
    std::vector<int32_t> expect = {3112, 813, 836, 374, 3842, 89663, 13};
    REQUIRE_THAT(actual, Catch::Matchers::Equals(expect));
}


TEST_CASE("Encode ipython word", "[bpe][integration]")
{
    auto tokenizer = make_tokenizer();
    auto ids = tokenizer.encode(" ipython");

    auto str = tokenizer.decode(ids.data_ptr(), ids.data_ptr() + ids.size(0));
    REQUIRE(str == " ipython");
}


TEST_CASE("Encode unknown words", "[bpe][integration]")
{
    auto tokenizer = make_tokenizer();
    auto ids = tokenizer.encode("This is debatable topic.");

    REQUIRE(ids.size(0) > 0);
}


TEST_CASE("Decode control token", "[bpe][integration]")
{
    auto tokenizer = make_tokenizer();
    auto token = tokenizer.decode(128001);

    REQUIRE(token == "<|end_of_text|>");
}
