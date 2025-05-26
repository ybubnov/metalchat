#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>


#include <metalchat/bpe.h>


using namespace metalchat;


TEST_CASE("Test BPE encode and decode", "[bpe]")
{
    byte_pair_encoder tokenizer("../Llama-3.2-1B/original/tokenizer.model");

    auto ids = tokenizer.encode("This is a test sentence.");
    REQUIRE(ids.size(0) == 6);

    std::vector<int32_t> actual(ids.begin(), ids.end());
    std::vector<int32_t> expect = {2028, 374, 264, 1296, 11914, 13};
    REQUIRE_THAT(actual, Catch::Matchers::Equals(expect));

    auto str = tokenizer.decode(ids.data_ptr(), ids.data_ptr() + ids.size(0));
    REQUIRE(str == "This is a test sentence.");

    std::vector<int32_t> tokens;
    tokenizer.encode(special_token::begin_text, tokens);
    REQUIRE(tokens.size() == 1);
    REQUIRE(tokens[0] == 128000);
}


TEST_CASE("Encode pairs with byte merge", "[bpe]")
{
    byte_pair_encoder tokenizer("../Llama-3.2-1B/original/tokenizer.model");

    auto ids = tokenizer.encode("And his name is John Cena.");

    REQUIRE(ids.size(0) == 7);
    std::vector<int32_t> actual(ids.begin(), ids.end());
    std::vector<int32_t> expect = {3112, 813, 836, 374, 3842, 89663, 13};
    REQUIRE_THAT(actual, Catch::Matchers::Equals(expect));
}


TEST_CASE("Encode unknown words", "[bpe]")
{
    byte_pair_encoder tokenizer("../Llama-3.2-1B/original/tokenizer.model");

    auto ids = tokenizer.encode("This is debatable topic.");
    REQUIRE(ids.size(0) > 0);
}


TEST_CASE("Decode special token", "bpe")
{
    byte_pair_encoder tokenizer("../Llama-3.2-1B/original/tokenizer.model");

    auto token = tokenizer.decode(128001);
    REQUIRE(token == "<|end_of_text|>");
}
