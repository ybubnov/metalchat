#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>


#include <metalchat/bpe.h>


using namespace metalchat;


TEST_CASE("Test BPE encode and decode", "[bpe]")
{
    bpe tokenizer("../Llama-3.2-1B/original/tokenizer.model");

    auto ids = tokenizer.encode("This is a test sentence.");
    REQUIRE(ids.size(0) == 6);

    std::vector<int32_t> actual(ids.begin(), ids.end());
    std::vector<bpe::index_type> expect = {2028, 374, 264, 1296, 11914, 13};
    REQUIRE_THAT(actual, Catch::Matchers::Equals(expect));

    auto str = tokenizer.decode(ids.data_ptr(), ids.data_ptr() + ids.size(0));
    REQUIRE(str == "This is a test sentence.");

    std::vector<int32_t> tokens;
    tokenizer.encode(special_token::begin_text, tokens);
    REQUIRE(tokens.size() == 1);
    REQUIRE(tokens[0] == 128000);
}
