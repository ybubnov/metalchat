#include <catch2/catch_test_macros.hpp>


#include <metalchat/bpe.h>


using namespace metalchat;


TEST_CASE("Test BPE open", "[bpe]") { bpe("../Llama-3.2-1B/original/tokenizer.model"); }
