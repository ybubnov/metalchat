#include <catch2/catch_test_macros.hpp>

#include <metalchat/allocator.h>
#include <metalchat/interpreter.h>
#include <metalchat/nn.h>
#include <metalchat/text.h>


using namespace metalchat;
using namespace metalchat::text;


TEST_CASE("Test interpreter", "[llama]")
{
    std::filesystem::path weights_path("../llama32.safetensors");
    std::filesystem::path tokens_path("../Llama-3.2-1B-Instruct/original/tokenizer.model");

    auto interp = make_llama3(weights_path, tokens_path);

    interp.send(basic_message("system", "You are a helpful interpreter"));
    interp.send(basic_message("user", "What is the capital of France?"));
    std::cout << interp.receive_text() << std::endl;

    interp.send(basic_message("user", "what is the capital of Belgium?"));
    std::cout << interp.receive_text() << std::endl;
}
