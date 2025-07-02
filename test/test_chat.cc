#include <catch2/catch_test_macros.hpp>

#include <metalchat/allocator.h>
#include <metalchat/chat.h>
#include <metalchat/nn.h>


using namespace metalchat;


TEST_CASE("Test chat2", "[llama]")
{
    std::filesystem::path weights_path("../llama32.safetensors");
    std::filesystem::path tokens_path("../Llama-3.2-1B-Instruct/original/tokenizer.model");

    auto agent = construct_llama3_1b_minimal(weights_path, tokens_path);

    agent.send(basic_message("system", "You are a helpful assistant"));
    agent.send(basic_message("user", "What is the capital of France?"));
    std::cout << agent.receive_text() << std::endl;

    agent.send(basic_message("user", "what is the capital of Belgium?"));
    std::cout << agent.receive_text() << std::endl;
}
