#include <catch2/catch_test_macros.hpp>

#include <metalchat/allocator.h>
#include <metalchat/command.h>
#include <metalchat/interpreter.h>
#include <metalchat/nn.h>


using namespace metalchat;


TEST_CASE("Test interpreter", "[llama]")
{
    std::filesystem::path weights_path("../llama32.safetensors");
    std::filesystem::path tokens_path("../Llama-3.2-1B-Instruct/original/tokenizer.model");

    auto options = nn::default_llama3_1b_options().heap_size(0);
    auto interp = make_llama3(weights_path, tokens_path, options);

    // interp.register_command(command_schema, [](const command_params& params) {
    // });

    command_metadata mul
        = {.name = "multiply",
           .description = "Multiply two numbers",
           .parameters
           = {.type = "object",
              .properties
              = {{"a", {.type = "number", .description = "First number"}},
                 {"b", {.type = "number", .description = "Second number"}}},
              .required = {"a", "b"}}};

    auto command = mul.write_json();

    auto prompt = R"(
Environment: ipython
Tools: multiply

# Tool Instructions
- When you need to multiply numbers, use the multiply tool
- Always call tools when appropriate rather than guessing

You have access to the following tool:

)" + command + R"(

To use a tool, respond with JSON in this format:
{"name": "multiply", "parameters": {"a": 5, "b": 3}}
)";

    interp.write(basic_message("system", prompt));
    interp.write(basic_message("user", "What is 12135 multiplied by 9312?"));

    std::cout << interp.read_text() << std::endl;

    interp.write(basic_message("user", "what is the capital of Belgium?"));
    std::cout << interp.read_text() << std::endl;
}
