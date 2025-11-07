// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

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

    command_metadata mul
        = {.name = "multiply",
           .type = "function",
           .description = "Multiply two numbers",
           .parameters
           = {.type = "object",
              .properties
              = {{"a", {.type = "number", .description = "First number"}},
                 {"b", {.type = "number", .description = "Second number"}}},
              .required = {"a", "b"}}};

    auto command = mul.write_json();

    auto prompt = R"(Environment: ipython

# Tool Instructions
- When you need to multiply numbers, use the multiply tool
- Always call tools when appropriate rather than guessing

You have access to the following tools:

{{ $METALCHAT_COMMANDS }}
{{ $METALCHAT_COMMAND_FORMAT }}
{{ $MYVAR }}
)";

    interp.declare_variable("MYVAR", "you're cute");
    interp.declare_command(command, [](const command_statement&) -> std::string {
        return R"(113001120)";
    });
    interp.write(basic_message("system", prompt));
    interp.write(basic_message("user", "What is 12135 multiplied by 9312?"));

    std::cout << interp.exec().content() << std::endl;

    interp.write(basic_message("user", "what is the capital of Belgium?"));
    std::cout << interp.read_text() << std::endl;
}
