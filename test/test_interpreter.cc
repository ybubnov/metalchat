// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>

#include <metalchat/allocator.h>
#include <metalchat/autoloader.h>
#include <metalchat/command.h>
#include <metalchat/interpreter.h>
#include <metalchat/nn.h>

#include "metalchat/testing.h"


using namespace metalchat;


TEST_CASE("Test interpreter", "[llama]")
{
    auto repo_path = test_fixture_path() / "llama3.2:1b-instruct";
    auto tokens_path = repo_path / "original/tokenizer.model";

    hardware_accelerator accelerator;
    autoloader loader(repo_path);
    text::bpe tokenizer(tokens_path);

    auto transformer = loader.load(nn::default_llama3_1b_options(), accelerator);
    auto interp = interpreter(transformer, tokenizer);

    auto command = R"({
"name":"multiply",
"type": "function",
"description":"multiply two numbers",
"parameters":{
  "a":{"type":"number","description":"first number"},
  "b":{"type":"number","description":"second number"}
}})";

    auto prompt = R"(Environment: ipython

# Tool Instructions
- When you need to multiply numbers, use the multiply tool
- Always call tools when appropriate rather than guessing

You have access to the following tools:

{{ $METALCHAT_COMMANDS }}
{{ $METALCHAT_COMMAND_FORMAT }}
{{ $EXTRA_INSTRUCTIONS }}
)";

    interp.declare_variable("EXTRA_INSTRUCTIONS", "answer in english");
    interp.declare_command(command, [](const command_statement&) -> std::string {
        return R"(113001120)";
    });
    interp.write(basic_message("system", prompt));
    interp.write(basic_message("user", "What is 12135 multiplied by 9312?"));

    std::cout << interp.exec().content() << std::endl;

    interp.write(basic_message("user", "what is the capital of Belgium?"));
    std::cout << interp.read_text() << std::endl;
}
