// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>

#include <metalchat/allocator.h>
#include <metalchat/command.h>
#include <metalchat/interpreter.h>
#include <metalchat/nn.h>
#include <metalchat/reference.h>
#include <metalchat/repository.h>

#include "metalchat/testing.h"


using namespace metalchat;


TEST_CASE("Test interpreter", "[llama]")
{
    auto repo_path = test_fixture_path() / "meta-llama/Llama-3.2-1B-Instruct/original";

    auto repository = filesystem_repository<reference::llama3>(repo_path);
    auto options = nn::default_llama3_1b_options();
    auto tokenizer = repository.retrieve_tokenizer("tokenizer.model");
    auto transformer = repository.retrieve_transformer("model.safetensors", options);

    auto interp = interpreter(transformer, tokenizer);

    auto command = R"({"name":"multiply",
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

{{ #metalchat.commands }}
{{ . }}
{{ /metalchat.commands }}

{{ metalchat.command_format }}
{{ extra_instructions }}
)";

    interp.declare_variable("extra_instructions", "answer in json");
    interp.declare_command(command, [](const command_statement&) -> std::string {
        return R"(print 113001120)";
    });
    interp.write(basic_message("system", prompt));
    interp.write(basic_message("user", "What is 12135 multiplied by 9312?"));

    std::cout << interp.exec().content() << std::endl;

    interp.write(basic_message("user", "what is the capital of Belgium?"));
    std::cout << interp.read_text() << std::endl;
}


TEST_CASE("Test filebuf interpreter", "[llama]")
{
    SKIP();

    auto repo_path = test_fixture_path() / "meta-llama/Llama-3.2-1B-Instruct/original";

    using Transformer = reference::llama3_traits<bf16, filebuf_memory_container<bf16>>;
    using Repository = filesystem_repository<Transformer>;
    using Allocator = filebuf_memory_allocator<void>;

    Repository repository(repo_path);
    auto options = nn::default_llama3_1b_options();
    auto tokenizer = repository.retrieve_tokenizer("tokenizer.model");
    auto transformer = repository.retrieve_transformer("model.safetensors", options, Allocator());

    auto interp = interpreter(transformer, tokenizer);

    interp.write(basic_message("system", "You are a helpful assistant"));
    interp.write(basic_message("user", "What is the capital of Germany?"));

    std::cout << interp.read_text() << std::endl;
}
