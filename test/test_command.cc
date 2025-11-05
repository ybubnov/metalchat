// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>

#include <metalchat/command.h>


using namespace metalchat;


TEST_CASE("Test JSON command scanner", "[interpreter]")
{
    std::string statement = R"({
"type": "function",
"name": "get_weather",
"description": "Get weather in a particular location",
"parameters": {
  "type": "object",
  "properties": {
    "location": {"type": "string", "description": "Location to get weather from"}
  },
  "required": ["location"]
}
})";

    json_command_scanner scanner;
    auto command_name = scanner.declare(statement);
    REQUIRE(command_name == "get_weather");

    auto text = R"({"name": "get_weather", "parameters": {"location": "Berlin"}})";
    auto stmt = scanner.scan(text);

    REQUIRE(stmt.has_value());
    REQUIRE(stmt.value().get_name() == "get_weather");
    REQUIRE(stmt.value().get_parameter("location") == R"("Berlin")");
}
