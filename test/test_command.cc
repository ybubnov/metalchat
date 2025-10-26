#include <catch2/catch_test_macros.hpp>

#include <metalchat/command.h>


using namespace metalchat;


TEST_CASE("Test JSON command scanner", "[interpreter]")
{
    std::string declaration = R"({
"type": "function",
"name": "get_weather",
"description": "Get weather in a particular location",
"parameters": {
  "location": {"type": "string", "description": "Location to get weather from"}
}
})";

    auto scanner = make_json_scanner();
    auto command_name = scanner->match(declaration);
    REQUIRE(command_name == "get_weather");

    scanner->scan(R"({"name": "get_weather", "parameters": {"location": "Berlin"}})");
}
