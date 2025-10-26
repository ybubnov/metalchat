#include <string_view>

#include <jsoncons/json.hpp>
#include <jsoncons_ext/jsonschema/jsonschema.hpp>

#include <metalchat/command.h>


JSONCONS_ALL_MEMBER_TRAITS(metalchat::command_property, type, description, default_value);
JSONCONS_ALL_MEMBER_TRAITS(metalchat::command_parameters, type, required, properties);
JSONCONS_ALL_MEMBER_TRAITS(metalchat::command_metadata, name, description, parameters);


namespace metalchat {


namespace jsonschema = jsoncons::jsonschema;


class json_command_scanner : public basic_command_scanner {
private:
    using schema_type = jsonschema::json_schema<jsoncons::json>;

    schema_type _M_command_schema;
    std::unordered_map<std::string, schema_type> _M_commands;

    /// The structure represents the JSON format of the tool calling defined in
    /// https://platform.openai.com/docs/guides/function-calling#defining-functions
    /// user guide.
    static constexpr std::string_view command_schema = R"({
    "$id": "https://openai.com/schemas/function-call",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "type": {
        "type": "string",
        "enum": ["function"],
        "description": "This should always be function"
      },
      "name": {
        "type": "string",
        "description": "The function name"
      },
      "description": {
        "type": "string",
        "description": "Details on when and how to use the function"
      },
      "parameters": {
        "$ref": "https://json-schema.org/draft/2020-12/schema"
      },
      "strict": {
        "type": "boolean",
        "description": "Whether to enforce strict mode for the function call"
      }
    },
    "required": ["type", "name", "description", "parameters"]
    })";

public:
    json_command_scanner()
    : _M_command_schema(jsonschema::make_json_schema(
          jsoncons::json::parse(command_schema),
          jsonschema::evaluation_options().require_format_validation(true)
      ))
    {}

    std::string
    match(const std::string& declaration)
    {
        auto command = jsoncons::json::parse(declaration);
        if (!_M_command_schema.is_valid(command)) {
            // TODO: improve error.
            throw std::runtime_error("json_command_scanner: command schema is not valid");
        }

        auto command_name = command["name"].as<std::string>();

        // The name and type properties of the function declaration are not valid
        // properties of the JSON schema, therefore we remove name, and change type.
        command.erase("name");
        command["type"] = std::string("object");

        auto command_opts = jsonschema::evaluation_options().require_format_validation(true);
        auto command_schema = jsonschema::make_json_schema(command, command_opts);

        _M_commands.insert_or_assign(command_name, std::move(command_schema));
        return command_name;
    }

    command_call
    scan(const std::string& text)
    {
        auto command = jsoncons::json::parse(text);
        auto command_name = command["name"].as<std::string>();
        auto& command_schema = _M_commands.at(command_name);

        if (!command_schema.is_valid(command)) {
            throw std::runtime_error("json_command_scanner: wrong command format");
        }

        return command_call{};
    }
};


std::shared_ptr<basic_command_scanner>
make_json_scanner()
{
    return std::make_shared<json_command_scanner>();
}


std::string
command_metadata::write_json() const
{
    std::string output;
    jsoncons::encode_json(*this, output);

    return output;
}


} // namespace metalchat
