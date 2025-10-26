#include <string_view>

#include <jsoncons/json.hpp>
#include <jsoncons_ext/jsonschema/jsonschema.hpp>

#include <metalchat/command.h>


JSONCONS_ALL_MEMBER_TRAITS(metalchat::command_property, type, description, default_value);
JSONCONS_ALL_MEMBER_TRAITS(metalchat::command_parameters, type, required, properties);
JSONCONS_ALL_MEMBER_TRAITS(metalchat::command_metadata, name, description, parameters);


namespace metalchat {


namespace jsonschema = jsoncons::jsonschema;


struct json_command_scanner::_Members {
    using schema_type = jsonschema::json_schema<jsoncons::json>;

    schema_type command_schema;
    std::unordered_map<std::string, schema_type> commands;

    _Members()
    : command_schema(jsonschema::make_json_schema(
          jsoncons::json::parse(json_command_scanner::command_schema),
          jsonschema::evaluation_options().require_format_validation(true)
      )),
      commands()
    {}
};


json_command_scanner::json_command_scanner()
: _M_data(std::make_shared<_Members>())
{}


std::string
json_command_scanner::declare(const std::string& declaration)
{
    auto command = jsoncons::json::parse(declaration);
    if (!_M_data->command_schema.is_valid(command)) {
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

    _M_data->commands.insert_or_assign(command_name, std::move(command_schema));
    return command_name;
}


command_call
json_command_scanner::scan(const std::string& text)
{
    auto command = jsoncons::json::parse(text);
    auto command_name = command["name"].as<std::string>();
    auto& command_schema = _M_data->commands.at(command_name);

    if (!command_schema.is_valid(command)) {
        throw std::runtime_error("json_command_scanner: wrong command format");
    }

    return command_call{};
}


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
