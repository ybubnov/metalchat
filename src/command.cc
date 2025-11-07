// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <string_view>

#include <jsoncons/json.hpp>
#include <jsoncons_ext/jsonschema/jsonschema.hpp>

#include <metalchat/command.h>


namespace metalchat {


namespace jsonschema = jsoncons::jsonschema;


command_statement::command_statement(const std::shared_ptr<basic_command_statement>& stmt)
: _M_ptr(stmt)
{}


std::string
command_statement::get_name() const
{
    return _M_ptr->get_name();
}


std::optional<std::string>
command_statement::get_parameter(const std::string& p) const
{
    return _M_ptr->get_parameter(p);
}


std::string
command_statement::str() const
{
    return _M_ptr->str();
}


struct json_command_statement::_Members {
    std::string str;
    std::string name;
    jsoncons::json params;
};


json_command_statement::json_command_statement(_Members&& data)
: _M_data(std::make_shared<_Members>(std::move(data)))
{}


std::string
json_command_statement::get_name() const
{
    return _M_data->name;
}


std::optional<std::string>
json_command_statement::get_parameter(const std::string& p) const
{
    if (_M_data->params.contains(p)) {
        std::string value;
        jsoncons::encode_json(_M_data->params[p], value);
        return value;
    }

    return std::nullopt;
}


std::string
json_command_statement::str() const
{
    return _M_data->str;
}


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
json_command_scanner::declare(const std::string& decl)
{
    auto command = jsoncons::json::parse(decl);
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


std::optional<command_statement>
json_command_scanner::scan(const std::string& text)
{
    auto tag = std::string("<|python_tag|>");

    if (!text.starts_with(tag)) {
        return std::nullopt;
    }
    auto str = text.substr(tag.size(), text.size() - tag.size());

    auto command = jsoncons::json::parse(str);
    auto command_name = command["name"].as<std::string>();
    auto& command_schema = _M_data->commands.at(command_name);

    if (!command_schema.is_valid(command)) {
        return std::nullopt;
    }

    auto stmt = json_command_statement(json_command_statement::_Members{
        .str = str,
        .name = command_name,
        .params = std::move(command["parameters"]),
    });

    return command_statement(std::move(stmt));
}


std::shared_ptr<basic_command_scanner>
make_json_scanner()
{
    return std::make_shared<json_command_scanner>();
}


} // namespace metalchat
