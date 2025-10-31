#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>


namespace metalchat {


struct command_property {
    std::string type;
    std::string description;
    std::optional<std::string> default_value;
};


struct command_parameters {
    std::string type;
    std::vector<std::string> required;
    std::unordered_map<std::string, command_property> properties;
};


struct command_metadata {
    const std::string type = "function";
    std::string name;
    std::string description;
    command_parameters parameters;

    std::string
    write_json() const;
};


class basic_command_statement {
public:
    virtual std::string
    get_name() const
        = 0;

    virtual std::optional<std::string>
    get_parameter(const std::string&) const = 0;

    virtual std::string
    str() const
        = 0;

    virtual ~basic_command_statement() {}
};


class command_statement : public basic_command_statement {
private:
    std::shared_ptr<basic_command_statement> _M_ptr;

public:
    command_statement(const std::shared_ptr<basic_command_statement>& call);

    template <typename CommandStatement>
    requires std::derived_from<CommandStatement, basic_command_statement>
    command_statement(CommandStatement&& call)
    : _M_ptr(std::make_shared<CommandStatement>(std::move(call)))
    {}

    std::string
    get_name() const;

    std::optional<std::string>
    get_parameter(const std::string&) const;

    std::string
    str() const;
};


class basic_command_scanner {
public:
    virtual std::string
    declare(const std::string& decl)
        = 0;

    virtual command_statement
    scan(const std::string& text)
        = 0;

    virtual ~basic_command_scanner() {}
};


class json_command_scanner;


class json_command_statement : public basic_command_statement {
private:
    friend class json_command_scanner;

    struct _Members;
    std::shared_ptr<_Members> _M_data;

    json_command_statement(_Members&&);

public:
    std::string
    get_name() const;

    std::optional<std::string>
    get_parameter(const std::string&) const;

    std::string
    str() const;
};


class json_command_scanner : public basic_command_scanner {
private:
    struct _Members;
    std::shared_ptr<_Members> _M_data;

public:
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

    json_command_scanner();
    json_command_scanner(const json_command_scanner&) = default;

    std::string
    declare(const std::string& decl);

    command_statement
    scan(const std::string& text);
};


} // namespace metalchat
