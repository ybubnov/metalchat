#pragma once

#include <filesystem>
#include <optional>
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


class command_call {};


class basic_command_scanner {
public:
    virtual std::string
    match(const std::string& declaration)
        = 0;

    virtual command_call
    scan(const std::string& text)
        = 0;

    virtual ~basic_command_scanner() {}
};


std::shared_ptr<basic_command_scanner>
make_json_scanner();


} // namespace metalchat
