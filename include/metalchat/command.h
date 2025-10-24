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


/// The structure represents the JSON format of the tool calling defined in
/// https://platform.openai.com/docs/guides/function-calling?strict-mode=enabled#defining-functions
/// user guide.
struct command_metadata {
    const std::string type = "function";
    std::string name;
    std::string description;
    command_parameters parameters;

    std::string
    write_json() const;
};


struct argument {
private:
    std::string _M_name;

public:
    argument(const std::string& name)
    : _M_name(name)
    {}

    argument&
    description(const std::string& description)
    {
        return *this;
    }
};


/// This class provides a functionality of declaring an interpreter command in multiple steps.
class command {
private:
    std::string _M_name;
    std::optional<std::string> _M_description;

public:
    command(const std::string& name)
    : _M_name(name),
      _M_description(std::nullopt)
    {}

    command&
    description(const std::string& d)
    {
        _M_description = d;
        return *this;
    }

    command&
    argument(const argument& arg)
    {
        return *this;
    }
};


} // namespace metalchat
