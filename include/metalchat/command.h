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
/// https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/ user guide. Note, that
/// format for Llama 3.1 and 3.2 are different according to that documentation, and here format
/// for Llama 3.2 is used.
struct command_metadata {
    std::string name;
    std::string description;
    command_parameters parameters;

    std::string
    write_json() const;
};


/// This class provides a functionality of declaring an interpreter command in multiple steps.
class command {
private:
    std::optional<std::string> _M_name;
    std::optional<std::string> _M_description;

public:
    command()
    : _M_name(std::nullopt),
      _M_description(std::nullopt)
    {}

    void
    name(const std::string& n)
    {
        _M_name = n;
    }

    void
    description(const std::string& d)
    {
        _M_description = d;
    }
};


} // namespace metalchat
