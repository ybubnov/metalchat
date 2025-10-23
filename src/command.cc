#include <glaze/json.hpp>

#include <metalchat/command.h>


template <> struct glz::meta<metalchat::command_property> {
    using T = metalchat::command_property;

    static constexpr auto value
        = object("type", &T::type, "description", &T::description, "default", &T::default_value);
};


template <> struct glz::meta<metalchat::command_parameters> {
    using T = metalchat::command_parameters;

    static constexpr auto value
        = object("type", &T::type, "required", &T::required, "properties", &T::properties);
};


template <> struct glz::meta<metalchat::command_metadata> {
    using T = metalchat::command_metadata;

    static constexpr auto value
        = object("name", &T::name, "description", &T::description, "parameters", &T::parameters);
};


namespace metalchat {


std::string
command_metadata::write_json() const
{
    std::string output;
    auto err = glz::write_json(*this, output);
    if (err) {
        throw std::runtime_error(glz::format_error(err, output));
    }

    return output;
}


} // namespace metalchat
