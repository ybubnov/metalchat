#include <jsoncons/json.hpp>

#include <metalchat/command.h>


JSONCONS_ALL_MEMBER_TRAITS(metalchat::command_property, type, description, default_value);
JSONCONS_ALL_MEMBER_TRAITS(metalchat::command_parameters, type, required, properties);
JSONCONS_ALL_MEMBER_TRAITS(metalchat::command_metadata, name, description, parameters);


namespace metalchat {


std::string
command_metadata::write_json() const
{
    std::string output;
    jsoncons::encode_json(*this, output);

    return output;
}


} // namespace metalchat
