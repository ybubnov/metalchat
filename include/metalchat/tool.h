#pragma once

#include <filesystem>


namespace metalchat {
namespace text {


class tool {
public:
    static tool
    load(const std::filesystem::path& p);

    void
    description(const std::string& d);
};


class tool_call {};


class tool_parser {};


} // namespace text
} // namespace metalchat
