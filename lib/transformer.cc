// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <jsoncons/json.hpp>
#include <jsoncons_ext/jsonpath/jsonpath.hpp>
#include <jsoncons_ext/jsonpointer/jsonpointer.hpp>

#include <metalchat/transformer.h>


namespace metalchat {
namespace detail {


namespace jsonpath = jsoncons::jsonpath;
namespace jsonpointer = jsoncons::jsonpointer;


struct json_object::_Members {
    jsoncons::json data;

    template <typename T>
    void
    replace(const std::string& key, T&& value)
    {
        auto query = std::string("$.") + key;
        jsonpath::json_replace(data, query, std::move(value));
    }
};


json_object::json_object(std::istream& is)
: _M_members(std::make_shared<_Members>())
{
    _M_members->data = jsoncons::json::parse(is);
}


void
json_object::merge(const std::string& key, bool value)
{
    _M_members->replace(key, std::move(value));
}


void
json_object::merge(const std::string& key, int value)
{
    _M_members->replace(key, std::move(value));
}


void
json_object::merge(const std::string& key, float value)
{
    _M_members->replace(key, std::move(value));
}


void
json_object::merge(const std::string& key, std::string&& value)
{
    _M_members->replace(key, std::move(value));
}


void
json_object::write(std::ostream& os) const
{
    os << _M_members->data;
}


json_object::iterator
json_object::begin() const
{
    std::unordered_map<std::string, std::string> container;

    auto flat_json = jsonpointer::flatten(_M_members->data);
    for (const auto& member : flat_json.object_range()) {
        auto key = member.key();
        std::replace(key.begin(), key.end(), '/', '.');
        container.insert_or_assign(key.substr(1), member.value().as<std::string>());
    }

    return iterator(std::move(container));
}


json_object::iterator
json_object::end() const
{
    return iterator();
}


} // namespace detail
} // namespace metalchat
