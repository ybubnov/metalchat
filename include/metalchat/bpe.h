#pragma once

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cppcodec/base64_rfc4648.hpp>


namespace std {


template <typename SizeT, typename T>
void
hash_combine(SizeT& seed, T value)
{
    hash<T> make_hash;
    seed ^= make_hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}


template <std::integral T, class Allocator> struct hash<vector<T, Allocator>> {
    using argument_type = vector<T, Allocator>;

    size_t
    operator()(const argument_type& argument) const noexcept
    {
        size_t seed = argument.size();
        for (auto& element : argument) {
            hash_combine(seed, element);
        }
        return seed;
    }
};


} // namespace std


namespace metalchat {


class bpe {
private:
    using string_type = std::vector<uint8_t>;
    using index_type = uint32_t;

    using base64 = cppcodec::base64_rfc4648;

    std::unordered_map<string_type, index_type> _m_fmap;
    std::unordered_map<index_type, string_type> _m_rmap;

public:
    bpe(const std::filesystem::path& p)
    : _m_fmap(),
      _m_rmap()
    {
        std::ifstream file(p.c_str(), std::ios::binary);

        if (!file.is_open()) {
            throw std::invalid_argument(std::format("unable to open file '{}'", p.string()));
        }

        std::string line;
        while (std::getline(file, line)) {
            auto delim = line.find(" ");
            auto key_part = line.substr(0, delim);
            auto value_part = line.substr(delim + 1);

            index_type key = std::stoi(value_part);
            string_type value = base64::decode(key_part.c_str(), delim);

            _m_fmap.insert(std::make_pair(value, key));
            _m_rmap.insert(std::make_pair(key, value));
        }

        file.close();
    }
};


} // namespace metalchat
