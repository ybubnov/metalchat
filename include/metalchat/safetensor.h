#pragma once

#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <simdjson.h>
#include <sys/mman.h>

#include <metalchat/format.h>
#include <metalchat/tensor.h>


namespace metalchat {


class safetensor {
public:
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;
    void* data;

    safetensor(const std::vector<std::size_t>& shape_, void* data_)
    : shape(shape_),
      strides(shape_.size(), /*value=*/1),
      data(data_)
    {
        for (std::size_t i = shape_.size() - 2; i < shape_.size(); --i) {
            strides[i] = strides[i + 1] * shape_[i + 1];
        }
    }

    std::size_t
    dim()
    {
        return shape.size();
    }

    template <typename T, std::size_t N>
    tensor<T, N>
    as()
    {
        return tensor<T, N>(static_cast<T*>(data), shape.data(), strides.data());
    }

    friend std::ostream&
    operator<<(std::ostream& os, const safetensor& st)
    {
        os << "safetensor(shape=[" << st.shape << "], ";
        os << "strides=[" << st.strides << "])";
        return os;
    }
};


struct safetensor_ptr {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> data_offsets;

    friend std::ostream&
    operator<<(std::ostream& os, const safetensor_ptr& st)
    {
        os << "safetensor_ptr(name='" << st.name << "', dtype=" << st.dtype;
        os << ", shape=[" << st.shape << "]";
        os << ", data_offsets=[" << st.data_offsets << "]";
        os << ")";
        return os;
    }
};


} // namespace metalchat


using namespace simdjson;


template <>
simdjson_inline simdjson_result<std::vector<std::size_t>>
simdjson::ondemand::value::get() noexcept
{
    ondemand::array array;
    auto error = get_array().get(array);

    if (error) {
        return error;
    }

    std::vector<std::size_t> vec;

    for (auto v : array) {
        int64_t val;
        if ((error = v.get_int64().get(val))) {
            return error;
        }
        vec.push_back(static_cast<std::size_t>(val));
    }
    return vec;
}


template <>
simdjson_inline simdjson_result<metalchat::safetensor_ptr>
simdjson::ondemand::value::get() noexcept
{
    ondemand::object object;
    auto error = get_object().get(object);
    if (error) {
        return error;
    }

    metalchat::safetensor_ptr ptr;
    if ((error = object["dtype"].get_string(ptr.dtype))) {
        return error;
    }
    if ((error = object["shape"].get<std::vector<std::size_t>>().get(ptr.shape))) {
        return error;
    }
    if ((error = object["data_offsets"].get<std::vector<std::size_t>>().get(ptr.data_offsets))) {
        return error;
    }

    return ptr;
}


namespace metalchat {


class safetensor_file {
public:
    using iterator = std::unordered_map<std::string, safetensor>::iterator;

    safetensor_file(const std::filesystem::path& p)
    {
        m_file = std::fopen(p.c_str(), "r");
        if (m_file == nullptr) {
            throw std::invalid_argument(std::format("unable to open file '{}'", p.string()));
        }

        m_file_size = static_cast<std::size_t>(std::filesystem::file_size(p));
        int fd = fileno(m_file);
        if (fd == -1) {
            throw std::invalid_argument(
                std::format("unable to get file descriptor for file '{}'", p.string())
            );
        }

        m_map = static_cast<uint8_t*>(mmap(nullptr, m_file_size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (m_map == MAP_FAILED) {
            throw std::invalid_argument(
                std::format("unable to memory-map safetensors file '{}'", p.string())
            );
        }

        parse();
    }

    safetensor_file(safetensor_file&& other)
    {
        m_file = other.m_file;
        m_file_size = other.m_file_size;
        m_file_off = other.m_file_off;
        m_map = other.m_map;

        other.m_file_size = 0;
        other.m_file_off = 0;
        other.m_file = nullptr;
        other.m_map = nullptr;
    }

    inline std::size_t
    size() const noexcept
    {
        return m_file_size;
    }

    iterator
    begin()
    {
        return m_tensors.begin();
    }

    iterator
    end()
    {
        return m_tensors.end();
    }

    safetensor&
    operator[](const std::string& tensor_name)
    {
        return m_tensors.at(tensor_name);
    }

    ~safetensor_file()
    {
        if (m_map != MAP_FAILED) {
            munmap(m_map, m_file_size);
        }
        if (m_file != nullptr) {
            std::fclose(m_file);
        }
    }

private:
    std::FILE* m_file = nullptr;
    std::size_t m_file_size = 0;
    std::size_t m_file_off = 0;

    uint8_t* m_map = nullptr;

    std::unordered_map<std::string, safetensor> m_tensors;

    void
    parse()
    {
        // Read the length of the header and then the header itself, ensure that the
        // the file contains enough data to avoid reading from inaccessible regions.
        uint64_t header_size = 0;
        memory_read(&header_size, sizeof(header_size));

        char header[header_size];
        memory_read(&header, sizeof(header));

        simdjson::ondemand::parser json_parser;
        simdjson::padded_string header_padded(header, header_size);

        auto json_document = json_parser.iterate(header_padded);
        auto json_object = json_document.get_object();

        for (auto json_field : json_object) {
            std::string_view field_name = json_field.unescaped_key();
            if (field_name == "__metadata__") {
                continue;
            }

            safetensor_ptr tensor_ptr;
            auto error = json_field.value().get<safetensor_ptr>().get(tensor_ptr);
            if (error) {
                throw std::runtime_error(simdjson::error_message(error));
            }

            auto tensor_name = std::string(field_name);
            auto data = m_map + m_file_off + tensor_ptr.data_offsets[0];
            auto tensor = safetensor(tensor_ptr.shape, data);

            m_tensors.insert_or_assign(tensor_name, tensor);
        }
    }

    void
    memory_read(void* dest, std::size_t size)
    {
        if (m_file_size < m_file_off + size) {
            throw std::out_of_range(
                std::format("file is too short ({}) to read ({}) more bytes", m_file_size, size)
            );
        }

        std::memcpy(dest, m_map + m_file_off, size);
        m_file_off += size;
    }
};


} // namespace metalchat
