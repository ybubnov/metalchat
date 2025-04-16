#pragma once

#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <simdjson.h>
#include <sys/mman.h>

#include <metalchat/container.h>
#include <metalchat/device.h>
#include <metalchat/format.h>
#include <metalchat/tensor.h>


namespace metalchat {


class safetensor {
private:
    std::vector<std::size_t> _m_shape;
    void* _m_data;

public:
    safetensor(const std::vector<std::size_t>& shape, void* data)
    : _m_shape(shape),
      _m_data(data)
    {}

    inline std::size_t
    dim() const
    {
        return _m_shape.size();
    }

    template <typename T, std::size_t N>
    auto
    as() const
    {
        assert((N == _m_shape.size()));

        auto data = std::make_shared<weak_ref<T>>(static_cast<T*>(_m_data));
        return tensor<T, N, weak_ref<T>>(_m_shape.cbegin(), _m_shape.cend(), data);
    }

    template <typename T, std::size_t N>
    auto
    as(device& device) const
    {
        assert((N == _m_shape.size()));

        std::size_t numel = 1;
        for (const auto s : _m_shape) {
            numel *= s;
        }

        auto buf_size = numel * sizeof(T);
        auto buf
            = NS::TransferPtr(device->newBuffer(_m_data, buf_size, MTL::ResourceStorageModeShared));

        auto data = std::make_shared<device_ref<T>>(buf);
        return tensor<T, N, device_ref<T>>(_m_shape.cbegin(), _m_shape.cend(), data);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const safetensor& st)
    {
        os << "safetensor(shape=[" << st._m_shape << "])";
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
simdjson::ondemand::value::get() // noexcept
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
        // TODO: rework this implementation, so that vector is created with a
        // fixed number of elements and does not grow too much.
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

    using const_iterator = std::unordered_map<std::string, safetensor>::const_iterator;

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

    const_iterator
    begin() const
    {
        return m_tensors.begin();
    }

    iterator
    end()
    {
        return m_tensors.end();
    }

    const_iterator
    end() const
    {
        return m_tensors.end();
    }

    const safetensor&
    operator[](const std::string& tensor_name) const
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
