#pragma once

#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/format.h>
#include <metalchat/tensor/basic.h>


namespace metalchat {


class basic_memfile {
public:
    basic_memfile(const std::filesystem::path& p)
    {
        _m_file = std::fopen(p.c_str(), "r");
        if (_m_file == nullptr) {
            throw std::invalid_argument(std::format("unable to open file '{}'", p.string()));
        }

        _m_file_size = static_cast<std::size_t>(std::filesystem::file_size(p));
        int fd = fileno(_m_file);
        if (fd == -1) {
            throw std::invalid_argument(
                std::format("unable to get file descriptor for file '{}'", p.string())
            );
        }

        _m_map = static_cast<uint8_t*>(mmap(nullptr, _m_file_size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (_m_map == MAP_FAILED) {
            throw std::invalid_argument(
                std::format("unable to memory-map safetensors file '{}'", p.string())
            );
        }
    }

    std::size_t
    size() const noexcept
    {
        return _m_file_size;
    }

    basic_memfile&
    read(void* dest, std::size_t size)
    {
        if (_m_file_size < _m_file_off + size) {
            throw std::out_of_range(
                std::format("file is too short ({}) to read ({}) more bytes", _m_file_size, size)
            );
        }

        std::memcpy(dest, _m_map + _m_file_off, size);
        _m_file_off += size;

        return *this;
    }

    std::uint8_t*
    tellp() const
    {
        return _m_map + _m_file_off;
    }

    ~basic_memfile()
    {
        if (_m_map != MAP_FAILED && _m_map != nullptr) {
            munmap(_m_map, _m_file_size);
            _m_map = nullptr;
        }
        if (_m_file != nullptr) {
            std::fclose(_m_file);
            _m_file = nullptr;
        }
    }

private:
    std::FILE* _m_file = nullptr;
    std::size_t _m_file_size = 0;
    std::size_t _m_file_off = 0;
    std::uint8_t* _m_map = nullptr;
};


class safetensor {
private:
    std::vector<std::size_t> _m_shape;
    void* _m_data;

public:
    safetensor(const std::vector<std::size_t>& shape, void* data)
    : _m_shape(shape),
      _m_data(data)
    {}

    std::size_t
    dim() const
    {
        return _m_shape.size();
    }

    template <std::size_t N, allocator Allocator>
    auto
    as(Allocator alloc) const
    {
        using container_type = Allocator::container_type;
        using value_type = Allocator::value_type;

        auto data_ptr = static_cast<value_type*>(_m_data);

        return tensor<value_type, N, container_type>(
            _m_shape.cbegin(), _m_shape.cend(), data_ptr, alloc
        );
    }

    friend std::ostream&
    operator<<(std::ostream& os, const safetensor& st)
    {
        os << "safetensor(shape=[" << st._m_shape << "])";
        return os;
    }
};


class safetensor_file {
public:
    using iterator = std::unordered_map<std::string, safetensor>::iterator;

    using const_iterator = std::unordered_map<std::string, safetensor>::const_iterator;

    safetensor_file(const std::filesystem::path& p)
    : _m_memfile(std::make_shared<basic_memfile>(p))
    {
        parse();
    }

    std::size_t
    size() const noexcept
    {
        return _m_tensors.size();
    }

    iterator
    begin()
    {
        return _m_tensors.begin();
    }

    const_iterator
    begin() const
    {
        return _m_tensors.begin();
    }

    iterator
    end()
    {
        return _m_tensors.end();
    }

    const_iterator
    end() const
    {
        return _m_tensors.end();
    }

    const_iterator
    find(const std::string& tensor_name) const
    {
        return _m_tensors.find(tensor_name);
    }

    const safetensor&
    operator[](const std::string& tensor_name) const
    {
        return _m_tensors.at(tensor_name);
    }

private:
    std::shared_ptr<basic_memfile> _m_memfile;
    std::unordered_map<std::string, safetensor> _m_tensors;

    void
    parse();
};


} // namespace metalchat
