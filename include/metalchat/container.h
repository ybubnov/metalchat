#pragma once

#include <cstdio>
#include <filesystem>
#include <functional>
#include <vector>

#include <metalchat/metal.h>


namespace metalchat {


class basic_memfile {
public:
    using char_type = std::uint8_t;
    using pos_type = std::size_t;

    basic_memfile(const std::filesystem::path& p);
    basic_memfile(const std::filesystem::path& p, const std::string& mode);
    basic_memfile();

    bool
    is_mapped() const noexcept;

    basic_memfile&
    declare_mapped();

    basic_memfile&
    undeclare_mapped();

    std::size_t
    size() const noexcept;

    const char_type*
    data() const noexcept;

    char_type*
    data() noexcept;

    /// Returns output position indicator.
    pos_type
    tellp() const noexcept;

    /// Returns input position indicator.
    pos_type
    tellg() const noexcept;

    /// Extract characters from the file.
    basic_memfile&
    read(char_type* d, std::size_t size);

    basic_memfile&
    read(void* d, std::size_t size);

    /// Insert characters to the file.
    basic_memfile&
    write(const char_type* s, std::size_t size);

    basic_memfile&
    write(const void* s, std::size_t size);

    void
    close();

    ~basic_memfile();

private:
    std::FILE* _M_file = nullptr;
    std::size_t _M_file_size = 0;
    pos_type _M_file_p = 0;
    pos_type _M_file_g = 0;
    char_type* _M_map = nullptr;
};


template <typename T> struct memory_container {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    /// Get a write access to the underlying container data.
    virtual pointer
    data()
        = 0;

    /// Get a read access to the underlying container data.
    virtual const_pointer
    data() const
        = 0;

    pointer
    operator*()
    {
        return data();
    }

    virtual ~memory_container() {}
};


template <typename Container>
concept contiguous_container = requires {
    typename Container::value_type;
} && std::derived_from<Container, memory_container<typename Container::value_type>>;


template <typename It>
concept forward_container_iterator_t = std::forward_iterator<It> && requires(It it) {
    typename std::iterator_traits<It>::value_type;
    typename std::iterator_traits<It>::value_type::element_type;

    requires contiguous_container<typename std::iterator_traits<It>::value_type::element_type>;
};


template <typename T> class rebind_memory_container : memory_container<T> {
private:
    std::shared_ptr<memory_container<void>> _M_container;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    template <contiguous_container Container>
    requires std::same_as<typename Container::value_type, void>
    rebind_memory_container(const std::shared_ptr<Container>& container)
    {}

    pointer
    data()
    {
        return static_cast<T>(_M_container->data());
    }

    const_pointer
    data() const
    {
        return static_cast<T>(_M_container->data());
    }
};


template <typename T> struct reference_memory_container : public memory_container<T> {
private:
    T* _M_data = nullptr;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    reference_memory_container(pointer data)
    : _M_data(data)
    {}

    reference_memory_container(const reference_memory_container& ref)
    : reference_memory_container(ref._M_data)
    {}

    pointer
    data()
    {
        return _M_data;
    }

    const_pointer
    data() const
    {
        return _M_data;
    }

    ~reference_memory_container() { _M_data = nullptr; }
};


template <typename T>
auto
make_reference_container(T* data)
{
    return std::make_shared<reference_memory_container<T>>(data);
}


template <typename T> struct random_memory_container : public memory_container<T> {
private:
    std::shared_ptr<T[]> _M_data = nullptr;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    random_memory_container(T* data)
    : _M_data(data)
    {}

    pointer
    data()
    {
        return _M_data.get();
    }

    const_pointer
    data() const
    {
        return _M_data.get();
    }

    template <typename U> requires std::convertible_to<U, T>
    operator random_memory_container<U>() const
    {
        return random_memory_container<U>(_M_data);
    }
};


template <typename T> struct vector_memory_container : public memory_container<T> {
private:
    std::vector<T> _M_data;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    vector_memory_container(std::vector<T>&& data)
    : _M_data(std::move(data))
    {}

    vector_memory_container()
    : _M_data()
    {}

    pointer
    data()
    {
        return _M_data.data();
    }

    const value_type*
    data() const
    {
        const value_type* ptr = _M_data.data();
        return ptr;
    }
};


template <typename T> struct hardware_memory_container : public memory_container<T> {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    metal::shared_buffer _M_mem;
    std::size_t _M_off;

    hardware_memory_container(metal::shared_buffer mem, std::size_t off = 0)
    : _M_mem(mem),
      _M_off(off)
    {}

    pointer
    data()
    {
        return static_cast<T*>(storage_ptr());
    }

    const_pointer
    data() const
    {
        return static_cast<const_pointer>(storage_ptr());
    }

    template <typename U> requires std::convertible_to<U, T>
    operator hardware_memory_container<U>() const
    {
        return hardware_memory_container<U>(_M_mem);
    }

    bool
    contains(const void* ptr) const
    {
        return (ptr >= begin()) && (ptr <= end());
    }

    const void*
    begin() const
    {
        return storage_ptr();
    }

    const void*
    end() const
    {
        return static_cast<const std::uint8_t*>(begin()) + metal::size(_M_mem);
    }

    metal::shared_buffer
    storage() const
    {
        return _M_mem;
    }

    std::size_t
    storage_offset() const
    {
        return _M_off;
    }

    void*
    storage_ptr() const
    {
        return static_cast<std::uint8_t*>(metal::data(_M_mem)) + storage_offset();
    }
};


template <typename T> struct scalar_memory_container : public memory_container<T> {
private:
    T _M_data;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    scalar_memory_container(T data)
    : _M_data(data)
    {}

    pointer
    data()
    {
        return &_M_data;
    }

    const_pointer
    data() const
    {
        return const_pointer(&_M_data);
    }
};


template <typename T>
auto
make_scalar_container(T data)
{
    return std::make_shared<scalar_memory_container<T>>(data);
}


/// A container that keeps data within a temporary file.
///
/// When users need to get read (or write) access to the file, it's mapped to the memory and
/// remains mapped, when `filebuf_memory_container::park` method is called.
template <typename T> struct filebuf_memory_container : public memory_container<T> {
private:
    std::shared_ptr<basic_memfile> _M_filebuf;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;

    /// Constructs a new instance of a file-buffered container and initializes it with
    /// the provided data. After construction file is not mapped into the memory.
    filebuf_memory_container(const_pointer data, std::size_t size)
    : _M_filebuf(std::make_shared<basic_memfile>())
    {
        _M_filebuf->write(data, sizeof(value_type) * size);
    }

    /// Method evicts memory-mapped file from the memory. When the file is not memory-mapped
    /// method does absolutely nothing, so calling method multiple time is safe.
    void
    park() const
    {
        _M_filebuf->undeclare_mapped();
    }

    pointer
    data()
    {
        _M_filebuf->declare_mapped();
        return reinterpret_cast<pointer>(_M_filebuf->data());
    }

    const_pointer
    data() const
    {
        _M_filebuf->declare_mapped();
        return reinterpret_cast<const_pointer>(_M_filebuf->data());
    }
};


} // namespace metalchat
