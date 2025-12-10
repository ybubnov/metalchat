// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <streambuf>
#include <vector>

#include <metalchat/metal.h>


namespace metalchat {


class basic_memfile {
public:
    using char_type = char;
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


template <typename T, typename U>
std::shared_ptr<T>
make_pointer_alias(const std::shared_ptr<T>& ptr1, const std::shared_ptr<U>& ptr2)
{
    using t_pointer = std::shared_ptr<T>;
    using u_pointer = std::shared_ptr<U>;
    using union_pointer = std::pair<t_pointer, u_pointer>;

    auto ptr = std::make_shared<union_pointer>(ptr1, ptr2);
    return t_pointer(ptr, ptr->first.get());
}


struct basic_container {

    virtual void*
    data_ptr() = 0;

    virtual const void*
    data_ptr() const = 0;

    virtual ~basic_container() = default;
};


template <typename T> struct memory_container : public basic_container {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    /// Get a write access to the underlying container data.
    virtual pointer
    data() = 0;

    /// Get a read access to the underlying container data.
    virtual const_pointer
    data() const = 0;

    virtual std::size_t
    size() const = 0;

    void*
    data_ptr() override
    {
        return data();
    }

    const void*
    data_ptr() const override
    {
        return data();
    }

    pointer
    operator*()
    {
        return data();
    }

    virtual ~memory_container() = default;
};


template <typename Container>
concept contiguous_container = requires {
    typename Container::value_type;
    typename Container::pointer;
    typename Container::const_pointer;

    requires std::derived_from<Container, memory_container<typename Container::value_type>>;
};


template <typename T, contiguous_container Container> struct container_rebind;


template <contiguous_container> struct container_remove_type;


template <contiguous_container Container> struct container_offset;


/// This template class provides the standardized way to access various properties of
/// \ref contiguous_container. The library allocators and other components access containers
/// through this template.
///
/// \tparam Container a contiguous memory container type.
template <contiguous_container Container> struct container_traits {
    using container_type = Container;
    using container_pointer = std::shared_ptr<container_type>;

    using value_type = container_type::value_type;

    using pointer = container_type::pointer;
    using const_pointer = const container_type::pointer;
    using void_pointer = void*;
    using const_void_pointer = const void*;

    /// Container type after type rebinding.
    template <typename T> using rebind_container = container_rebind<T, Container>::type;

    /// Container traits of the rebind container type.
    template <typename T> using rebind_traits = container_traits<rebind_container<T>>;

    /// A template method to rebind container type, when logic is present.
    template <typename T> static constexpr auto rebind = container_rebind<T, Container>::rebind;

    /// Container type after storage offset.
    using offset_container = container_offset<container_type>::type;

    /// Container traifs of the offset container type.
    using offset_traits = container_traits<offset_container>;

    /// A method to shift container type, when logic is present.
    static constexpr auto offset = container_offset<Container>::offset;

    /// Returns a pointer to the beginning of the container underlying storage.
    ///
    /// \param container a contiguous memory container.
    static const_void_pointer
    begin(const container_type& container)
    {
        return static_cast<const std::uint8_t*>(container.data_ptr());
    }

    /// Returns a pointer to the beginning of the container underlying storage.
    static const_void_pointer
    begin(const container_pointer& container_ptr)
    {
        return begin(*container_ptr);
    }

    /// Returns a pointer to the end (i.e. the element after the last element) of
    /// the container underlying storage.
    ///
    /// \param container a contiguous memory container.
    static const_void_pointer
    end(const container_type& container)
    {
        return static_cast<const std::uint8_t*>(begin(container)) + container.size();
    }

    /// Returns a pointer to the end of the container underlying storage.
    static const_void_pointer
    end(const container_pointer& container_ptr)
    {
        return end(*container_ptr);
    }

    /// Checks whether or not a given container contains the specified pointer.
    ///
    /// \param container a contiguous memory container.
    /// \param ptr a pointer that is checked.
    static bool
    contains(const container_type& container, const_void_pointer ptr)
    {
        return (ptr >= begin(container)) && (ptr <= end(container));
    }

    /// Checks whether or not a given container contains the specified range contiguous memory.
    ///
    /// \param container a contiguous memory container.
    /// \param first a start position of the contiguous memory.
    /// \param size a size of the contiguous memory.
    static bool
    contains(const container_type& container, const_void_pointer first, std::size_t size)
    {
        const_void_pointer last = static_cast<const std::uint8_t*>(first) + size;
        return contains(container, first) && contains(container, last);
    }

    /// Checks whether or not a given container contains the specified range contiguous memory.
    ///
    /// \param container_ptr a pointer to a contiguous memory container.
    /// \param first a start position of the contiguous memory.
    /// \param size a size of the contiguous memory.
    static bool
    contains(const container_pointer& container_ptr, const_void_pointer first, std::size_t size)
    {
        return contains(*container_ptr, first, size);
    }
};


template <typename T> struct random_memory_container : public memory_container<T> {
private:
    std::shared_ptr<void> _M_data = nullptr;
    std::size_t _M_size;
    std::size_t _M_offset;

    template <typename U> friend struct random_memory_container;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    random_memory_container(std::shared_ptr<void> data, std::size_t size, std::size_t offset = 0)
    : _M_data(data),
      _M_size(size),
      _M_offset(offset)
    {}

    std::size_t
    size() const
    {
        return _M_size;
    }

    pointer
    data()
    {
        return static_cast<pointer>(storage_ptr());
    }

    const_pointer
    data() const
    {
        return static_cast<const_pointer>(storage_ptr());
    }

    std::shared_ptr<void>
    storage() const
    {
        return _M_data;
    }

    std::size_t
    storage_offset() const
    {
        return _M_offset;
    }

    void*
    storage_ptr() const
    {
        return static_cast<std::uint8_t*>(_M_data.get()) + storage_offset();
    }
};


template <typename T> struct container_remove_type<random_memory_container<T>> {
    using type = random_memory_container<void>;
};


template <typename T> struct container_rebind<T, random_memory_container<void>> {
    using type = random_memory_container<T>;
    using pointer = std::shared_ptr<type>;

    static pointer
    rebind(std::shared_ptr<random_memory_container<void>> ptr)
    {
        auto size = ptr->size();
        auto offset = ptr->storage_offset();
        auto container_ptr = std::make_shared<type>(ptr->storage(), size, offset);

        return make_pointer_alias(container_ptr, ptr);
    }
};


template <typename T> struct container_offset<random_memory_container<T>> {
    using type = random_memory_container<T>;
    using pointer = std::shared_ptr<type>;

    static pointer
    offset(pointer ptr, std::size_t off)
    {
        auto size = ptr->size() - off;
        auto offset = ptr->storage_offset() + off;
        auto container_ptr = std::make_shared<type>(ptr->storage(), size, offset);

        return make_pointer_alias(container_ptr, ptr);
    }
};


template <typename T> struct vector_memory_container : public memory_container<T> {
private:
    std::vector<T> _M_data;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    vector_memory_container(std::vector<T>&& data)
    : _M_data(std::move(data))
    {}

    vector_memory_container()
    : _M_data()
    {}

    std::size_t
    size() const
    {
        return _M_data.size() * sizeof(value_type);
    }

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
    using const_pointer = const value_type*;

    metal::shared_buffer _M_mem;
    std::size_t _M_size;
    std::size_t _M_offset;

    hardware_memory_container(metal::shared_buffer mem, std::size_t offset = 0)
    : _M_mem(mem),
      _M_size(metal::size(_M_mem) - offset),
      _M_offset(offset)
    {}

    std::size_t
    size() const
    {
        return _M_size;
    }

    pointer
    data()
    {
        return static_cast<pointer>(storage_ptr());
    }

    const_pointer
    data() const
    {
        return static_cast<const_pointer>(storage_ptr());
    }

    metal::shared_buffer
    storage() const
    {
        return _M_mem;
    }

    std::size_t
    storage_offset() const
    {
        return _M_offset;
    }

    void*
    storage_ptr() const
    {
        return static_cast<std::uint8_t*>(metal::data(_M_mem)) + storage_offset();
    }
};


template <typename T> struct container_remove_type<hardware_memory_container<T>> {
    using type = hardware_memory_container<void>;
};


template <typename T> struct container_rebind<T, hardware_memory_container<void>> {
    using type = hardware_memory_container<T>;
    using pointer = std::shared_ptr<type>;

    static pointer
    rebind(std::shared_ptr<hardware_memory_container<void>> ptr)
    {
        // You might wonder, why would somebody create an alias to the container that
        // was just converted the type. The answer is in the implementation of the
        // safetensors. The most efficient method of opening a safetensors file is mapping
        // it to the memory (mmap), and then using a single buffer for slicing storage
        // for tensors. But that file must remain in the memory until the last tensor that
        // is associated with that memory-mapped memory is not removed.
        //
        // API of the Allocators in this library return containers, therefore safetensor
        // loading logic creates an alias to the container types, not to the underlying
        // metal buffers.
        //
        // In order to keep the mmap-file pointer even after rebinding of the container type,
        // we must store it into the final pointer.
        auto container_ptr = std::make_shared<type>(ptr->storage(), ptr->storage_offset());
        return make_pointer_alias(container_ptr, ptr);
    }
};


template <typename T> struct container_offset<hardware_memory_container<T>> {
    using type = hardware_memory_container<T>;
    using pointer = std::shared_ptr<type>;

    static pointer
    offset(pointer ptr, std::size_t offset)
    {
        auto container_ptr = std::make_shared<type>(ptr->storage(), ptr->storage_offset() + offset);
        return make_pointer_alias(container_ptr, ptr);
    }
};


template <typename T> struct scalar_memory_container : public memory_container<T> {
private:
    T _M_data;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    scalar_memory_container(T data)
    : _M_data(data)
    {}

    std::size_t
    size() const
    {
        return sizeof(T);
    }

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
    std::size_t _M_size;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    /// Constructs a new instance of a file-buffered container and initializes it with
    /// the provided data. After construction file is not mapped into the memory.
    filebuf_memory_container(const_pointer data, std::size_t size)
    : _M_filebuf(std::make_shared<basic_memfile>()),
      _M_size(size * sizeof(value_type))
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

    std::size_t
    size() const
    {
        return _M_size;
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


template <typename T> struct container_remove_type<filebuf_memory_container<T>> {
    using type = filebuf_memory_container<void>;
};


template <typename T> class offsetted_container_adapter : public memory_container<T> {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    using container_type = memory_container<T>;
    using container_pointer = std::shared_ptr<container_type>;

    offsetted_container_adapter(const container_pointer& container_ptr, std::size_t offset)
    : _M_container(container_ptr),
      _M_offset(offset)
    {}

    pointer
    data()
    {
        void* ptr = static_cast<std::uint8_t*>(_M_container->data_ptr()) + _M_offset;
        return static_cast<pointer>(ptr);
    }

    const_pointer
    data() const
    {
        const void* ptr = static_cast<const std::uint8_t*>(_M_container->data_ptr()) + _M_offset;
        return static_cast<const_pointer>(ptr);
    }

    std::size_t
    size() const
    {
        return _M_container->size() - _M_offset;
    }

private:
    std::shared_ptr<memory_container<T>> _M_container;
    std::size_t _M_offset;
};


template <typename T> struct container_remove_type<offsetted_container_adapter<T>> {
    using type = offsetted_container_adapter<void>;
};


} // namespace metalchat
