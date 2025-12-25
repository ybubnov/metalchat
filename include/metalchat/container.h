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

    basic_memfile(const std::filesystem::path& p, std::ios::openmode mode);
    basic_memfile(const std::filesystem::path& p);
    basic_memfile(std::ios::openmode mode);
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
    std::ios::openmode _M_mode = std::ios::in;

    bool
    writable() const;
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
    virtual std::size_t
    size() const = 0;

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

    /// Get a type-erased write access to the underlying container data.
    void*
    data_ptr() override
    {
        return data();
    }

    /// Get a type-erased read access to the underlying container data.
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
concept contiguous_container = requires(std::remove_reference_t<Container> const c) {
    typename Container::value_type;
    typename Container::pointer;
    typename Container::const_pointer;
    typename Container::storage_type;

    requires std::derived_from<Container, memory_container<typename Container::value_type>>;

    { c.storage() } -> std::same_as<const typename Container::storage_type&>;
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
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = std::shared_ptr<void>;

    random_memory_container(const storage_type& storage, std::size_t size, std::size_t offset = 0)
    : _M_storage(storage),
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

    const storage_type&
    storage() const
    {
        return _M_storage;
    }

    std::size_t
    storage_offset() const
    {
        return _M_offset;
    }

    void*
    storage_ptr() const
    {
        return static_cast<std::uint8_t*>(_M_storage.get()) + storage_offset();
    }

private:
    storage_type _M_storage = nullptr;
    std::size_t _M_size;
    std::size_t _M_offset;

    template <typename U> friend struct random_memory_container;
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
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = std::vector<T>;

    vector_memory_container(storage_type&& storage)
    : _M_storage(std::move(storage))
    {}

    vector_memory_container()
    : _M_storage()
    {}

    std::size_t
    size() const
    {
        return _M_storage.size() * sizeof(value_type);
    }

    pointer
    data()
    {
        return _M_storage.data();
    }

    const value_type*
    data() const
    {
        const value_type* ptr = _M_storage.data();
        return ptr;
    }

    const storage_type&
    storage() const
    {
        return _M_storage;
    }

private:
    storage_type _M_storage;
};


template <typename T> struct hardware_memory_container : public memory_container<T> {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = metal::shared_buffer;

    hardware_memory_container(const storage_type& storage, std::size_t offset = 0)
    : _M_storage(storage),
      _M_size(metal::size(storage) - offset),
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

    const storage_type&
    storage() const
    {
        return _M_storage;
    }

    std::size_t
    storage_offset() const
    {
        return _M_offset;
    }

    void*
    storage_ptr() const
    {
        return static_cast<std::uint8_t*>(metal::data(_M_storage)) + storage_offset();
    }

private:
    storage_type _M_storage;
    std::size_t _M_size;
    std::size_t _M_offset;
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
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = T;

    scalar_memory_container(const storage_type& storage)
    : _M_storage(storage)
    {}

    std::size_t
    size() const
    {
        return sizeof(T);
    }

    pointer
    data()
    {
        return &_M_storage;
    }

    const_pointer
    data() const
    {
        return const_pointer(&_M_storage);
    }

    const storage_type&
    storage() const
    {
        return _M_storage;
    }

private:
    storage_type _M_storage;
};


/// A container that keeps data within a temporary file.
///
/// When users need to get read (or write) access to the file, it's mapped to the memory and
/// remains mapped, when `filebuf_memory_container::park` method is called.
template <typename T> struct filebuf_memory_container : public memory_container<T> {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = std::shared_ptr<basic_memfile>;

    /// Constructs a new instance of a file-buffered container and initializes it with
    /// the provided data. After construction file is not mapped into the memory.
    filebuf_memory_container(const_pointer data, std::size_t size)
    : _M_storage(std::make_shared<basic_memfile>(std::ios::in | std::ios::out)),
      _M_size(size),
      _M_offset(0)
    {
        _M_storage->write(data, size);
        _M_storage->undeclare_mapped();
    }

    filebuf_memory_container(const storage_type& storage, std::size_t size, std::size_t offset)
    : _M_storage(storage),
      _M_size(size),
      _M_offset(offset)
    {}

    /// Method evicts memory-mapped file from the memory. When the file is not memory-mapped
    /// method does absolutely nothing, so calling method multiple time is safe.
    void
    park() const
    {
        _M_storage->undeclare_mapped();
    }

    void
    unpark() const
    {
        _M_storage->declare_mapped();
    }

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

    const storage_type&
    storage() const
    {
        return _M_storage;
    }

    std::size_t
    storage_offset() const
    {
        return _M_offset;
    }

    void*
    storage_ptr() const
    {
        _M_storage->declare_mapped();
        return _M_storage->data() + storage_offset();
    }

private:
    storage_type _M_storage;
    std::size_t _M_size;
    std::size_t _M_offset;
};


template <typename T> struct container_remove_type<filebuf_memory_container<T>> {
    using type = filebuf_memory_container<void>;
};


template <typename T> struct container_rebind<T, filebuf_memory_container<void>> {
    using type = filebuf_memory_container<T>;
    using pointer = std::shared_ptr<type>;

    static pointer
    rebind(std::shared_ptr<filebuf_memory_container<void>> ptr)
    {
        auto size = ptr->size();
        auto offset = ptr->storage_offset();
        auto container_ptr = std::make_shared<type>(ptr->storage(), size, offset);

        return make_pointer_alias(container_ptr, ptr);
    }
};


template <typename T> struct container_offset<filebuf_memory_container<T>> {
    using type = filebuf_memory_container<T>;
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


template <typename T> class offsetted_container_adapter : public memory_container<T> {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = std::shared_ptr<memory_container<T>>;

    offsetted_container_adapter(const storage_type& storage, std::size_t offset)
    : _M_storage(storage),
      _M_offset(offset)
    {}

    pointer
    data()
    {
        void* ptr = static_cast<std::uint8_t*>(_M_storage->data_ptr()) + _M_offset;
        return static_cast<pointer>(ptr);
    }

    const_pointer
    data() const
    {
        const void* ptr = static_cast<const std::uint8_t*>(_M_storage->data_ptr()) + _M_offset;
        return static_cast<const_pointer>(ptr);
    }

    std::size_t
    size() const
    {
        return _M_storage->size() - _M_offset;
    }

    const storage_type&
    storage() const
    {
        return _M_storage;
    }

private:
    storage_type _M_storage;
    std::size_t _M_offset;
};


template <typename T> struct container_remove_type<offsetted_container_adapter<T>> {
    using type = offsetted_container_adapter<void>;
};


} // namespace metalchat
