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


/// A memory-mapped file abstraction for efficient file I/O operations.
///
/// This class provides a low-level interface for reading and writing to files with optional
/// memory mapping support. It maintains separate read and write position indicators similar
/// to standard C++ streams.
///
/// \note Memory mapping is explicitly controlled via \ref declare_mapped and
///       \ref undeclare_mapped methods.
///
/// Example usage:
/// ```c++
/// basic_memfile file(std::ios::in | std::ios::out);
///
/// const char* str = "this is a string";
/// file.write(data, std::strlen(str));
///
/// // Access memory-mapped data directly
/// file.declare_mapped();
/// char* ptr = file.data();
/// *ptr = 'T';
/// file.undeclare_mapped();
///
/// char buf[32] = {0};
/// file.read(buf, strlen(str));
/// ```
class basic_memfile {
public:
    using char_type = char;
    using pos_type = std::size_t;

    /// Constructs a memory file with specified path and mode.
    ///
    /// \param p The filesystem path to the file.
    /// \param mode The open mode flags (e.g., std::ios::in | std::ios::out).
    basic_memfile(const std::filesystem::path& p, std::ios::openmode mode);

    /// Constructs a read-only memory file with specified path.
    ///
    /// \param p The filesystem path to the file.
    basic_memfile(const std::filesystem::path& p);

    /// Constructs an anonymous memory file with specified mode.
    ///
    /// \param mode The open mode flags.
    basic_memfile(std::ios::openmode mode);

    /// Constructs an anonymous read-only memory file.
    basic_memfile();

    /// Checks if the file is currently memory-mapped.
    bool
    is_mapped() const noexcept;

    /// Declares the file as memory-mapped.
    ///
    /// It's safe to execute method multiple times, even when the file is already memory-mapped.
    basic_memfile&
    declare_mapped();

    /// Undeclares the file as memory-mapped.
    ///
    /// It's safe to execute method multiple times, even when the file is already unmapped.
    basic_memfile&
    undeclare_mapped();

    /// Returns the size of the file in bytes.
    std::size_t
    size() const noexcept;

    /// Returns a const pointer to the file data.
    const char_type*
    data() const noexcept;

    /// Returns a pointer to the file data.
    char_type*
    data() noexcept;

    /// Returns the current output position indicator.
    pos_type
    tellp() const noexcept;

    /// Returns the current input position indicator.
    pos_type
    tellg() const noexcept;

    /// Extracts characters from the file at the current get position.
    ///
    /// \param d Destination buffer for the read data.
    /// \param size Number of bytes to read.
    basic_memfile&
    read(char_type* d, std::size_t size);

    /// Extracts bytes from the file at the current get position.
    ///
    /// \param d Destination buffer for the read data.
    /// \param size Number of bytes to read.
    basic_memfile&
    read(void* d, std::size_t size);

    /// Inserts characters to the file at the current put position.
    ///
    /// \param s Source buffer containing the data to write.
    /// \param size Number of bytes to write.
    basic_memfile&
    write(const char_type* s, std::size_t size);

    /// Inserts bytes to the file at the current put position.
    ///
    /// \param s Source buffer containing the data to write.
    /// \param size Number of bytes to write.
    basic_memfile&
    write(const void* s, std::size_t size);

    /// Closes the file and releases associated resources.
    void
    close();

    /// Destructor that ensures proper cleanup of file resources.
    ~basic_memfile();

private:
    std::FILE* _M_file = nullptr;
    std::size_t _M_file_size = 0;
    pos_type _M_file_p = 0;
    pos_type _M_file_g = 0;
    char_type* _M_map = nullptr;
    std::ios::openmode _M_mode = std::ios::in;

    /// Checks if the file is opened in writable mode.
    bool
    writable() const;
};


/// Creates a shared pointer alias that keeps both pointers alive.
///
/// This function creates a shared pointer to T that shares ownership with both ptr1 and ptr2.
/// The resulting pointer points to the object owned by ptr1, but the reference count includes
/// both ptr1 and ptr2, ensuring both remain alive as long as the returned pointer exists.
///
/// \tparam T The type of the first pointer.
/// \tparam U The type of the second pointer.
/// \param ptr1 The primary shared pointer (determines what the result points to).
/// \param ptr2 The secondary shared pointer (kept alive along with ptr1).
///
/// Example usage:
/// ```c++
/// auto file_ptr = std::make_shared<basic_memfile>("data.bin");
/// file_ptr->declare_mapped();
///
/// auto container_ptr = std::make_shared<spanbuf>(file_ptr->data(), file_ptr->size());
///
/// // Keep file alive as long as buffer exists
/// auto alias_ptr = make_pointer_alias(container_ptr, file_ptr);
/// ```
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


/// Abstract base interface for all memory containers.
///
/// Provides type-erased access to contiguous memory regions. All concrete container types must
/// implement this interface to enable polymorphic usage.
struct basic_container {
    /// Returns the size of the container in bytes.
    virtual std::size_t
    size() const = 0;

    /// Returns a type-erased pointer to the container data.
    virtual void*
    data_ptr() = 0;

    /// Returns a type-erased const pointer to the container data.
    virtual const void*
    data_ptr() const = 0;

    /// Virtual destructor for proper cleanup.
    virtual ~basic_container() = default;
};


/// Typed memory container interface.
///
/// Extends basic_container with type-safe access to contiguous memory. All memory containers
/// in the library derive from this template.
///
/// \tparam T The value type stored in the container.
///
/// Example usage:
/// ```c++
/// class some_container : public memory_container<float> {
/// private:
///     std::vector<float> _M_data;
///
/// public:
///     pointer data()
///     {
///         return _M_data.data();
///     }
///
///     const_pointer data() const
///     {
///         return _M_data.data();
///     }
///
///     std::size_t size() const
///     {
///         return _M_data.size() * sizeof(float);
///     }
/// };
/// ```
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

    /// Dereference operator for convenient access.
    pointer
    operator*()
    {
        return data();
    }

    /// Virtual destructor for proper cleanup.
    virtual ~memory_container() = default;
};


/// Concept defining requirements for contiguous memory containers.
///
/// A container satisfies this concept if it:
/// - Derives from memory_container<T>.
/// - Defines `value_type`, `pointer`, `const_pointer`, and `storage_type`.
/// - Provides a `storage()` method returning const reference to `storage_type`.
///
/// \tparam Container The container type to check.
template <typename Container>
concept contiguous_container = requires(std::remove_reference_t<Container> const c) {
    typename Container::value_type;
    typename Container::pointer;
    typename Container::const_pointer;
    typename Container::storage_type;

    requires std::derived_from<Container, memory_container<typename Container::value_type>>;

    { c.storage() } -> std::same_as<const typename Container::storage_type&>;
};


/// Template for rebinding container types to different value types.
///
/// Specialized for each container type to enable type conversion while preserving the
/// underlying storage.
///
/// \tparam T The new value type.
/// \tparam Container The container type to rebind.
template <typename T, contiguous_container Container> struct container_rebind;


/// Template for removing the value type from a container.
///
/// Specialized for each container type to create a void-typed version.
///
/// \tparam Container The container type.
template <contiguous_container> struct container_remove_type;


/// Template for creating offset views of containers.
///
/// Specialized for each container type to enable creating sub-views at specific byte offsets.
///
/// \tparam Container The container type.
template <contiguous_container> struct container_offset;


/// This template class provides the standardized way to access various properties of
/// \ref contiguous_container. The library allocators and other components access containers
/// through this template.
///
/// \tparam Container a contiguous memory container type.
template <contiguous_container Container> struct container_traits {
    /// A type of the container.
    using container_type = Container;

    /// A shared pointer type of the container.
    using container_pointer = std::shared_ptr<container_type>;

    /// A value type of the container's underlying type.
    using value_type = container_type::value_type;

    /// A pointer type of the container's underlying type.
    using pointer = container_type::pointer;

    /// A const pointer type of the container's underlying type.
    using const_pointer = const container_type::pointer;

    /// A void pointer type.
    using void_pointer = void*;

    /// A const void pointer type.
    using const_void_pointer = const void*;

    /// Container type after type rebinding.
    template <typename T> using rebind_container = container_rebind<T, Container>::type;

    /// Container traits of the rebind container type.
    template <typename T> using rebind_traits = container_traits<rebind_container<T>>;

    /// A template method to rebind container type, when logic is present.
    template <typename T> static constexpr auto rebind = container_rebind<T, Container>::rebind;

    /// Container type after storage offset.
    using offset_container = container_offset<container_type>::type;

    /// Container traits of the offset container type.
    using offset_traits = container_traits<offset_container>;

    /// A method to shift container type, when logic is present.
    static constexpr auto offset = container_offset<Container>::offset;

    /// Returns a pointer to the beginning of the container underlying storage.
    ///
    /// \param container A contiguous memory container.
    static const_void_pointer
    begin(const container_type& container)
    {
        return static_cast<const std::uint8_t*>(container.data_ptr());
    }

    /// Returns a pointer to the beginning of the container underlying storage.
    ///
    /// \param container_ptr A pointer to the contiguous memory container.
    static const_void_pointer
    begin(const container_pointer& container_ptr)
    {
        return begin(*container_ptr);
    }

    /// Returns a pointer to the end (i.e. the element after the last element) of
    /// the container underlying storage.
    ///
    /// \param container A contiguous memory container.
    static const_void_pointer
    end(const container_type& container)
    {
        return static_cast<const std::uint8_t*>(begin(container)) + container.size();
    }

    /// Returns a pointer to the end of the container underlying storage.
    ///
    /// \param container_ptr A pointer to the contiguous memory container.
    static const_void_pointer
    end(const container_pointer& container_ptr)
    {
        return end(*container_ptr);
    }

    /// Checks whether or not a given container contains the specified pointer.
    ///
    /// \param container A contiguous memory container.
    /// \param ptr A pointer that is checked.
    static bool
    contains(const container_type& container, const_void_pointer ptr)
    {
        return (ptr >= begin(container)) && (ptr <= end(container));
    }

    /// Checks whether or not a given container contains the specified range contiguous memory.
    ///
    /// \param container A contiguous memory container.
    /// \param first A start position of the contiguous memory.
    /// \param size A size of the contiguous memory.
    static bool
    contains(const container_type& container, const_void_pointer first, std::size_t size)
    {
        const_void_pointer last = static_cast<const std::uint8_t*>(first) + size;
        return contains(container, first) && contains(container, last);
    }

    /// Checks whether or not a given container contains the specified range contiguous memory.
    ///
    /// \param container_ptr A pointer to a contiguous memory container.
    /// \param first A start position of the contiguous memory.
    /// \param size A size of the contiguous memory.
    static bool
    contains(const container_pointer& container_ptr, const_void_pointer first, std::size_t size)
    {
        return contains(*container_ptr, first, size);
    }
};


/// Container wrapping arbitrary memory allocated via shared_ptr.
///
/// This container provides a view over memory managed by a `std::shared_ptr<void>`. Supports
/// offset-based sub-views of the same underlying storage.
///
/// \tparam T The value type.
///
/// Example usage:
/// ```c++
/// // Allocate raw memory.
/// std::shared_ptr<void> storage = std::make_shared<float[]>(1024);
///
/// // Create container.
/// using Container = random_memory_container<float>;
/// auto container = std::make_shared<Container>(storage, 1024);
///
/// // Access data.
/// float* data = container->data();
///
/// // Create offset view of same storage.
/// auto offsetted_container = std::make_shared<Container>(storage, 512, 512);
/// ```
template <typename T> struct random_memory_container : public memory_container<T> {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = std::shared_ptr<void>;

    /// Constructs a container from shared storage.
    ///
    /// \param storage Shared pointer to the underlying memory.
    /// \param size Size of the accessible region in bytes.
    /// \param offset Byte offset from the storage start (default: 0).
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

    /// Returns the byte offset from the storage start.
    std::size_t
    storage_offset() const
    {
        return _M_offset;
    }

    /// Returns a void pointer to the data at the current offset.
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


/// Specialization to remove type from random_memory_container.
template <typename T> struct container_remove_type<random_memory_container<T>> {
    using type = random_memory_container<void>;
};


/// Specialization for rebinding random_memory_container<void> to a typed version.
template <typename T> struct container_rebind<T, random_memory_container<void>> {
    using type = random_memory_container<T>;
    using pointer = std::shared_ptr<type>;

    /// Rebinds a void container to a typed container.
    ///
    /// \param ptr Pointer to the void container.
    static pointer
    rebind(std::shared_ptr<random_memory_container<void>> ptr)
    {
        auto size = ptr->size();
        auto offset = ptr->storage_offset();
        auto container_ptr = std::make_shared<type>(ptr->storage(), size, offset);

        return make_pointer_alias(container_ptr, ptr);
    }
};


/// Specialization for creating offset views of random_memory_container.
template <typename T> struct container_offset<random_memory_container<T>> {
    using type = random_memory_container<T>;
    using pointer = std::shared_ptr<type>;

    /// Creates an offset view of the container.
    ///
    /// \param ptr Pointer to the original container.
    /// \param off Byte offset from the current position.
    static pointer
    offset(pointer ptr, std::size_t off)
    {
        auto size = ptr->size() - off;
        auto offset = ptr->storage_offset() + off;
        auto container_ptr = std::make_shared<type>(ptr->storage(), size, offset);

        return make_pointer_alias(container_ptr, ptr);
    }
};


/// Container backed by std::vector.
///
/// Wraps a std::vector providing container interface. The vector is moved into the container,
/// ensuring efficient memory ownership transfer.
///
/// \tparam T The value type.
///
/// Example usage:
/// ```c++
/// std::vector<int> vec = {1, 2, 3, 4, 5};
///
/// using Container = vector_memory_container<int>;
/// auto container = std::make_shared<Container>(std::move(vec));
///
/// int* data = container->data();
/// std::size_t byte_size = container->size(); // Returns sizeof(int) * 5
/// ```
template <typename T> struct vector_memory_container : public memory_container<T> {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = std::vector<T>;

    /// Constructs a container from an existing vector (moved).
    ///
    /// \param storage The vector to take ownership of.
    vector_memory_container(storage_type&& storage)
    : _M_storage(std::move(storage))
    {}

    /// Constructs an empty container.
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


/// Container backed by Metal framework buffer (GPU memory).
///
/// Wraps Metal GPU buffers for use in the container system. Supports offset-based sub-views
/// for efficient buffer slicing without copying.
///
/// \tparam T The value type.
template <typename T> struct hardware_memory_container : public memory_container<T> {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = metal::shared_buffer;

    /// Constructs a container from a Metal buffer.
    ///
    /// \param storage The Metal buffer.
    /// \param offset Byte offset from the buffer start (default: 0).
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

    /// Returns the byte offset from the buffer start.
    std::size_t
    storage_offset() const
    {
        return _M_offset;
    }

    /// Returns a void pointer to the buffer data at the current offset.
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


/// Specialization to remove type from \ref hardware_memory_container.
template <typename T> struct container_remove_type<hardware_memory_container<T>> {
    using type = hardware_memory_container<void>;
};


/// Specialization for rebinding \ref hardware_memory_container<void> to a typed version.
template <typename T> struct container_rebind<T, hardware_memory_container<void>> {
    using type = hardware_memory_container<T>;
    using pointer = std::shared_ptr<type>;

    /// Rebinds a void container to a typed container.
    ///
    /// This rebinding creates an alias to ensure the original container (and any memory-mapped
    /// files it references) remains alive. This is critical for safetensors loading where a
    /// single memory-mapped file backs multiple tensors.
    ///
    /// \param ptr Pointer to the void container.
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


/// Specialization for creating offset views of hardware_memory_container.
template <typename T> struct container_offset<hardware_memory_container<T>> {
    using type = hardware_memory_container<T>;
    using pointer = std::shared_ptr<type>;

    /// Creates an offset view of the container.
    ///
    /// \param ptr Pointer to the original container.
    /// \param offset Byte offset from the current position.
    static pointer
    offset(pointer ptr, std::size_t offset)
    {
        auto container_ptr = std::make_shared<type>(ptr->storage(), ptr->storage_offset() + offset);
        return make_pointer_alias(container_ptr, ptr);
    }
};


/// Container holding a single scalar value.
///
/// This container wraps a single value of type T, providing the container
/// interface. Useful for uniform treatment of scalar and array data.
///
/// \tparam T The value type.
///
/// Example usage:
/// ```c++
/// using Container = scalar_memory_container<double>;
/// auto container = std::make_shared<Container>(3.14159);
///
/// double* ptr = container->data();
/// std::size_t size = container->size(); // Returns sizeof(double)
/// ```
template <typename T> struct scalar_memory_container : public memory_container<T> {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = T;

    /// Constructs a container holding the given value.
    /// \param storage The value to store.
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
/// remains mapped, until \ref filebuf_memory_container::park method is called.
///
/// \tparam T The value type.
///
/// Example usage:
/// ```c++
/// std::vector<float> data = {1.0f, 2.0f, 3.0f};
///
/// using Container = filebuf_memory_container<float>;
/// auto container = std::make_shared<Container>(data.data(), data.size() * sizeof(float));
///
/// // Data is written to file, not mapped yet.
/// float* ptr = container->data(); // Maps file to memory
///
/// // Evict from memory when not needed.
/// container->park();
/// ```
template <typename T> struct filebuf_memory_container : public memory_container<T> {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = std::shared_ptr<basic_memfile>;

    /// Constructs a new instance of a file-buffered container and initializes it with
    /// the provided data. After construction file is not mapped into the memory.
    ///
    /// \param data Pointer to the source data to write to file.
    /// \param size Size of the data in bytes.
    filebuf_memory_container(const_pointer data, std::size_t size)
    : _M_storage(std::make_shared<basic_memfile>(std::ios::in | std::ios::out)),
      _M_size(size),
      _M_offset(0)
    {
        _M_storage->write(data, size);
        _M_storage->undeclare_mapped();
    }

    /// Constructs a container from existing file storage.
    ///
    /// \param storage Shared pointer to the underlying file.
    /// \param size Size of the accessible region in bytes.
    /// \param offset Byte offset from the file start.
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

    /// Maps the file into memory if not already mapped.
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

    /// Returns the byte offset from the file start.
    std::size_t
    storage_offset() const
    {
        return _M_offset;
    }

    /// Returns a void pointer to the data at the current offset.
    /// Automatically maps the file into memory if needed.
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


/// Specialization to remove type from \ref filebuf_memory_container.
template <typename T> struct container_remove_type<filebuf_memory_container<T>> {
    using type = filebuf_memory_container<void>;
};


/// Specialization for rebinding \ref filebuf_memory_container to a typed version.
template <typename T> struct container_rebind<T, filebuf_memory_container<void>> {
    using type = filebuf_memory_container<T>;
    using pointer = std::shared_ptr<type>;

    /// Rebinds a void container to a typed container.
    ///
    /// \param ptr Pointer to the void container.
    static pointer
    rebind(std::shared_ptr<filebuf_memory_container<void>> ptr)
    {
        auto size = ptr->size();
        auto offset = ptr->storage_offset();
        auto container_ptr = std::make_shared<type>(ptr->storage(), size, offset);

        return make_pointer_alias(container_ptr, ptr);
    }
};


/// Specialization for creating offset views of filebuf_memory_container.
template <typename T> struct container_offset<filebuf_memory_container<T>> {
    using type = filebuf_memory_container<T>;
    using pointer = std::shared_ptr<type>;

    /// Creates an offset view of the container.
    ///
    /// \param ptr Pointer to the original container.
    /// \param off Byte offset from the current position.
    static pointer
    offset(pointer ptr, std::size_t off)
    {
        auto size = ptr->size() - off;
        auto offset = ptr->storage_offset() + off;
        auto container_ptr = std::make_shared<type>(ptr->storage(), size, offset);

        return make_pointer_alias(container_ptr, ptr);
    }
};


/// Adapter that creates offset views of any memory container.
///
/// This adapter wraps an existing memory_container and provides an offsetted view without
/// copying data. Unlike container-specific offset implementations, this works with any
/// \ref memory_container subclass.
///
/// \tparam T The value type.
///
/// Example usage:
/// ```c++
/// using Container = vector_memory_container<int>;
/// auto container = std::make_shared<Container>(std::vector<int>{1, 2, 3, 4, 5});
///
/// // Create view starting at byte offset 8 (skipping first 2 integers)
/// using ContainerView = offsetted_container_adapter<int>;
/// auto offset_view = std::make_shared<ContainerView>(container, 8);
///
/// int* data = offset_view->data(); // Points to element [2]
/// std::size_t size = offset_view->size(); // Returns size minus 8 bytes
/// ```
template <typename T> class offsetted_container_adapter : public memory_container<T> {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using storage_type = std::shared_ptr<memory_container<T>>;

    /// Constructs an offset adapter from an existing container.
    ///
    /// \param storage Shared pointer to the base container.
    /// \param offset Byte offset from the base container's start.
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


/// Specialization to remove type from offsetted_container_adapter.
template <typename T> struct container_remove_type<offsetted_container_adapter<T>> {
    using type = offsetted_container_adapter<void>;
};


} // namespace metalchat
