#pragma once

#include <concepts>
#include <new>

#include <metalchat/container.h>


namespace metalchat {


class alloc_error : public std::bad_alloc {
public:
    alloc_error(const std::string& what)
    : _m_what(what)
    {}

private:
    std::string _m_what;
};


/// The concept specifies the requirements for a type to allocate elements contiguously stored
/// in the memory. The allocator is used to allocate underlying memory for a tensor.
///
/// Depending on the tensor type, memory could be allocated on stack, within random accessible
/// memory, or memory, that is shared between CPU and GPU, using different implementations of
/// hardware allocators.
template <typename Allocator>
concept allocator = requires(std::remove_reference_t<Allocator> a) {
    typename Allocator::value_type;
    typename Allocator::pointer;
    typename Allocator::const_pointer;
    typename Allocator::size_type;
    typename Allocator::container_type;
    typename Allocator::container_pointer;

    {
        a.allocate(std::declval<typename Allocator::size_type>())
    } -> std::same_as<typename Allocator::container_pointer>;

    {
        a.allocate(
            std::declval<typename Allocator::const_pointer>(),
            std::declval<typename Allocator::size_type>()
        )
    } -> std::same_as<typename Allocator::container_pointer>;
} && contiguous_container<typename Allocator::container_type>;


/// The concept specifies the requirements for a type to allocate elements of type `T`
/// contiguously stored in the memory.
template <typename Allocator, typename T>
concept allocator_t = allocator<Allocator> && std::same_as<typename Allocator::value_type, T>;


/// The concept specifies the requirements for a type to allocate elements contiguously stored
/// in the hardware (Metal) memory.
template <typename Allocator>
concept hardware_allocator = allocator<Allocator>
                             && std::same_as<
                                 typename Allocator::container_type,
                                 hardware_memory_container<typename Allocator::value_type>>;


/// The concept specifies the requirements for a type to allocate elements of a type `T`
/// conguously stored in the hardware (Metal) memory.
template <typename Allocator, typename T>
concept hardware_allocator_t
    = allocator_t<Allocator, T>
      && std::same_as<typename Allocator::container_type, hardware_memory_container<T>>;


/// This class template is a virtual class that should be inherited by allocator implementations
/// that are expected to be used within a polymorphic hardware memory allocator.
///
/// Essentially, all virtual methods presented in this class represent all necessary methods that
/// are requested by `metalchat::allocator` concept, so all allocators should automatically
/// implement this virtual class, if inherited from this struct.
///
/// Example of usage:
/// ```cpp
/// using namespace metalchat;
///
/// template <typename T> struct custom_hardware_allocator :
/// public basic_hardware_memory_allocator<T> {
///
///     using value_type = T;
///     using pointer = value_type*;
///     using const_pointer = const pointer;
///     using size_type = std::size_t;
///     using container_type = hardware_memory_container<value_type>;
///     using container_pointer = std::shared_ptr<container_type>;
///
///     container_pointer
///     allocate(size_type size)
///     {
///         // allocate a new container.
///     }
///
///     container_pointer
///     allocate(const_pointer ptr, size_type size)
///     {
///         // allocate a new container and initialize with data from ptr.
///     }
/// };
/// ```
template <typename T> struct basic_hardware_memory_allocator {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;

    /// Allocates `size * sizeof(T)` bytes of uninitialized memory by calling an outer
    /// allocator type.
    virtual container_pointer allocate(size_type) = 0;

    /// Allocates `size * sizeof(T)` bytes and initializes them with the data stored at `ptr`.
    virtual container_pointer allocate(const_pointer, size_type) = 0;

    virtual ~basic_hardware_memory_allocator() {};
};


template <typename Allocator, typename T>
concept basic_hardware_allocator_t
    = hardware_allocator_t<Allocator, T>
      && std::derived_from<Allocator, basic_hardware_memory_allocator<T>>;


/// The class template is an `metalchat::allocator` which exhibits different allocation behaviour
/// depending on a particular implementation of the `metalchat::basic_hardware_memory_allocator`.
///
/// This allocator is used in order to avoid creating separate instances of device and
/// thread, when kernel of different types (bf16, float, double) are expected to be scheduled
/// within a single device.
///
/// Example:
/// ```cpp
/// using namespace metalchat;
///
/// // Create a default hardware accelerator, then decorate the default allocator
/// // with no-copy allocator (keep all CPU allocations shared with GPU), and resident
/// // allocator (which moves all allocations to a resident set on request).
/// auto gpu = hardware_accelerator("metalchat.metallib");
/// auto alloc1 = hardware_nocopy_allocator(alloc0, gpu.get_metal_device());
/// auto alloc2 = hardware_resident_allocator(alloc1, gpu.get_metal_device());
/// auto alloc3 = polymorphic_hardware_memory_allocator(alloc2);
///
/// // Update device allocator with a new implementation of the allocator.
/// auto alloc_ptr = std::make_shared(std::move(alloc3));
/// gpu.set_allocator(alloc_ptr);
/// ```
template <typename T> class polymorphic_hardware_memory_allocator {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;
    using outer_allocator_type = basic_hardware_memory_allocator<T>;

    /// Construct a new allocator instance given an implementation of the
    /// `basic_hardware_memory_allocator`.
    polymorphic_hardware_memory_allocator(std::shared_ptr<outer_allocator_type> alloc)
    : _m_alloc(alloc)
    {}

    template <basic_hardware_allocator_t<T> Allocator>
    polymorphic_hardware_memory_allocator(Allocator&& alloc)
    : _m_alloc(std::make_shared<Allocator>(std::move(alloc)))
    {}

    /// Allocates `size * sizeof(T)` bytes of uninitialized memory by calling an outer
    /// allocator type.
    container_pointer
    allocate(size_type size)
    {
        return _m_alloc->allocate(size);
    }

    /// Allocates `size * sizeof(T)` bytes and initializes them with the data stored at `ptr`.
    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return _m_alloc->allocate(ptr, size);
    }

private:
    std::shared_ptr<outer_allocator_type> _m_alloc;
};


/// This allocator is used to cast type of elements allocated in the contiguous hardware memory,
/// that are allocated with incomplete allocator type. Allocator is incomplete, when
/// `Allocator::value_type` is equal to `void`.
///
/// The implementation only allows cast from incomplete allocator type, since the parent
/// allocator might exploit different memory alignment depending from the underlying type.
///
/// Example:
/// ```cpp
/// auto gpu = hardware_accelerator("metalchat.metallib");
/// auto alloc = rebind_hardware_allocator<float>(gpu.get_allocator());
/// auto floats_container_ptr = alloc.allocate(10);
/// ```
template <typename T, hardware_allocator_t<void> Allocator>
class rebind_hardware_allocator : public basic_hardware_memory_allocator<T> {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;

    rebind_hardware_allocator(Allocator alloc)
    : _m_alloc(alloc)
    {}

    /// Allocates `size * sizeof(T)` bytes of uninitialized memory by calling an underlying
    /// hardware allocator.
    ///
    /// Use of this function is ill-formed if `T` is incomplete type.
    container_pointer
    allocate(size_type size)
    {
        // It is totally fine to use reinterpret pointer case here, since the template
        // value type of a hardware memory container does not influence on a memory layout
        // of the container and only used to cast buffer contents to the necessary type.
        return std::reinterpret_pointer_cast<container_type>(_m_alloc.allocate(sizeof(T) * size));
    }

    /// Allocates `size * sizeof(T)` bytes and initializes them with the data stored at `ptr`.
    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return std::reinterpret_pointer_cast<container_type>(
            _m_alloc.allocate(ptr, sizeof(T) * size)
        );
    }

private:
    Allocator _m_alloc;
};


template <typename T, hardware_allocator_t<void> Allocator>
rebind_hardware_allocator<T, Allocator>
make_rebind_allocator(Allocator allocator)
{
    return rebind_hardware_allocator<T, Allocator>(allocator);
}


class _HardwareNocopyAllocator {
private:
    struct _HardwareNocopyAllocator_data;

    std::shared_ptr<_HardwareNocopyAllocator_data> _m_data;

public:
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    _HardwareNocopyAllocator(metal::shared_device device);

    container_pointer
    allocate(const void* ptr, std::size_t size);
};


/// The hardware allocator that creates a shallow buffer resource for allocations with memory-move
/// semantic. All buffers created with that method do not manage the underlying memory (specified
/// by a `const_pointer`). And caller is responsible for a proper memory management.
template <hardware_allocator_t<void> Allocator>
class hardware_nocopy_allocator
: public basic_hardware_memory_allocator<typename Allocator::value_type> {
public:
    using value_type = typename Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_nocopy_allocator(Allocator alloc, metal::shared_device device)
    : _m_alloc(alloc),
      _m_nocopy_alloc(device)
    {}

    container_pointer
    allocate(size_type size)
    {
        return _m_alloc.allocate(size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return _m_nocopy_alloc.allocate(ptr, size);
    }

private:
    Allocator _m_alloc;
    _HardwareNocopyAllocator _m_nocopy_alloc;
};


class _HardwareBufferAllocator {
private:
    metal::shared_buffer _m_buffer;

public:
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    _HardwareBufferAllocator(metal::shared_buffer buffer);

    container_pointer
    allocate(const void* ptr, std::size_t size);
};


template <hardware_allocator_t<void> Allocator>
class hardware_buffer_allocator
: public basic_hardware_memory_allocator<typename Allocator::value_type> {
public:
    using value_type = typename Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_buffer_allocator(Allocator alloc, metal::shared_buffer buffer)
    : _m_alloc(alloc),
      _m_buffer_alloc(buffer)
    {}

    container_pointer
    allocate(size_type size)
    {
        return _m_alloc.allocate(size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return _m_buffer_alloc.allocate(ptr, size);
    }

private:
    Allocator _m_alloc;
    _HardwareBufferAllocator _m_buffer_alloc;
};


class _HardwareResidentAllocator {
private:
    struct _HardwareResidentAllocator_data;

    std::shared_ptr<_HardwareResidentAllocator_data> _m_data;
    std::shared_ptr<std::size_t> _m_size;

public:
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    _HardwareResidentAllocator(metal::shared_device device, std::size_t capacity);
    ~_HardwareResidentAllocator();

    void
    detach();

    container_pointer
    allocate(container_pointer&&);

    template <typename T>
    container_pointer
    allocate(std::shared_ptr<hardware_memory_container<T>>&& p)
    {
        return allocate(std::reinterpret_pointer_cast<container_type>(p));
    }
};


/// This class template moves all allocations to the residency set. On container destruction
/// allocations are removed from the residency set. When all allocations are remove, the set
/// ends it's residency.
///
/// All containers produced by this allocator keep pointers to the residency set, so it is
/// safe to use this class within a scope.
///
/// Users could explicitly call `hardware_resident_allocator::detach`, when the underlying set is
/// supposed to be made resident. End of residency will happen automatically, once all allocations
/// are removed. Also, allocator makes all containers resident on the object destruction.
///
/// Example:
/// ```cpp
/// using namespace metalchat;
///
/// std::shared_ptr<hardware_memory_container<void>> c1;
/// std::shared_ptr<hardware_memory_container<void>> c2;
///
/// auto gpu = hardware_accelerator();
/// {
///    auto alloc0 = gpu.get_allocator();
///    auto alloc = hardware_resident_allocator(alloc0, gpu.get_metal_device());
///
///    c1 = alloc.allocate(10);
///    c2 = alloc.allocate(20);
///
///    // Scope ends, c1 and c2 become resident. This could be done explicitly
///    // by calling alloc.detach();
/// }
///
///
/// c1 = nullptr;
/// c2 = nullptr;
///
/// // Containers are deleted, end of the residency happens here.
/// ```
template <hardware_allocator Allocator>
class hardware_resident_allocator
: public basic_hardware_memory_allocator<typename Allocator::value_type> {
public:
    using value_type = typename Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_resident_allocator(
        Allocator alloc, metal::shared_device device, std::size_t capacity = 256
    )
    : _m_alloc(alloc),
      _m_resident_alloc(device, capacity)
    {}

    /// Permit allocations to be moved to resident memory and be used idependently
    /// from the given allocator.
    void
    detach()
    {
        _m_resident_alloc.detach();
    }

    container_pointer
    allocate(size_type size)
    {
        auto container = _m_resident_alloc.allocate(_m_alloc.allocate(size));
        return std::reinterpret_pointer_cast<container_type>(container);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto container = _m_resident_alloc.allocate(_m_alloc.allocate(ptr, size));
        return std::reinterpret_pointer_cast<container_type>(container);
    }

private:
    Allocator _m_alloc;
    _HardwareResidentAllocator _m_resident_alloc;
};


class _HardwareMemoryAllocator {
private:
    struct _HardwareMemoryAllocator_data;

    std::shared_ptr<_HardwareMemoryAllocator_data> _m_data;

public:
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    _HardwareMemoryAllocator(metal::shared_device device);

    container_pointer
    allocate(std::size_t size);

    container_pointer
    allocate(const void* ptr, std::size_t size);
};


/// This class creates tracked buffer resources directly from the device.
///
/// This is the default implementation of the hardware memory allocator, all resources are
/// tracked and shared with CPU. In some workloads this implementation might provide
/// suboptimal results due to frequent allocation/deallocation/wiring of the memory.
template <typename T> class hardware_memory_allocator : public basic_hardware_memory_allocator<T> {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<T>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_memory_allocator(metal::shared_device device)
    : _m_alloc(device)
    {}

    container_pointer
    allocate(size_type size)
    {
        auto container = _m_alloc.allocate(size * sizeof(value_type));
        return std::reinterpret_pointer_cast<container_type>(container);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto container = _m_alloc.allocate(ptr, size * sizeof(value_type));
        return std::reinterpret_pointer_cast<container_type>(container);
    }

private:
    _HardwareMemoryAllocator _m_alloc;
};


class _HardwareHeapAllocator {
private:
    struct _HardwareHeapAllocator_data;

    std::shared_ptr<_HardwareHeapAllocator_data> _m_data;
    std::shared_ptr<std::size_t> _m_size;

public:
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    _HardwareHeapAllocator(metal::shared_device device, std::size_t capacity);

    container_pointer
    allocate(std::size_t size);
};


/// This class creates a GPU-CPU shared memory fixed sized heap.
///
/// This allocator pre-allocates a fixed-sized contiguous shared memory and make it resident. All
/// subsequent allocations are happening within that memory and are added to the resident set.
/// Once the allocation is deleted, it also freed from the heap and from the residence set.
///
/// When there is not enough memory in the heap to allocate the requested amount of memory,
/// the implementation throws a `metalchat::alloc_error` exception.
template <typename T> class hardware_heap_allocator {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<T>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_heap_allocator(metal::shared_device device, std::size_t capacity)
    : _m_alloc(device, capacity)
    {}

    container_pointer
    allocate(size_type size)
    {
        auto container = _m_alloc.allocate(sizeof(T) * size);
        return std::reinterpret_pointer_cast<container_type>(container);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto container = _m_alloc.allocate(size);
        std::memcpy(container->data(), ptr, sizeof(T) * size);
        return std::reinterpret_pointer_cast<container_type>(container);
    }

private:
    _HardwareHeapAllocator _m_alloc;
};


/// Specialization of `metalchat::hardware_heap_allocator` type for the void type.
template <> class hardware_heap_allocator<void> : public basic_hardware_memory_allocator<void> {
public:
    using value_type = void;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_heap_allocator(metal::shared_device device, std::size_t capacity)
    : _m_alloc(device, capacity)
    {}

    container_pointer
    allocate(size_type size)
    {
        auto container = _m_alloc.allocate(size);
        return std::reinterpret_pointer_cast<container_type>(container);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto container = _m_alloc.allocate(size);
        std::memcpy(container->data(), ptr, size);
        return std::reinterpret_pointer_cast<container_type>(container);
    }

private:
    _HardwareHeapAllocator _m_alloc;
};


template <> class hardware_memory_allocator<void> : public basic_hardware_memory_allocator<void> {
public:
    using value_type = void;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_memory_allocator(metal::shared_device device)
    : _m_alloc(device)
    {}

    container_pointer
    allocate(size_type size)
    {
        return _m_alloc.allocate(size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return _m_alloc.allocate(ptr, size);
    }

private:
    _HardwareMemoryAllocator _m_alloc;
};


template <typename T> struct random_memory_allocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = random_memory_container<T>;
    using container_pointer = std::shared_ptr<container_type>;

    random_memory_allocator() {}

    container_pointer
    allocate(size_type size)
    {
        return std::make_shared<container_type>(new T[size]);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto memory_ptr = new T[size]();
        std::memcpy(memory_ptr, ptr, size);
        return std::make_shared<container_type>(memory_ptr);
    }
};


template <typename T> struct scalar_memory_allocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = scalar_memory_container<T>;
    using container_pointer = std::shared_ptr<container_type>;

    scalar_memory_allocator() {}

    container_pointer
    allocate(size_type size)
    {
        auto value = T(0);
        return allocate(&value, size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        if (size != 1) {
            throw std::invalid_argument(
                "scalar allocator allows to allocate only memory for scalar values"
            );
        }

        return std::make_shared<scalar_memory_container<T>>(*ptr);
    }
};


}; // namespace metalchat
