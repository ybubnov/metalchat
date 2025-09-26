#pragma once

#include <algorithm>
#include <concepts>
#include <mutex>
#include <new>
// TODO: Move implementation to .cc file
#include <unistd.h>

#include <metalchat/container.h>


namespace metalchat {


class alloc_error : public std::bad_alloc {
public:
    alloc_error(const std::string& what)
    : _M_what(what)
    {}

    const char*
    what() const noexcept
    {
        return _M_what.c_str();
    }

private:
    std::string _M_what;
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
/// public basic_hardware_allocator<T> {
///
///     using value_type = T;
///     using pointer = value_type*;
///     using const_pointer = const value_type*;
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
///
/// \note Alternatively, you could simply use \ref hardware_allocator_wrapper in order to avoid
/// creating a custom type.
template <typename T> struct basic_hardware_allocator {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;

    /// Allocates `size * sizeof(T)` bytes of uninitialized memory by calling an outer
    /// allocator type.
    virtual container_pointer allocate(size_type) = 0;

    /// Allocates `size * sizeof(T)` bytes and initializes them with the data stored at `ptr`.
    virtual container_pointer allocate(const_pointer, size_type) = 0;

    virtual ~basic_hardware_allocator() {};
};


template <hardware_allocator Allocator>
class hardware_allocator_wrapper : public basic_hardware_allocator<typename Allocator::value_type> {
public:
    using value_type = Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = Allocator::size_type;
    using container_type = Allocator::container_type;
    using container_pointer = Allocator::container_pointer;

    hardware_allocator_wrapper(Allocator&& alloc)
    : _M_alloc(std::move(alloc))
    {}

    container_pointer
    allocate(size_type size)
    {
        return _M_alloc.allocate(size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return _M_alloc.allocate(ptr, size);
    }

private:
    Allocator _M_alloc;
};


template <hardware_allocator Allocator>
hardware_allocator_wrapper(Allocator&& alloc) -> hardware_allocator_wrapper<Allocator>;


/// The class template is an `metalchat::allocator` which exhibits different allocation behaviour
/// depending on a particular implementation of the `metalchat::basic_hardware_allocator`.
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
/// auto alloc1 = nocopy_allocator(alloc0, gpu.get_metal_device());
/// auto alloc2 = hardware_resident_allocator(alloc1, gpu.get_metal_device());
/// auto alloc3 = polymorphic_hardware_allocator(alloc2);
///
/// // Update device allocator with a new implementation of the allocator.
/// auto alloc_ptr = std::make_shared(std::move(alloc3));
/// gpu.set_allocator(alloc_ptr);
/// ```
///
/// \tparam T Scalar type of the container data.
template <typename T> class polymorphic_hardware_allocator {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;
    using outer_allocator_type = basic_hardware_allocator<T>;

    /// Construct a new allocator instance given an implementation of the
    /// `basic_hardware_allocator`.
    polymorphic_hardware_allocator(std::shared_ptr<outer_allocator_type> alloc)
    : _M_alloc(alloc)
    {}

    template <hardware_allocator_t<T> Allocator>
    polymorphic_hardware_allocator(Allocator&& alloc)
    : _M_alloc(std::make_shared<hardware_allocator_wrapper<Allocator>>(std::move(alloc)))
    {}

    /// Allocates `size * sizeof(T)` bytes of uninitialized memory by calling an outer
    /// allocator type.
    container_pointer
    allocate(size_type size)
    {
        return _M_alloc->allocate(size);
    }

    /// Allocates `size * sizeof(T)` bytes and initializes them with the data stored at `ptr`.
    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return _M_alloc->allocate(ptr, size);
    }

private:
    std::shared_ptr<outer_allocator_type> _M_alloc;
};


template <allocator Allocator> struct null_allocator {
public:
    using value_type = Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = Allocator::size_type;
    using container_type = Allocator::container_type;
    using container_pointer = Allocator::container_pointer;

    container_pointer
    allocate(size_type size)
    {
        return nullptr;
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return nullptr;
    }
};


class _HardwareNocopyAllocator {
private:
    struct _HardwareNocopyAllocator_data;

    std::shared_ptr<_HardwareNocopyAllocator_data> _M_data;

public:
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    _HardwareNocopyAllocator(metal::shared_device device);

    container_pointer
    allocate(const void* ptr, std::size_t size);
};


/// The allocator that creates a shallow container resource for allocations with memory-copy
/// semantic. All containers created with that method do not manage the underlying memory
/// (specified by a `const_pointer`). And caller is responsible for a proper memory management
/// of the original memory deallocation.
template <typename T, allocator Allocator> class nocopy_allocator;


template <hardware_allocator_t<void> Allocator> class nocopy_allocator<void, Allocator> {
public:
    using value_type = Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = Allocator::size_type;
    using container_type = Allocator::container_type;
    using container_pointer = Allocator::container_pointer;

    nocopy_allocator(const Allocator& alloc, metal::shared_device device)
    : _M_alloc(alloc),
      _M_nocopy_alloc(device)
    {}

    nocopy_allocator(Allocator&& alloc, metal::shared_device device)
    : _M_alloc(std::move(alloc)),
      _M_nocopy_alloc(device)
    {}

    container_pointer
    allocate(size_type size)
    {
        return _M_alloc.allocate(size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return _M_nocopy_alloc.allocate(ptr, size);
    }

private:
    Allocator _M_alloc;
    _HardwareNocopyAllocator _M_nocopy_alloc;
};


template <hardware_allocator_t<void> Allocator>
nocopy_allocator(Allocator alloc, metal::shared_device device) -> nocopy_allocator<void, Allocator>;


/// This class creates buffer resources with an offset from the specified buffer.
///
/// Use this class when you want to maintain a single buffer (potentially mapped to another
/// memory, like memory-mapped file). And want to allocate containers that point to the same
/// underlying buffer with a different size and offset.
///
/// When the specified pointer does not belong to the memory pools, the implementation raises
/// a `metalchat::alloc_error` exception.
template <allocator_t<void> Allocator> class pooling_allocator_adapter {
public:
    using value_type = Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = Allocator::size_type;
    using container_type = Allocator::container_type;
    using container_pointer = Allocator::container_pointer;

    /// Construct a new pooling allocator with the specified container. All allocations with
    /// "new" semantic will be proxied to the specified allocator.
    ///
    /// \param alloc Proxy allocator for allocations without backing memory.
    /// \param container_ptr Underlying container from which allocations are created.
    pooling_allocator_adapter(const Allocator& alloc, container_pointer container_ptr)
    : _M_alloc(alloc),
      _M_containers({container_ptr})
    {}

    /// Constructs a new pooling allocator with the specified container.
    ///
    /// \param alloc Proxy allocator for allocations without backing memory.
    /// \param container_ptr Underlying container from which allocations are created.
    pooling_allocator_adapter(Allocator&& alloc, container_pointer container_ptr)
    : _M_alloc(std::move(alloc)),
      _M_containers({container_ptr})
    {}

    /// Constructs a new pooling allocator with the specified sequence of containers.
    ///
    /// \param alloc Proxy allocator for allocations without backing memory.
    /// \param containers Underlying containers from which allocations are created.
    pooling_allocator_adapter(const Allocator& alloc, std::vector<container_pointer> containers)
    : _M_alloc(alloc),
      _M_containers(containers)
    {
        std::sort(_M_containers.begin(), _M_containers.end(), container_less);
    }

    /// Constructs a new pooling allocator with the specified sequence of containers.
    ///
    /// \param alloc Proxy allocator for allocations without backing memory.
    /// \param containers Underlying containers from which allocations are created.
    pooling_allocator_adapter(Allocator&& alloc, std::vector<container_pointer> containers)
    : _M_alloc(std::move(alloc)),
      _M_containers(containers)
    {
        std::sort(_M_containers.begin(), _M_containers.end(), container_less);
    }

    container_pointer
    allocate(size_type size)
    {
        return _M_alloc.allocate(size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        using const_byte_pointer = const std::uint8_t*;

        auto alloc_ptr = static_cast<const_byte_pointer>(ptr);
        auto container_it
            = std::lower_bound(_M_containers.begin(), _M_containers.end(), ptr, container_less_ptr);

        for (; container_it != _M_containers.end(); ++container_it) {
            auto& container_ptr = *container_it;
            auto container_begin = _Container_traits::begin(container_ptr);
            auto begin_ptr = static_cast<const_byte_pointer>(container_begin);

            if (_Container_traits::contains(container_ptr, ptr, size)) {
                std::size_t offset = alloc_ptr - begin_ptr;
                return _Container_traits::offset(container_ptr, offset);
            }
        }

        throw alloc_error(std::format(
            "pooling_allocator_adapter: container not found for pointer {} and size {}", ptr, size
        ));
    }

private:
    using _Container_traits = container_traits<typename Allocator::container_type>;

    static bool
    container_less(const container_pointer& a, const container_pointer b)
    {
        return a->data() < b->data();
    }

    static bool
    container_less_ptr(const container_pointer& a, const void* p)
    {
        return _Container_traits::end(a) < p;
    }

    Allocator _M_alloc;
    std::vector<container_pointer> _M_containers;
};


class _HardwareResidentAllocator {
private:
    struct _HardwareResidentAllocator_data;

    std::shared_ptr<_HardwareResidentAllocator_data> _M_data;
    std::shared_ptr<std::mutex> _M_mutex;
    std::shared_ptr<std::size_t> _M_size;

public:
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    _HardwareResidentAllocator(metal::shared_device device, std::size_t capacity);
    _HardwareResidentAllocator(_HardwareResidentAllocator&&) = default;

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
template <hardware_allocator Allocator> class hardware_resident_allocator {
public:
    using value_type = Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_resident_allocator(
        const Allocator& alloc, metal::shared_device device, std::size_t capacity = 256
    )
    : _M_alloc(alloc),
      _M_resident_alloc(device, capacity)
    {}

    hardware_resident_allocator(
        Allocator&& alloc, metal::shared_device device, std::size_t capacity = 256
    )
    : _M_alloc(std::move(alloc)),
      _M_resident_alloc(device, capacity)
    {}

    hardware_resident_allocator(hardware_resident_allocator&& other) = default;
    hardware_resident_allocator(const hardware_resident_allocator& other) = delete;

    /// Permit allocations to be moved to resident memory and be used idependently
    /// from the given allocator.
    void
    detach()
    {
        _M_resident_alloc.detach();
    }

    container_pointer
    allocate(size_type size)
    {
        auto container = _M_resident_alloc.allocate(_M_alloc.allocate(size));
        // TODO: replace with container rebind.
        return std::reinterpret_pointer_cast<container_type>(container);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto container = _M_resident_alloc.allocate(_M_alloc.allocate(ptr, size));
        // TODO: replace with container rebind.
        return std::reinterpret_pointer_cast<container_type>(container);
    }

private:
    Allocator _M_alloc;
    _HardwareResidentAllocator _M_resident_alloc;
};


class _HardwareMemoryAllocator {
private:
    struct _HardwareMemoryAllocator_data;

    std::shared_ptr<_HardwareMemoryAllocator_data> _M_data;

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
class hardware_memory_allocator {
public:
    using value_type = void;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_memory_allocator(metal::shared_device device)
    : _M_alloc(device)
    {}

    container_pointer
    allocate(size_type size)
    {
        return _M_alloc.allocate(size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return _M_alloc.allocate(ptr, size);
    }

private:
    _HardwareMemoryAllocator _M_alloc;
};


class _HardwareHeapAllocator {
private:
    struct _HardwareHeapAllocator_data;

    std::shared_ptr<_HardwareHeapAllocator_data> _M_data;
    std::shared_ptr<std::mutex> _M_mutex;
    std::shared_ptr<std::size_t> _M_size;

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
    using const_pointer = const value_type*;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<T>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_heap_allocator(metal::shared_device device, std::size_t capacity)
    : _M_alloc(device, capacity)
    {}

    container_pointer
    allocate(size_type size)
    {
        auto container = _M_alloc.allocate(sizeof(T) * size);
        // TODO: replace with container rebind
        return std::reinterpret_pointer_cast<container_type>(container);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto container = _M_alloc.allocate(size);
        std::memcpy(container->data(), ptr, sizeof(T) * size);
        // TODO: replace with container rebind
        return std::reinterpret_pointer_cast<container_type>(container);
    }

private:
    _HardwareHeapAllocator _M_alloc;
};


/// Specialization of `metalchat::hardware_heap_allocator` type for the void type.
template <> class hardware_heap_allocator<void> {
public:
    using value_type = void;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_heap_allocator(metal::shared_device device, std::size_t capacity)
    : _M_alloc(device, capacity)
    {}

    container_pointer
    allocate(size_type size)
    {
        auto container = _M_alloc.allocate(size);
        return std::reinterpret_pointer_cast<container_type>(container);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto container = _M_alloc.allocate(size);
        std::memcpy(container->data(), ptr, size);
        return std::reinterpret_pointer_cast<container_type>(container);
    }

private:
    _HardwareHeapAllocator _M_alloc;
};


template <typename T> struct random_memory_allocator {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = std::size_t;
    using container_type = random_memory_container<T>;
    using container_pointer = std::shared_ptr<container_type>;

    random_memory_allocator() {}

    container_pointer
    allocate(size_type size)
    {
        std::shared_ptr<T[]> memory_ptr(new T[size]());
        return std::make_shared<container_type>(memory_ptr, size * sizeof(T));
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto container_ptr = allocate(size);
        std::memcpy(container_ptr->data(), ptr, size);
        return container_ptr;
    }
};


template <> struct random_memory_allocator<void> {
    using value_type = void;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = std::size_t;
    using container_type = random_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    random_memory_allocator() {}

    container_pointer
    allocate(size_type size)
    {
        std::shared_ptr<std::uint8_t[]> memory_ptr(new std::uint8_t[size]());
        return std::make_shared<container_type>(memory_ptr, size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto container_ptr = allocate(size);
        std::memcpy(container_ptr->data(), ptr, size);
        return container_ptr;
    }
};


template <typename T> class nocopy_allocator<T, random_memory_allocator<T>> {
private:
    using allocator_type = random_memory_allocator<T>;

    allocator_type _M_alloc;

    struct null_deleter {
        using result_type = void;

        template <typename U>
        void
        operator()(U*) const noexcept
        {}
    };

public:
    using value_type = allocator_type::value_type;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = allocator_type::size_type;
    using container_type = allocator_type::container_type;
    using container_pointer = allocator_type::container_pointer;

    nocopy_allocator(allocator_type alloc)
    : _M_alloc(alloc)
    {}

    container_pointer
    allocate(size_type size)
    {
        return _M_alloc.allocate(size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto data_ptr = std::shared_ptr<value_type>(const_cast<pointer>(ptr), null_deleter{});
        return std::make_shared<container_type>(data_ptr, size);
    }
};


template <typename T>
nocopy_allocator(random_memory_allocator<T> alloc)
    -> nocopy_allocator<T, random_memory_allocator<T>>;


template <typename T> struct scalar_memory_allocator {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
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


/// This allocator is used to cast type of elements allocated in the contiguous memory,
/// that are allocated with incomplete allocator type. Allocator is incomplete, when
/// `Allocator::value_type` is equal to `void`.
///
/// The implementation only allows cast from incomplete allocator type, since the parent
/// allocator might exploit different memory alignment depending from the underlying type.
///
/// Example:
/// ```cpp
/// auto gpu = hardware_accelerator("metalchat.metallib");
/// auto alloc = rebind_allocator<float>(gpu.get_allocator());
/// auto floats_container_ptr = alloc.allocate(10);
/// ```
template <typename T, allocator_t<void> Allocator> struct rebind_allocator {
private:
    Allocator _M_alloc;

    using _Traits = container_traits<typename Allocator::container_type>;
    using _Container_traits = _Traits::template rebind_traits<T>;

public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = Allocator::size_type;
    using container_type = _Container_traits::container_type;
    using container_pointer = _Container_traits::container_pointer;

    rebind_allocator(const Allocator& alloc)
    : _M_alloc(alloc)
    {}

    rebind_allocator(Allocator&& alloc)
    : _M_alloc(std::move(alloc))
    {}

    static std::shared_ptr<basic_container>
    static_allocate(const void* data, size_type size, const Allocator& alloc)
    {
        using allocator_type = rebind_allocator<T, Allocator>;

        auto allocator = allocator_type(alloc);
        const auto ptr = reinterpret_cast<allocator_type::const_pointer>(data);

        return allocator.allocate(ptr, size / sizeof(T));
    }

    static std::shared_ptr<basic_container>
    static_allocate(size_type size, const Allocator& alloc)
    {
        using allocator_type = rebind_allocator<T, Allocator>;

        auto allocator = allocator_type(alloc);
        return allocator.allocate(size / sizeof(T));
    }

    /// Allocates `size * sizeof(T)` bytes of uninitialized memory by calling an underlying
    /// allocator.
    ///
    /// Use of this function is ill-formed if `T` is incomplete type.
    container_pointer
    allocate(size_type size)
    {
        return _Traits::template rebind<T>(_M_alloc.allocate(sizeof(T) * size));
    }

    /// Allocates `size * sizeof(T)` bytes and initializes them with the data stored at `ptr`.
    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        const void* alloc_ptr = static_cast<const void*>(ptr);
        return _Traits::template rebind<T>(_M_alloc.allocate(alloc_ptr, sizeof(T) * size));
    }
};


template <typename T, allocator_t<void> Allocator>
rebind_allocator<T, Allocator>
make_rebind_allocator(Allocator allocator)
{
    return rebind_allocator<T, Allocator>(allocator);
}


template <allocator Allocator> class aliasing_allocator {
public:
    using value_type = Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = Allocator::size_type;
    using container_type = Allocator::container_type;
    using container_pointer = Allocator::container_pointer;

    aliasing_allocator(const Allocator& alloc, std::shared_ptr<void> ptr)
    : _M_alloc(alloc),
      _M_ptr(ptr)
    {}

    aliasing_allocator(Allocator&& alloc, std::shared_ptr<void> ptr)
    : _M_alloc(std::move(alloc)),
      _M_ptr(ptr)
    {}

    container_pointer
    allocate(size_type size)
    {
        auto container_ptr = _M_alloc.allocate(size);
        return make_pointer_alias(container_ptr, _M_ptr);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto container_ptr = _M_alloc.allocate(ptr, size);
        return make_pointer_alias(container_ptr, _M_ptr);
    }

private:
    Allocator _M_alloc;
    std::shared_ptr<void> _M_ptr;
};


template <typename T> struct filebuf_memory_allocator {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = std::size_t;
    using container_type = filebuf_memory_container<T>;
    using container_pointer = std::shared_ptr<container_type>;

    filebuf_memory_allocator() {}

    container_pointer
    allocate(size_type size)
    {
        auto data = std::make_shared<value_type[]>(size);
        return allocate(data.get(), size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return std::make_shared<container_type>(ptr, size);
    }
};


template <allocator_t<void> Allocator> class paginated_allocator_adapter {
public:
    using value_type = Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = Allocator::size_type;
    using container_type = Allocator::container_type;
    using container_pointer = Allocator::container_pointer;

    paginated_allocator_adapter(const Allocator& alloc, size_type max_size, size_type page_size)
    : _M_alloc(alloc),
      _M_max_size(max_size),
      _M_page_size(page_size)
    {}

    paginated_allocator_adapter(Allocator&& alloc, size_type max_size, size_type page_size)
    : _M_alloc(std::move(alloc)),
      _M_max_size(max_size),
      _M_page_size(page_size)
    {}

    paginated_allocator_adapter(const Allocator& alloc, size_type max_size)
    : paginated_allocator_adapter(alloc, max_size, page_size())
    {}

    paginated_allocator_adapter(Allocator&& alloc, size_type max_size)
    : paginated_allocator_adapter(std::move(alloc), max_size, page_size())
    {}

    auto
    allocate(std::vector<size_type> sizes)
    {
        std::vector<container_pointer> containers;
        size_type block_size = 0;

        for (std::size_t i = 0; i < sizes.size(); i++) {
            if (sizes[i] > _M_max_size) {
                return std::vector<container_pointer>();
            }

            if (block_size + sizes[i] >= _M_max_size) {
                containers.push_back(_M_alloc.allocate(block_size));
                block_size = 0;
            }

            block_size += sizes[i];
        }

        containers.push_back(_M_alloc.allocate(block_size));
        return containers;
    }

    auto
    allocate(size_type size)
    {
        return allocate(std::vector({size}));
    }

    auto
    allocate(const_pointer ptr, std::vector<size_type> sizes)
    {
        std::vector<container_pointer> containers;
        size_type block_size = 0;
        const std::uint8_t* pointer = static_cast<const std::uint8_t*>(ptr);

        for (std::size_t i = 0; i < sizes.size(); i++) {
            if (sizes[i] > _M_max_size) {
                return std::vector<container_pointer>();
            }

            if (block_size + sizes[i] >= _M_max_size) {
                // TODO: align by page size?.
                containers.push_back(_M_alloc.allocate(const_pointer(pointer), block_size));
                pointer += block_size;
                block_size = 0;
            }

            block_size += sizes[i];
        }

        containers.push_back(_M_alloc.allocate(const_pointer(pointer), block_size));
        return containers;
    }

    auto
    allocate(const_pointer ptr, size_type size)
    {
        return allocate(ptr, std::vector({size}));
    }

private:
    Allocator _M_alloc;
    size_type _M_page_size;
    size_type _M_max_size;

    size_type
    page_size() const
    {
        auto page_size = sysconf(_SC_PAGESIZE);
        if (page_size == -1) {
            throw std::runtime_error("paginated_allocator_adapter: failed to query system page size"
            );
        }
        return size_type(page_size);
    }
};


}; // namespace metalchat
