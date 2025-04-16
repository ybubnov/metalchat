#pragma once

#include <concepts>

#include <metalchat/container.h>


namespace metalchat {


template <typename Allocator>
concept allocator = requires(std::remove_reference_t<Allocator> a) {
    typename Allocator::value_type;
    typename Allocator::pointer;
    typename Allocator::const_pointer;
    typename Allocator::size_type;
    typename Allocator::container_type;
    typename Allocator::container_pointer;

    {
        a.allocate(typename Allocator::size_type())
    } -> std::same_as<typename Allocator::container_pointer>;

    {
        a.allocate(typename Allocator::const_pointer(), typename Allocator::size_type())
    } -> std::same_as<typename Allocator::container_pointer>;
} && contiguous_container<typename Allocator::container_type>;


template <typename Allocator, typename T>
concept allocator_t = allocator<Allocator> && std::same_as<typename Allocator::value_type, T>;


template <typename Allocator>
concept hardware_allocator = allocator<Allocator>
                             && std::same_as<
                                 typename Allocator::container_type,
                                 hardware_memory_container<typename Allocator::value_type>>;


template <typename Allocator, typename T>
concept hardware_allocator_t
    = allocator_t<Allocator, T>
      && std::same_as<typename Allocator::container_type, hardware_memory_container<T>>;


template <typename T> struct basic_hardware_memory_allocator {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;

    virtual container_pointer allocate(size_type) = 0;
    virtual container_pointer allocate(const_pointer, size_type) = 0;
    virtual ~basic_hardware_memory_allocator() {};
};


template <typename Allocator, typename T>
concept basic_hardware_allocator_t
    = hardware_allocator_t<Allocator, T>
      && std::derived_from<Allocator, basic_hardware_memory_allocator<T>>;


/// The class template `polymorphic_hardware_memory_allocator` is an `allocator` which
/// exhibits different allocation behaviour depending on a particular implementation of
/// the `basic_hardware_memory_allocator`.
///
/// This allocator is used in order to avoid creating separate instances of device and
/// thread, when kernel of different types (bf16, float, double) are expected to be scheduled
/// within a single device.
template <typename T> class polymorphic_hardware_memory_allocator {
public:
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;
    using outer_allocator_type = basic_hardware_memory_allocator<T>;

    polymorphic_hardware_memory_allocator(std::shared_ptr<outer_allocator_type> alloc)
    : _m_alloc(alloc)
    {}

    container_pointer
    allocate(size_type size)
    {
        return _m_alloc->allocate(size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        return _m_alloc->allocate(ptr, size);
    }

private:
    std::shared_ptr<outer_allocator_type> _m_alloc;
};


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

    container_pointer
    allocate(size_type size) override
    {
        // It is totally fine to use reinterpret pointer case here, since the template
        // value type of a hardware memory container does not influence on a memory layout
        // of the container and only used to cast buffer contents to the necessary type.
        return std::reinterpret_pointer_cast<container_type>(_m_alloc.allocate(sizeof(T) * size));
    }

    container_pointer
    allocate(const_pointer ptr, size_type size) override
    {
        return std::reinterpret_pointer_cast<container_type>(
            _m_alloc.allocate(ptr, sizeof(T) * size)
        );
    }

private:
    Allocator _m_alloc;
};


template <hardware_allocator Allocator>
class hardware_nocopy_allocator
: public basic_hardware_memory_allocator<typename Allocator::value_type> {
public:
    using value_type = typename Allocator::value_type;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_nocopy_allocator(Allocator alloc, NS::SharedPtr<MTL::Device> device)
    : _m_alloc(alloc),
      _m_device(device)
    {}

    container_pointer
    allocate(size_type size) override
    {
        return _m_alloc.allocate(size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size) override
    {
        auto options = MTL::ResourceStorageModeManaged | MTL::ResourceHazardTrackingModeUntracked;
        auto memory_ptr = _m_device->newBuffer(ptr, size * sizeof(value_type), options, nullptr);

        if (memory_ptr == nullptr) {
            throw std::runtime_error(std::format(
                "metalchat::hardware_nocopy_allocator: failed to allocate no-copy buffer"
            ));
        }
        return std::make_shared<container_type>(memory_ptr);
    }

private:
    Allocator _m_alloc;
    NS::SharedPtr<MTL::Device> _m_device;
};


template <typename T> struct __hardware_iterative_residence_deleter {
    NS::SharedPtr<MTL::ResidencySet> rset;
    std::shared_ptr<std::size_t> allocations;

    void
    operator()(const hardware_memory_container<T>* p)
    {
        rset->removeAllocation(p->storage().get());
        *allocations = (*allocations) - 1;

        if ((*allocations) == 0) {
            rset->endResidency();
        }
    }
};


template <typename T> struct __hardware_complete_residence_deleter {
    NS::SharedPtr<MTL::ResidencySet> rset;
    std::shared_ptr<std::size_t> allocations;

    void
    operator()(const hardware_memory_container<T>* p)
    {
        *allocations = (*allocations) - 1;

        if ((*allocations) == 0) {
            rset->removeAllAllocations();
            rset->endResidency();
        }
    }
};


/// This class template moves all allocations to the residency set. On container destruction
/// allocations are removed from the residency set. When all allocations are remove, the set
/// ends it's residency.
///
/// All containers produced by this allocator keep pointers to the residency set, so it is
/// safe to use this class within a scope.
///
/// Users should explicitly call `wire_memory`, when the underlying set is supposed to be
/// made resident. End of residency will happen automatically, once all allocations are removed.
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
        Allocator alloc, NS::SharedPtr<MTL::Device> device, std::size_t capacity = 256
    )
    : _m_alloc(alloc),
      _m_allocations(std::make_shared<std::size_t>(0))
    {
        auto rset_options_ptr = MTL::ResidencySetDescriptor::alloc();
        auto rset_options = NS::TransferPtr(rset_options_ptr->init());
        rset_options->setInitialCapacity(capacity);

        NS::SharedPtr<NS::Error> error = NS::TransferPtr(NS::Error::alloc());
        NS::Error* error_ptr = error.get();

        _m_rset = NS::TransferPtr(device->newResidencySet(rset_options.get(), &error_ptr));
        if (!_m_rset) {
            auto failure_reason = error_ptr->localizedDescription();
            throw std::runtime_error(std::format(
                "metalchat::hardware_resident_allocator: {}", failure_reason->utf8String()
            ));
        }
    }

    /// Commits set and requests residency.
    void
    wire_memory()
    {
        _m_rset->commit();
        _m_rset->requestResidency();
    }

    container_pointer
    allocate(size_type size) override
    {
        return _m_memory_move(_m_alloc.allocate(size));
    }

    container_pointer
    allocate(const_pointer ptr, size_type size) override
    {
        return _m_memory_move(_m_alloc.allocate(ptr, size));
    }

private:
    container_pointer
    _m_memory_move(container_pointer container)
    {
        _m_rset->addAllocation(container->storage().get());
        *_m_allocations = (*_m_allocations) + 1;

        return std::make_shared<container_type>(
            container->storage(),
            __hardware_iterative_residence_deleter<value_type>{_m_rset, _m_allocations}
        );
    }

    Allocator _m_alloc;
    NS::SharedPtr<MTL::ResidencySet> _m_rset;
    std::shared_ptr<std::size_t> _m_allocations;
};


template <typename T> class hardware_memory_allocator : public basic_hardware_memory_allocator<T> {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<T>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_memory_allocator(NS::SharedPtr<MTL::Device> device)
    : _m_device(device)
    {}

    container_pointer
    allocate(size_type size) override
    {
        auto memory_size = size * sizeof(value_type);
        auto memory_ptr = _m_device->newBuffer(memory_size, MTL::ResourceStorageModeShared);

        return std::make_shared<container_type>(memory_ptr);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size) override
    {
        auto memory_size = size * sizeof(value_type);
        auto memory_ptr = _m_device->newBuffer(ptr, memory_size, MTL::ResourceStorageModeShared);

        return std::make_shared<container_type>(memory_ptr);
    }

private:
    NS::SharedPtr<MTL::Device> _m_device;
};


template <typename T> class hardware_heap_allocator {};


template <> class hardware_heap_allocator<void> : public basic_hardware_memory_allocator<void> {
public:
    using value_type = void;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_heap_allocator(NS::SharedPtr<MTL::Device> device, std::size_t capacity)
    : _m_allocations(std::make_shared<std::size_t>(0))
    {
        auto heap_options_ptr = MTL::HeapDescriptor::alloc();
        auto heap_options = NS::TransferPtr(heap_options_ptr->init());

        heap_options->setType(MTL::HeapTypeAutomatic);
        heap_options->setStorageMode(MTL::StorageModeShared);
        heap_options->setResourceOptions(MTL::ResourceStorageModeShared);
        heap_options->setSize(NS::UInteger(capacity));
        heap_options->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
        heap_options->setCpuCacheMode(MTL::CPUCacheModeDefaultCache);

        _m_heap = NS::TransferPtr(device->newHeap(heap_options.get()));
        if (!_m_heap) {
            throw std::runtime_error(
                "metalchat::hardware_heap_allocator: failed creating a new heap"
            );
        }

        // This residency set is supposed to be used only for the heap, therefore
        // it make sense to keep the number of allocations only to 1, since we anyway
        // won't add anything apart from heap to that set.
        auto rset_options_ptr = MTL::ResidencySetDescriptor::alloc();
        auto rset_options = NS::TransferPtr(rset_options_ptr->init());
        rset_options->setInitialCapacity(1);

        NS::SharedPtr<NS::Error> error = NS::TransferPtr(NS::Error::alloc());
        NS::Error* error_ptr = error.get();

        _m_rset = NS::TransferPtr(device->newResidencySet(rset_options.get(), &error_ptr));
        if (!_m_rset) {
            auto failure_reason = error_ptr->localizedDescription();
            throw std::runtime_error(failure_reason->utf8String());
        }

        _m_rset->addAllocation(_m_heap.get());
        _m_rset->commit();
        _m_rset->requestResidency();
    }

    container_pointer
    allocate(size_type size) override
    {
        return _m_allocate(size);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size) override
    {
        auto container = _m_allocate(size);
        std::memcpy(container->storage()->contents(), ptr, size);
        return container;
    }

private:
    NS::SharedPtr<MTL::Heap> _m_heap;
    NS::SharedPtr<MTL::ResidencySet> _m_rset;
    std::shared_ptr<std::size_t> _m_allocations;

    container_pointer
    _m_allocate(size_type size)
    {
        auto placement
            = _m_heap->device()->heapBufferSizeAndAlign(size, MTL::ResourceStorageModeShared);

        size_type mask = placement.align - 1;
        size_type alloc_size = ((placement.size + mask) & (~mask));

        auto memory_ptr = _m_heap->newBuffer(alloc_size, MTL::ResourceStorageModeShared);
        if (memory_ptr == nullptr) {
            auto cap = _m_heap->maxAvailableSize(placement.align);
            throw std::runtime_error(std::format(
                "metalchat::hardware_heap_allocator: failed to allocate buffer of size={}, "
                "heap remaining capacity={}",
                size, cap
            ));
        }

        *_m_allocations = (*_m_allocations) + 1;
        return std::make_shared<container_type>(
            NS::TransferPtr(memory_ptr),
            __hardware_complete_residence_deleter<value_type>{_m_rset, _m_allocations}
        );
    }
};


template <> class hardware_memory_allocator<void> : public basic_hardware_memory_allocator<void> {
public:
    using value_type = void;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_memory_allocator(NS::SharedPtr<MTL::Device> device)
    : _m_device(device)
    {}

    container_pointer
    allocate(size_type size) override
    {
        auto memory_ptr = _m_device->newBuffer(size, MTL::ResourceStorageModeShared);
        return std::make_shared<container_type>(memory_ptr);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size) override
    {
        auto memory_ptr = _m_device->newBuffer(ptr, size, MTL::ResourceStorageModeShared);
        return std::make_shared<container_type>(memory_ptr);
    }

private:
    NS::SharedPtr<MTL::Device> _m_device;
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
        auto memory_ptr = new T[size];
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
