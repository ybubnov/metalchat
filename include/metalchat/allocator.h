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


template <typename Allocator, typename T>
concept hardware_allocator_t
    = allocator_t<Allocator, T>
      && std::same_as<typename Allocator::container_type, hardware_memory_container<T>>;


template <typename T, hardware_allocator_t<void> Allocator> class rebind_hardware_allocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;

    rebind_hardware_allocator(Allocator alloc)
    : _m_alloc(alloc)
    {}

    container_pointer
    allocate(size_type size)
    {
        // It is totally fine to use reinterpret pointer case here, since the template
        // value type of a hardware memory container does not influence on a memory layout
        // of the container and only used to cast buffer contents to the necessary type.
        return std::reinterpret_pointer_cast<container_type>(_m_alloc.allocate(sizeof(T) * size));
    }

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


template <typename T> class hardware_memory_allocator {
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
    allocate(size_type size)
    {
        auto memory_size = size * sizeof(value_type);
        auto memory_ptr
            = NS::TransferPtr(_m_device->newBuffer(memory_size, MTL::ResourceStorageModeShared));
        return std::make_shared<container_type>(memory_ptr);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto memory_size = size * sizeof(value_type);
        auto memory_ptr
            = NS::TransferPtr(_m_device->newBuffer(ptr, memory_size, MTL::ResourceStorageModeShared)
            );
        return std::make_shared<container_type>(memory_ptr);
    }

private:
    NS::SharedPtr<MTL::Device> _m_device;
};


template <typename T> class hardware_heap_allocator {};


template <> class hardware_heap_allocator<void> {
public:
    using value_type = void;
    using pointer = value_type*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<void>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_heap_allocator(NS::SharedPtr<MTL::Device> device, std::size_t capacity)
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
            throw std::runtime_error("failed creating a new heap");
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
        _m_resident = std::make_shared<bool>(false);
    }

    container_pointer
    allocate(size_type size)
    {
        if (!(*_m_resident)) {
            _m_rset->commit();
            _m_rset->requestResidency();
            *_m_resident = true;
        }
        auto memory_ptr = _m_allocate(size);
        return std::make_shared<container_type>(memory_ptr);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        // auto memory_ptr = _m_allocate(size);
        // std::memcpy(memory_ptr->contents(), ptr, size);
        // return std::make_shared<container_type>(memory_ptr);

        auto memory_ptr = NS::TransferPtr(
            _m_heap->device()->newBuffer(ptr, size, MTL::ResourceStorageModeShared, nullptr)
        );

        _m_rset->addAllocation(memory_ptr.get());
        return std::make_shared<container_type>(memory_ptr);
    }

    // TODO: implement a deleter for std::shared_ptr that will drop residency set from GPU.
    //~hardware_heap_allocator()
    //{
    //    _m_rset->endResidency();
    //    _m_rset->removeAllAllocations();
    //}

private:
    NS::SharedPtr<MTL::Heap> _m_heap;
    NS::SharedPtr<MTL::ResidencySet> _m_rset;
    std::shared_ptr<bool> _m_resident;

    NS::SharedPtr<MTL::Buffer>
    _m_allocate(size_type size)
    {
        auto placement
            = _m_heap->device()->heapBufferSizeAndAlign(size, MTL::ResourceStorageModeShared);

        size_type mask = placement.align - 1;
        size_type alloc_size = ((placement.size + mask) & (~mask));

        return NS::TransferPtr(_m_heap->newBuffer(alloc_size, MTL::ResourceStorageModeShared));
    }
};


template <> class hardware_memory_allocator<void> {
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
    allocate(size_type size)
    {
        auto memory_ptr
            = NS::TransferPtr(_m_device->newBuffer(size, MTL::ResourceStorageModeShared));
        return std::make_shared<container_type>(memory_ptr);
    }

    container_pointer
    allocate(const_pointer ptr, size_type size)
    {
        auto memory_ptr
            = NS::TransferPtr(_m_device->newBuffer(ptr, size, MTL::ResourceStorageModeShared));
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
