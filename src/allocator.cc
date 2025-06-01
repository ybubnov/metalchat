#include <format>
#include <iostream>

#include <metalchat/allocator.h>

#include "metal_impl.h"


namespace metalchat {


struct _HardwareCompleteResidenceDeleter {
    NS::SharedPtr<MTL::ResidencySet> rset;
    std::shared_ptr<std::size_t> size;

    void
    operator()(const hardware_memory_container<void>* p)
    {
        *size = (*size) - 1;

        if ((*size) == 0) {
            rset->removeAllAllocations();
            rset->endResidency();
        }
    }
};


struct _HardwareHeapAllocator::_HardwareHeapAllocator_data {
    NS::SharedPtr<MTL::Heap> heap;
    NS::SharedPtr<MTL::ResidencySet> rset;
};


_HardwareHeapAllocator::_HardwareHeapAllocator(metal::shared_device device, std::size_t capacity)
: _m_data(std::make_shared<_HardwareHeapAllocator::_HardwareHeapAllocator_data>()),
  _m_size(std::make_shared<std::size_t>(0))
{
    auto heap_options_ptr = MTL::HeapDescriptor::alloc();
    auto heap_options = NS::TransferPtr(heap_options_ptr->init());

    heap_options->setType(MTL::HeapTypeAutomatic);
    heap_options->setStorageMode(MTL::StorageModeShared);
    heap_options->setResourceOptions(MTL::ResourceStorageModeShared);
    heap_options->setSize(NS::UInteger(capacity));
    heap_options->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    heap_options->setCpuCacheMode(MTL::CPUCacheModeDefaultCache);

    _m_data->heap = NS::TransferPtr(device->ptr->newHeap(heap_options.get()));
    if (!_m_data->heap) {
        throw std::runtime_error("metalchat::hardware_heap_allocator: failed creating a new heap");
    }

    // This residency set is supposed to be used only for the heap, therefore
    // it make sense to keep the number of allocations only to 1, since we anyway
    // won't add anything apart from heap to that set.
    auto rset_options_ptr = MTL::ResidencySetDescriptor::alloc();
    auto rset_options = NS::TransferPtr(rset_options_ptr->init());
    rset_options->setInitialCapacity(1);

    NS::SharedPtr<NS::Error> error = NS::TransferPtr(NS::Error::alloc());
    NS::Error* error_ptr = error.get();

    _m_data->rset = NS::TransferPtr(device->ptr->newResidencySet(rset_options.get(), &error_ptr));
    if (!_m_data->rset) {
        auto failure_reason = error_ptr->localizedDescription();
        throw std::runtime_error(failure_reason->utf8String());
    }

    _m_data->rset->addAllocation(_m_data->heap.get());
    _m_data->rset->commit();
    _m_data->rset->requestResidency();
}


_HardwareHeapAllocator::container_pointer
_HardwareHeapAllocator::allocate(std::size_t size)
{
    auto device = _m_data->heap->device();
    auto placement = device->heapBufferSizeAndAlign(size, MTL::ResourceStorageModeShared);

    auto mask = placement.align - 1;
    auto alloc_size = ((placement.size + mask) & (~mask));

    auto memory_ptr = _m_data->heap->newBuffer(alloc_size, MTL::ResourceStorageModeShared);
    if (memory_ptr == nullptr) {
        auto cap = _m_data->heap->maxAvailableSize(placement.align);
        throw std::runtime_error(std::format(
            "metalchat::hardware_heap_allocator: failed to allocate buffer of size={}, "
            "heap remaining capacity={}",
            size, cap
        ));
    }

    *_m_size = (*_m_size) + 1;

    return std::make_shared<container_type>(
        metal::make_buffer(memory_ptr), _HardwareCompleteResidenceDeleter{_m_data->rset, _m_size}
    );
}


struct _HardwareMemoryAllocator::_HardwareMemoryAllocator_data {
    NS::SharedPtr<MTL::Device> device;

    _HardwareMemoryAllocator_data(NS::SharedPtr<MTL::Device> d)
    : device(d)
    {}
};


_HardwareMemoryAllocator::_HardwareMemoryAllocator(metal::shared_device device)
: _m_data(std::make_shared<_HardwareMemoryAllocator::_HardwareMemoryAllocator_data>(device->ptr))
{}


_HardwareMemoryAllocator::container_pointer
_HardwareMemoryAllocator::allocate(std::size_t size)
{
    auto memory_ptr = _m_data->device->newBuffer(size, MTL::ResourceStorageModeShared);
    return std::make_shared<_HardwareMemoryAllocator::container_type>(metal::make_buffer(memory_ptr)
    );
}


_HardwareMemoryAllocator::container_pointer
_HardwareMemoryAllocator::allocate(const void* ptr, std::size_t size)
{
    auto memory_ptr = _m_data->device->newBuffer(ptr, size, MTL::ResourceStorageModeShared);
    return std::make_shared<container_type>(metal::make_buffer(memory_ptr));
}


struct _HardwareNocopyAllocator::_HardwareNocopyAllocator_data {
    NS::SharedPtr<MTL::Device> device;

    _HardwareNocopyAllocator_data(NS::SharedPtr<MTL::Device> d)
    : device(d)
    {}
};


_HardwareNocopyAllocator::_HardwareNocopyAllocator(metal::shared_device device)
: _m_data(std::make_shared<_HardwareNocopyAllocator::_HardwareNocopyAllocator_data>(device->ptr))
{}


_HardwareNocopyAllocator::container_pointer
_HardwareNocopyAllocator::allocate(const void* ptr, std::size_t size)
{
    auto options = MTL::ResourceStorageModeManaged | MTL::ResourceHazardTrackingModeUntracked;
    auto memory_ptr = _m_data->device->newBuffer(ptr, size, options, nullptr);

    if (memory_ptr == nullptr) {
        throw std::runtime_error(
            std::format("metalchat::hardware_nocopy_allocator: failed to allocate no-copy buffer")
        );
    }
    return std::make_shared<container_type>(metal::make_buffer(memory_ptr));
}


struct _HardwareIterativeResidenceDeleter {
    NS::SharedPtr<MTL::ResidencySet> rset;
    std::shared_ptr<std::size_t> size;

    void
    operator()(const hardware_memory_container<void>* p)
    {
        rset->removeAllocation(p->storage()->ptr.get());
        *size = (*size) - 1;

        if ((*size) == 0) {
            rset->endResidency();
        }
    }
};


struct _HardwareResidentAllocator::_HardwareResidentAllocator_data {
    NS::SharedPtr<MTL::ResidencySet> rset;
};


_HardwareResidentAllocator::_HardwareResidentAllocator(
    metal::shared_device device, std::size_t capacity
)
: _m_data(std::make_shared<_HardwareResidentAllocator::_HardwareResidentAllocator_data>()),
  _m_size(std::make_shared<std::size_t>(0))
{
    auto rset_options_ptr = MTL::ResidencySetDescriptor::alloc();
    auto rset_options = NS::TransferPtr(rset_options_ptr->init());
    rset_options->setInitialCapacity(capacity);

    NS::SharedPtr<NS::Error> error = NS::TransferPtr(NS::Error::alloc());
    NS::Error* error_ptr = error.get();

    _m_data->rset = NS::TransferPtr(device->ptr->newResidencySet(rset_options.get(), &error_ptr));
    if (!_m_data->rset) {
        auto failure_reason = error_ptr->localizedDescription();
        throw std::runtime_error(
            std::format("metalchat::hardware_resident_allocator: {}", failure_reason->utf8String())
        );
    }
}


_HardwareResidentAllocator::~_HardwareResidentAllocator()
{
    if ((*_m_size) != 0) {
        detach();
    }
}


void
_HardwareResidentAllocator::detach()
{
    _m_data->rset->commit();
    _m_data->rset->requestResidency();
}


_HardwareResidentAllocator::container_pointer
_HardwareResidentAllocator::allocate(_HardwareResidentAllocator::container_pointer container)
{
    _m_data->rset->addAllocation(container->storage()->ptr.get());
    *_m_size = (*_m_size) + 1;

    return std::make_shared<container_type>(
        container->storage(), _HardwareIterativeResidenceDeleter{_m_data->rset, _m_size}
    );
}


} // namespace metalchat
