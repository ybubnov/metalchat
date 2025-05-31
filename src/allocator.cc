#include <format>

#include <metalchat/allocator.h>

#include "metal_impl.h"


namespace metalchat {


struct _Hardware_complete_residence_deleter {
    NS::SharedPtr<MTL::ResidencySet> rset;
    std::shared_ptr<std::size_t> alloc;

    void
    operator()(const hardware_memory_container<void>* p)
    {
        *alloc = (*alloc) - 1;

        if ((*alloc) == 0) {
            rset->removeAllAllocations();
            rset->endResidency();
        }
    }
};


struct _Hardware_heap_allocator_impl::_Hardware_heap_allocator_impl_data {
    NS::SharedPtr<MTL::Heap> heap;
    NS::SharedPtr<MTL::ResidencySet> rset;
};


_Hardware_heap_allocator_impl::_Hardware_heap_allocator_impl(
    metal::shared_device device, std::size_t capacity
)
: _m_impl(std::make_shared<_Hardware_heap_allocator_impl::_Hardware_heap_allocator_impl_data>()),
  _m_alloc(std::make_shared<std::size_t>(0))
{
    auto heap_options_ptr = MTL::HeapDescriptor::alloc();
    auto heap_options = NS::TransferPtr(heap_options_ptr->init());

    heap_options->setType(MTL::HeapTypeAutomatic);
    heap_options->setStorageMode(MTL::StorageModeShared);
    heap_options->setResourceOptions(MTL::ResourceStorageModeShared);
    heap_options->setSize(NS::UInteger(capacity));
    heap_options->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    heap_options->setCpuCacheMode(MTL::CPUCacheModeDefaultCache);

    _m_impl->heap = NS::TransferPtr(device->ptr->newHeap(heap_options.get()));
    if (!_m_impl->heap) {
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

    _m_impl->rset = NS::TransferPtr(device->ptr->newResidencySet(rset_options.get(), &error_ptr));
    if (!_m_impl->rset) {
        auto failure_reason = error_ptr->localizedDescription();
        throw std::runtime_error(failure_reason->utf8String());
    }

    _m_impl->rset->addAllocation(_m_impl->heap.get());
    _m_impl->rset->commit();
    _m_impl->rset->requestResidency();
}


_Hardware_heap_allocator_impl::container_pointer
_Hardware_heap_allocator_impl::allocate(std::size_t size)
{
    auto device = _m_impl->heap->device();
    auto placement = device->heapBufferSizeAndAlign(size, MTL::ResourceStorageModeShared);

    auto mask = placement.align - 1;
    auto alloc_size = ((placement.size + mask) & (~mask));

    auto memory_ptr = _m_impl->heap->newBuffer(alloc_size, MTL::ResourceStorageModeShared);
    if (memory_ptr == nullptr) {
        auto cap = _m_impl->heap->maxAvailableSize(placement.align);
        throw std::runtime_error(std::format(
            "metalchat::hardware_heap_allocator: failed to allocate buffer of size={}, "
            "heap remaining capacity={}",
            size, cap
        ));
    }

    *_m_alloc = (*_m_alloc) + 1;

    return std::make_shared<container_type>(
        metal::buffer(metal::buffer::impl{NS::TransferPtr(memory_ptr)}),
        _Hardware_complete_residence_deleter{_m_impl->rset, _m_alloc}
    );
}


} // namespace metalchat
