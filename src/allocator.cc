// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <algorithm>
#include <format>
#include <memory>
#include <mutex>

#include <metalchat/allocator.h>

#include "metal_impl.h"


namespace metalchat {


struct _HardwareHeapAllocator::_Memory {
    NS::SharedPtr<MTL::Heap> heap;
    NS::SharedPtr<MTL::ResidencySet> rset;
    std::mutex mu;
    std::size_t size;
};


struct _HardwareHeapAllocator::_Deleter {
    std::shared_ptr<_Memory> memory;

    void
    operator()(metal::buffer* b)
    {
        auto& mem = *memory;
        const std::scoped_lock lock(mem.mu);

        mem.rset->removeAllocation(b->ptr);
        if (mem.size > 0) {
            mem.size--;
        }
        if (mem.size == 0) {
            mem.rset->endResidency();
        }
    }
};


_HardwareHeapAllocator::_HardwareHeapAllocator(metal::shared_device device, std::size_t capacity)
: _M_mem(std::make_shared<_HardwareHeapAllocator::_Memory>())
{
    auto heap_options_ptr = MTL::HeapDescriptor::alloc();
    auto heap_options = NS::TransferPtr(heap_options_ptr->init());

    heap_options->setType(MTL::HeapTypeAutomatic);
    heap_options->setStorageMode(MTL::StorageModeShared);
    heap_options->setResourceOptions(MTL::ResourceStorageModeShared);
    heap_options->setSize(NS::UInteger(capacity));
    heap_options->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    heap_options->setCpuCacheMode(MTL::CPUCacheModeDefaultCache);

    _M_mem->size = 0;
    _M_mem->heap = NS::TransferPtr(device->ptr->newHeap(heap_options.get()));
    if (!_M_mem->heap) {
        throw std::runtime_error("hardware_heap_allocator: failed creating a new heap");
    }

    // This residency set is supposed to be used only for the heap, therefore
    // it make sense to keep the number of allocations only to 1, since we anyway
    // won't add anything apart from heap to that set.
    auto rset_options_ptr = MTL::ResidencySetDescriptor::alloc();
    auto rset_options = NS::TransferPtr(rset_options_ptr->init());
    rset_options->setInitialCapacity(1);

    NS::SharedPtr<NS::Error> error;
    NS::Error* error_ptr = error.get();

    _M_mem->rset = NS::TransferPtr(device->ptr->newResidencySet(rset_options.get(), &error_ptr));
    if (!_M_mem->rset) {
        auto failure_reason = error_ptr->localizedDescription();
        throw std::runtime_error(failure_reason->utf8String());
    }

    _M_mem->rset->addAllocation(_M_mem->heap.get());
    _M_mem->rset->commit();
    _M_mem->rset->requestResidency();
}


_HardwareHeapAllocator::container_pointer
_HardwareHeapAllocator::allocate(std::size_t size)
{
    auto& mem = *_M_mem;
    const std::scoped_lock lock(mem.mu);

    auto device = _M_mem->heap->device();
    auto placement = device->heapBufferSizeAndAlign(size, MTL::ResourceStorageModeShared);

    auto mask = placement.align - 1;
    auto alloc_size = ((placement.size + mask) & (~mask));

    auto memory_ptr = _M_mem->heap->newBuffer(alloc_size, MTL::ResourceStorageModeShared);
    if (memory_ptr == nullptr) {
        auto cap = _M_mem->heap->maxAvailableSize(placement.align);
        throw alloc_error(std::format(
            "hardware_heap_allocator: failed to allocate buffer of size={}, "
            "heap remaining capacity={}",
            size, cap
        ));
    }

    mem.size++;
    auto buffer_ptr = metal::make_buffer(memory_ptr, _Deleter{_M_mem});
    return std::make_shared<container_type>(buffer_ptr);
}


struct _HardwareMemoryAllocator::_HardwareMemoryAllocator_data {
    NS::SharedPtr<MTL::Device> device;

    _HardwareMemoryAllocator_data(NS::SharedPtr<MTL::Device> d)
    : device(d)
    {}
};


_HardwareMemoryAllocator::_HardwareMemoryAllocator(metal::shared_device device)
: _M_data(std::make_shared<_HardwareMemoryAllocator::_HardwareMemoryAllocator_data>(device->ptr))
{}


_HardwareMemoryAllocator::container_pointer
_HardwareMemoryAllocator::allocate(std::size_t size)
{
    auto memory_ptr = _M_data->device->newBuffer(size, MTL::ResourceStorageModeShared);
    auto buffer_ptr = metal::make_buffer(memory_ptr);

    return std::make_shared<_HardwareMemoryAllocator::container_type>(buffer_ptr);
}


_HardwareMemoryAllocator::container_pointer
_HardwareMemoryAllocator::allocate(const void* ptr, std::size_t size)
{
    auto memory_ptr = _M_data->device->newBuffer(ptr, size, MTL::ResourceStorageModeShared);
    auto buffer_ptr = metal::make_buffer(memory_ptr);

    return std::make_shared<container_type>(buffer_ptr);
}


struct _HardwareNocopyAllocator::_HardwareNocopyAllocator_data {
    NS::SharedPtr<MTL::Device> device;

    _HardwareNocopyAllocator_data(NS::SharedPtr<MTL::Device> d)
    : device(d)
    {}
};


_HardwareNocopyAllocator::_HardwareNocopyAllocator(metal::shared_device device)
: _M_data(std::make_shared<_HardwareNocopyAllocator::_HardwareNocopyAllocator_data>(device->ptr))
{}


_HardwareNocopyAllocator::container_pointer
_HardwareNocopyAllocator::allocate(const void* ptr, std::size_t size)
{
    auto options = MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;
    auto memory_ptr = _M_data->device->newBuffer(ptr, size, options, nullptr);

    if (memory_ptr == nullptr) {
        throw alloc_error(std::format(
            "hardware_nocopy_allocator: failed to allocate no-copy buffer of size {}", size
        ));
    }

    auto buffer_ptr = metal::make_buffer(memory_ptr);
    return std::make_shared<container_type>(buffer_ptr);
}


struct _HardwareResidentAllocator::_Memory {
    NS::SharedPtr<MTL::ResidencySet> rset;
    bool committed;
    std::mutex mu;
    std::size_t size;
    std::size_t capacity;
};


/// The deleter is attached to every Metal buffer shared pointer. So once a pointer
/// owner count reaches 0, the deleter is called to end residency of the buffer set.
struct _HardwareResidentAllocator::_Deleter {
    std::shared_ptr<_Memory> memory;

    void
    operator()(metal::buffer* p)
    {
        auto& mem = *memory;
        const std::scoped_lock lock(mem.mu);

        if (mem.size > 0) {
            mem.size--;
        }
        if (mem.size == 0) {
            mem.rset->removeAllAllocations();
            mem.rset->endResidency();
        }
    }
};


_HardwareResidentAllocator::_HardwareResidentAllocator(
    metal::shared_device device, std::size_t capacity
)
: _M_mem(std::make_shared<_Memory>())
{
    auto rset_options_ptr = MTL::ResidencySetDescriptor::alloc();
    auto rset_options = NS::TransferPtr(rset_options_ptr->init());
    rset_options->setInitialCapacity(capacity);

    NS::SharedPtr<NS::Error> error;
    NS::Error* error_ptr = error.get();

    _M_mem->committed = false;
    _M_mem->size = 0;
    _M_mem->capacity = capacity;
    _M_mem->rset = NS::TransferPtr(device->ptr->newResidencySet(rset_options.get(), &error_ptr));

    if (!_M_mem->rset) {
        std::string failure_reason("failed creating residency set");
        // On simulated hardware, error pointer might be null in case of error,
        // therefore check it before accessing the localized description string.
        if (error_ptr != nullptr) {
            failure_reason = std::string(error_ptr->localizedDescription()->utf8String());
        }
        throw std::runtime_error(std::format("hardware_resident_allocator: {}", failure_reason));
    }
}


_HardwareResidentAllocator::~_HardwareResidentAllocator() { detach(); }


void
_HardwareResidentAllocator::detach()
{
    auto& mem = *_M_mem;
    const std::scoped_lock lock(mem.mu);

    if (mem.size > 0 && !mem.committed) {
        mem.rset->commit();
        mem.rset->requestResidency();
        mem.committed = true;

        // Ensure that no more allocations are allowed for a committed residency set.
        mem.capacity = mem.size;
    }
}


_HardwareResidentAllocator::container_pointer
_HardwareResidentAllocator::allocate(_HardwareResidentAllocator::container_pointer&& container)
{
    auto& mem = *_M_mem;
    const std::scoped_lock lock(mem.mu);

    if (mem.size >= mem.capacity) {
        throw alloc_error("hardware_resident_allocator: capacity exceeded");
    }

    auto buffer_ptr = container->storage();

    mem.rset->addAllocation(buffer_ptr->ptr);
    mem.size++;

    auto deleter_ptr = std::get_deleter<metal::buffer_deleter>(buffer_ptr);
    deleter_ptr->invoke_before_destroy(_Deleter{_M_mem});

    return std::make_shared<container_type>(buffer_ptr);
}


} // namespace metalchat
