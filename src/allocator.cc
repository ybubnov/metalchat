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


struct _HardwareCompleteResidenceDeleter {
    NS::SharedPtr<MTL::ResidencySet> rset;
    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<std::size_t> size;

    void
    operator()(metal::buffer* p)
    {
        const std::scoped_lock __lock(*mutex);

        *size = (*size) - 1;

        if ((*size) == 0) {
            rset->removeAllAllocations();
            rset->endResidency();
        }
    }
};


struct _HardwareIterativeResidenceDeleter {
    NS::SharedPtr<MTL::ResidencySet> rset;
    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<std::size_t> size;

    void
    operator()(metal::buffer* b)
    {
        const std::scoped_lock __lock(*mutex);

        rset->removeAllocation(b->ptr);
        *size = (*size) - 1;

        if ((*size) == 0) {
            rset->endResidency();
        }
    }
};


struct _HardwareHeapAllocator::_HardwareHeapAllocator_data {
    NS::SharedPtr<MTL::Heap> heap;
    NS::SharedPtr<MTL::ResidencySet> rset;
};


_HardwareHeapAllocator::_HardwareHeapAllocator(metal::shared_device device, std::size_t capacity)
: _M_data(std::make_shared<_HardwareHeapAllocator::_HardwareHeapAllocator_data>()),
  _M_mutex(std::make_shared<std::mutex>()),
  _M_size(std::make_shared<std::size_t>(0))
{
    auto heap_options_ptr = MTL::HeapDescriptor::alloc();
    auto heap_options = NS::TransferPtr(heap_options_ptr->init());

    heap_options->setType(MTL::HeapTypeAutomatic);
    heap_options->setStorageMode(MTL::StorageModeShared);
    heap_options->setResourceOptions(MTL::ResourceStorageModeShared);
    heap_options->setSize(NS::UInteger(capacity));
    heap_options->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
    heap_options->setCpuCacheMode(MTL::CPUCacheModeDefaultCache);

    _M_data->heap = NS::TransferPtr(device->ptr->newHeap(heap_options.get()));
    if (!_M_data->heap) {
        throw std::runtime_error("metalchat::hardware_heap_allocator: failed creating a new heap");
    }

    // This residency set is supposed to be used only for the heap, therefore
    // it make sense to keep the number of allocations only to 1, since we anyway
    // won't add anything apart from heap to that set.
    auto rset_options_ptr = MTL::ResidencySetDescriptor::alloc();
    auto rset_options = NS::TransferPtr(rset_options_ptr->init());
    rset_options->setInitialCapacity(1);

    NS::SharedPtr<NS::Error> error;
    NS::Error* error_ptr = error.get();

    _M_data->rset = NS::TransferPtr(device->ptr->newResidencySet(rset_options.get(), &error_ptr));
    if (!_M_data->rset) {
        auto failure_reason = error_ptr->localizedDescription();
        throw std::runtime_error(failure_reason->utf8String());
    }

    _M_data->rset->addAllocation(_M_data->heap.get());
    _M_data->rset->commit();
    _M_data->rset->requestResidency();
}


_HardwareHeapAllocator::container_pointer
_HardwareHeapAllocator::allocate(std::size_t size)
{
    const std::scoped_lock __lock(*_M_mutex);

    auto device = _M_data->heap->device();
    auto placement = device->heapBufferSizeAndAlign(size, MTL::ResourceStorageModeShared);

    auto mask = placement.align - 1;
    auto alloc_size = ((placement.size + mask) & (~mask));

    auto memory_ptr = _M_data->heap->newBuffer(alloc_size, MTL::ResourceStorageModeShared);
    if (memory_ptr == nullptr) {
        auto cap = _M_data->heap->maxAvailableSize(placement.align);
        throw alloc_error(std::format(
            "hardware_heap_allocator: failed to allocate buffer of size={}, "
            "heap remaining capacity={}",
            size, cap
        ));
    }

    *_M_size = (*_M_size) + 1;

    auto deleter = _HardwareIterativeResidenceDeleter{_M_data->rset, _M_mutex, _M_size};
    auto buffer_ptr = metal::make_buffer(memory_ptr, deleter);

    auto ptr = std::make_shared<container_type>(buffer_ptr);
    return ptr;
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


struct _HardwareResidentAllocator::_HardwareResidentAllocator_data {
    NS::SharedPtr<MTL::ResidencySet> rset;
};


_HardwareResidentAllocator::_HardwareResidentAllocator(
    metal::shared_device device, std::size_t capacity
)
: _M_data(std::make_shared<_HardwareResidentAllocator::_HardwareResidentAllocator_data>()),
  _M_mutex(std::make_shared<std::mutex>()),
  _M_size(std::make_shared<std::size_t>(0))
{
    auto rset_options_ptr = MTL::ResidencySetDescriptor::alloc();
    auto rset_options = NS::TransferPtr(rset_options_ptr->init());
    rset_options->setInitialCapacity(capacity);

    NS::SharedPtr<NS::Error> error;
    NS::Error* error_ptr = error.get();

    _M_data->rset = NS::TransferPtr(device->ptr->newResidencySet(rset_options.get(), &error_ptr));

    if (!_M_data->rset && error_ptr) {
        std::string failure_reason("failed creating residency set");
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
    if (_M_mutex != nullptr) {
        const std::scoped_lock __lock(*_M_mutex);

        if ((*_M_size) != 0) {
            _M_data->rset->commit();
            _M_data->rset->requestResidency();
        }
    }
}


_HardwareResidentAllocator::container_pointer
_HardwareResidentAllocator::allocate(_HardwareResidentAllocator::container_pointer&& container)
{
    const std::scoped_lock __lock(*_M_mutex);

    auto buffer_ptr = container->storage();

    _M_data->rset->addAllocation(buffer_ptr->ptr);
    *_M_size = (*_M_size) + 1;

    auto deleter = _HardwareCompleteResidenceDeleter{_M_data->rset, _M_mutex, _M_size};

    auto deleter_ptr = std::get_deleter<metal::buffer_deleter>(buffer_ptr);
    deleter_ptr->invoke_before_destroy(std::move(deleter));

    return std::make_shared<container_type>(buffer_ptr);
}


} // namespace metalchat
