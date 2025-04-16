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
};


template <typename T> class hardware_memory_allocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const pointer;
    using size_type = std::size_t;
    using container_type = hardware_memory_container<T>;
    using container_pointer = std::shared_ptr<container_type>;

    hardware_memory_allocator(MTL::Device* device)
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
    MTL::Device* _m_device;
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
