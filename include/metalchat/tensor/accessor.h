// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <format>
#include <iterator>
#include <span>

#include <metalchat/allocator.h>
#include <metalchat/container.h>


namespace metalchat {


template <typename Accessor>
concept stride_accessor = requires(const Accessor cacc, Accessor acc) {
    { cacc.stride(std::size_t()) } -> std::same_as<std::size_t>;
    { acc.set_stride(std::size_t(), std::size_t()) } -> std::same_as<void>;
};


template <typename Accessor>
concept size_accessor = requires(const Accessor cacc, Accessor acc) {
    { cacc.size(std::size_t()) } -> std::same_as<std::size_t>;
    { acc.set_size(std::size_t(), std::size_t()) } -> std::same_as<void>;
};


template <typename Accessor>
concept offset_accessor = requires(const Accessor cacc, Accessor acc) {
    { cacc.offset(std::size_t()) } -> std::same_as<std::size_t>;
    { acc.set_offset(std::size_t(), std::size_t()) } -> std::same_as<void>;
};


template <typename Accessor>
concept accessor =
    stride_accessor<Accessor> && size_accessor<Accessor> && offset_accessor<Accessor>;


class tensor_accessor {
public:
    using value_type = std::size_t;

    using container_type = memory_container<value_type>;
    using container_pointer = std::shared_ptr<container_type>;

    tensor_accessor(const tensor_accessor&) = default;

    template <allocator_t<value_type> Allocator>
    tensor_accessor(std::size_t dim, Allocator& alloc)
    : _M_dim(dim),
      _M_sizes(alloc.allocate(dim)),
      _M_strides(alloc.allocate(dim)),
      _M_offsets(alloc.allocate(dim))
    {}

    template <allocator_t<value_type> Allocator>
    tensor_accessor(std::size_t dim, Allocator&& alloc)
    : tensor_accessor(dim, alloc)
    {}

    tensor_accessor(std::size_t dim)
    : tensor_accessor(dim, random_memory_allocator<value_type>())
    {}

    template <std::bidirectional_iterator BidirIt, allocator_t<value_type> Allocator>
    tensor_accessor(BidirIt first, BidirIt last, Allocator& alloc)
    : tensor_accessor(std::distance(first, last), alloc)
    {
        std::copy(first, last, _M_sizes->data());
        resize(first, last, *this);
    }

    template <std::size_t N>
    tensor_accessor(std::size_t (&&sizes)[N])
    : tensor_accessor(sizes, sizes + N, random_memory_allocator<value_type>())
    {}

    template <std::bidirectional_iterator BidirIt, allocator_t<value_type> Allocator>
    tensor_accessor(BidirIt first, BidirIt last, Allocator&& alloc)
    : tensor_accessor(first, last, alloc)
    {}

    // TBD.
    template <std::bidirectional_iterator BidirIt, accessor Accessor>
    static void
    resize(BidirIt first, BidirIt last, Accessor& acc)
    {
        auto dim = std::distance(first, last);
        if (dim == 0) {
            return;
        }

        --last;
        acc.set_offset(dim - 1, 0);
        acc.set_stride(dim - 1, 1);
        acc.set_size(dim - 1, *last);

        for (std::size_t i = dim - 2; i < dim; --i) {
            --last;
            acc.set_offset(i, 0);
            acc.set_size(i, *last);
            acc.set_stride(i, acc.stride(i + 1) * acc.size(i + 1));
        }
    }

    template <accessor Accessor>
    static void
    resize(const std::span<std::size_t> sizes, Accessor& acc)
    {
        resize(sizes.begin(), sizes.end(), acc);
    }

    template <accessor Accessor1, accessor Accessor2>
    static void
    resize(const Accessor1& acc1, Accessor2 acc2, std::size_t dim)
    {
        for (std::size_t i = 0; i < dim; ++i) {
            acc2.set_stride(i, acc1.stride(i));
            acc2.set_size(i, acc1.size(i));
            acc2.set_offset(i, acc1.offset(i));
        }
    }


    value_type
    size(value_type dim) const
    {
        requires_dimension(dim);
        return _M_sizes->data()[dim];
    }

    container_pointer
    sizes() const
    {
        return _M_sizes;
    }

    void
    set_size(value_type dim, value_type size)
    {
        requires_dimension(dim);
        _M_sizes->data()[dim] = size;
    }

    value_type
    stride(value_type dim) const
    {
        requires_dimension(dim);
        return _M_strides->data()[dim];
    }

    container_pointer
    strides() const
    {
        return _M_strides;
    }

    void
    set_stride(value_type dim, value_type stride)
    {
        requires_dimension(dim);
        _M_strides->data()[dim] = stride;
    }

    value_type
    offset(value_type dim) const
    {
        requires_dimension(dim);
        return _M_offsets->data()[dim];
    }

    container_pointer
    offsets() const
    {
        return _M_offsets;
    }

    void
    set_offset(value_type dim, value_type offset)
    {
        requires_dimension(dim);
        _M_offsets->data()[dim] = offset;
    }

    tensor_accessor
    squeeze() const
    {
        using container_type = offsetted_container_adapter<value_type>;

        auto offset_bytes = sizeof(value_type);
        auto sizes = std::make_shared<container_type>(_M_sizes, offset_bytes);
        auto strides = std::make_shared<container_type>(_M_strides, offset_bytes);
        auto offsets = std::make_shared<container_type>(_M_offsets, offset_bytes);

        return tensor_accessor(sizes, strides, offsets);
    }

    template <allocator_t<value_type> Allocator = random_memory_allocator<value_type>>
    tensor_accessor
    copy(Allocator allocator = Allocator()) const
    {
        tensor_accessor other(_M_dim, allocator);

        std::copy_n(_M_sizes->data(), _M_dim, other._M_sizes->data());
        std::copy_n(_M_strides->data(), _M_dim, other._M_strides->data());
        std::copy_n(_M_offsets->data(), _M_dim, other._M_offsets->data());

        return other;
    }

private:
    value_type _M_dim;
    container_pointer _M_sizes;
    container_pointer _M_strides;
    container_pointer _M_offsets;

    tensor_accessor(container_pointer sizes, container_pointer strides, container_pointer offsets)
    : _M_sizes(sizes),
      _M_strides(strides),
      _M_offsets(offsets)
    {}

    void
    requires_dimension(value_type dim) const
    {
        if ((_M_dim > 0 && dim >= _M_dim) || (_M_dim == 0 && dim > _M_dim)) {
            throw std::out_of_range(std::format(
                "tensor::requires_dimension: dim {} exceeds tensor dimensionality {}", dim, _M_dim
            ));
        }
    }
};


} // namespace metalchat
