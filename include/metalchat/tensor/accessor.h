#pragma once

#include <format>
#include <iterator>

#include <metalchat/allocator.h>
#include <metalchat/container.h>


namespace metalchat {


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

    template <std::forward_iterator ForwardIt, allocator_t<value_type> Allocator>
    tensor_accessor(ForwardIt first, ForwardIt last, Allocator& alloc)
    : tensor_accessor(std::distance(first, last), alloc)
    {
        std::copy(first, last, _M_sizes->data());
        set_stride(_M_dim - 1, 1);

        for (std::size_t i = _M_dim - 2; i < _M_dim; --i) {
            set_stride(i, stride(i + 1) * size(i + 1));
        }
    }

    template <std::forward_iterator ForwardIt, allocator_t<value_type> Allocator>
    tensor_accessor(ForwardIt first, ForwardIt last, Allocator&& alloc)
    : tensor_accessor(first, last, alloc)
    {}

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
