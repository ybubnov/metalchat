// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <memory>

#include <metalchat/container.h>
#include <metalchat/tensor/basic.h>
#include <metalchat/tensor/concept.h>


namespace metalchat {


template <immutable_tensor Tensor> class shared_tensor_ptr {
public:
    using tensor_type = Tensor;

    using value_type = tensor_type::value_type;

    static constexpr std::size_t N = tensor_type::dim();

    using pointer_type = tensor_type::pointer_type;

    using accessor_type = tensor_accessor;

    using container_type = tensor_type::container_type;

    using container_pointer = tensor_type::container_pointer;

    using iterator = tensor_type::iterator;

    using const_iterator = tensor_type::const_iterator;

    shared_tensor_ptr(tensor_type&& t)
    : _M_value(std::make_shared<tensor_type>(std::move(t)))
    {}

    /// See \ref tensor::dim.
    static constexpr std::size_t
    dim()
    {
        return tensor_type::dim();
    }

    std::shared_ptr<tensor_type>
    get() const
    {
        return _M_value;
    }

    tensor_type&
    operator*()
    {
        return *_M_value;
    }

    const tensor_type&
    operator*() const
    {
        return *_M_value;
    }

    /// See \ref tensor::numel.
    std::size_t
    numel() const
    {
        return _M_value->numel();
    }

    const accessor_type&
    accessor() const
    {
        return _M_value->accessor();
    }

    container_type&
    container() const
    {
        return _M_value->container();
    }

    std::shared_ptr<basic_container>
    container_ptr() const
    {
        return _M_value->container_ptr();
    }

    /// See \ref tensor::data_ptr.
    pointer_type
    data_ptr()
    {
        return _M_value->data_ptr();
    }

    /// See \ref tensor::data_ptr.
    const pointer_type
    data_ptr() const
    {
        return _M_value->data_ptr();
    }

    /// See \ref tensor::size.
    std::size_t
    size(std::size_t dim) const
    {
        return _M_value->size(dim);
    }

    /// See \ref tensor::sizes.
    const std::span<std::size_t>
    sizes() const
    {
        return _M_value->sizes();
    }

    /// See \ref tensor::shape.
    const std::span<std::size_t, N>
    shape() const
    {
        return _M_value->shape();
    }

    /// See \ref tensor::stride.
    std::size_t
    stride(std::size_t dim) const
    {
        return _M_value->stride(dim);
    }

    /// See \ref tensor::strides.
    const std::span<std::size_t>
    strides() const
    {
        return _M_value->strides();
    }

    /// See \ref tensor::offset.
    std::size_t
    offset(std::size_t dim) const
    {
        return _M_value->offset(dim);
    }

    /// See \ref tensor::offsets.
    const std::span<std::size_t>
    offsets() const
    {
        return _M_value->offsets();
    }

    /// See \ref tensor::begin.
    iterator
    begin()
    {
        return _M_value->begin();
    }

    /// See \ref tensor::end.
    iterator
    end()
    {
        return _M_value->end();
    }

    /// See \ref tensor::begin.
    const_iterator
    begin() const
    {
        return _M_value->begin();
    }

    /// See \ref tensor::end.
    const_iterator
    end() const
    {
        return _M_value->end();
    }

    /// See \ref tensor::index_select.
    template <convertible_to_slice... SliceTypes>
    auto
    index_select(const SliceTypes&... slices) requires(sizeof...(slices) == N)
    {
        return shared_tensor_ptr(_M_value->index_select(slices...));
    }

    /// See \ref tensor::expand_dims.
    auto
    expand_dims(std::size_t dim) const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, N + 1>;
        return shared_tensor_ptr<tensor_t>(_M_value->expand_dims(dim));
    }

    /// See \ref tensor::view.
    template <std::size_t M>
    auto
    view(int (&&dims)[M]) const requires(M > 0)
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return shared_tensor_ptr<tensor_t>(_M_value->view(std::move(dims)));
    }

    /// See \ref tensor::view.
    template <std::size_t M>
    auto
    view(const std::span<int, M> dims) const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return shared_tensor_ptr<tensor_t>(_M_value->view(dims));
    }

    /// See \ref tensor::view.
    template <std::size_t M>
    auto
    view(const std::span<std::size_t, M> dims) const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return shared_tensor_ptr<tensor_t>(_M_value->view(dims));
    }

    /// See \ref tensor::flatten.
    template <std::size_t M>
    auto
    flatten() const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return shared_tensor_ptr<tensor_t>(_M_value->template flatten<M>());
    }

    /// See \ref tensor::transpose.
    shared_tensor_ptr
    narrow(std::size_t dim, std::size_t start, std::size_t length) const
    {
        return shared_tensor_ptr(_M_value->narrow(dim, start, length));
    }

    /// See \ref tensor::transpose.
    shared_tensor_ptr
    transpose(const std::size_t (&&dims)[N]) const
    {
        return shared_tensor_ptr(_M_value->transpose(std::move(dims)));
    }

    /// See \ref tensor::layout.
    tensor_layout<N>
    layout() const
    {
        return _M_value->layout();
    }

    template <convertible_to_index... IndexTypes>
    value_type&
    operator[](const IndexTypes&... indices) requires(sizeof...(indices) == N)
    {
        return _M_value->value_select(indices...);
    }

    template <convertible_to_index... IndexTypes>
    const value_type&
    operator[](const IndexTypes&... indices) const requires(sizeof...(indices) == N)
    {
        return _M_value->value_select(indices...);
    }

    template <convertible_to_slice... SliceTypes>
    const shared_tensor_ptr
    operator[](const SliceTypes&... slices) requires(sizeof...(slices) == N)
    {
        return shared_tensor_ptr(_M_value->index_select(slices...));
    }

    auto
    operator[](std::size_t i) requires(N > 1)
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, N - 1>;
        return shared_tensor_ptr<tensor_t>(_M_value->at(i));
    }

private:
    std::shared_ptr<tensor_type> _M_value;
};


template <typename T, std::size_t N, contiguous_container Container>
using shared_tensor = shared_tensor_ptr<tensor<T, N, Container>>;


template <typename T, std::size_t N>
using shared_hardware_tensor = shared_tensor<T, N, hardware_memory_container<T>>;


template <typename T, immutable_tensor Tensor, allocator Allocator>
auto
shared_empty_like(const Tensor& t, Allocator alloc)
{
    return shared_tensor(empty_like<T>(t, alloc));
}


template <typename T, std::size_t N, allocator Allocator>
auto
shared_empty(std::size_t (&&sizes)[N], Allocator alloc)
{
    return shared_tensor(empty<T>(std::move(sizes), alloc));
}


} // namespace metalchat
