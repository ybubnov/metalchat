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

    using container_type = tensor_type::container_type;

    using container_pointer = tensor_type::container_pointer;

    using iterator = tensor_type::iterator;

    using const_iterator = tensor_type::const_iterator;

    shared_tensor_ptr(tensor_type&& t)
    : _m_value(std::make_shared<tensor_type>(std::move(t)))
    {}

    static constexpr std::size_t
    dim()
    {
        return tensor_type::dim();
    }

    std::shared_ptr<tensor_type>
    get() const
    {
        return _m_value;
    }

    tensor_type&
    operator*()
    {
        return *_m_value;
    }

    const tensor_type&
    operator*() const
    {
        return *_m_value;
    }

    std::size_t
    numel() const
    {
        return _m_value->numel();
    }

    container_type&
    container() const
    {
        return _m_value->container();
    }

    container_pointer
    container_ptr() const
    {
        return _m_value->container_ptr();
    }

    pointer_type
    data_ptr()
    {
        return _m_value->data_ptr();
    }

    pointer_type
    data_ptr() const
    {
        return _m_value->data_ptr();
    }

    std::size_t
    size(std::size_t dim) const
    {
        return _m_value->size(dim);
    }

    const std::span<std::size_t>
    sizes() const
    {
        return _m_value->sizes();
    }

    const std::span<std::size_t, N>
    shape() const
    {
        return _m_value->shape();
    }

    std::size_t
    stride(std::size_t dim) const
    {
        return _m_value->stride(dim);
    }

    const std::span<std::size_t>
    strides() const
    {
        return _m_value->strides();
    }

    std::size_t
    offset(std::size_t dim) const
    {
        return _m_value->offset(dim);
    }

    const std::span<std::size_t>
    offsets() const
    {
        return _m_value->offsets();
    }

    iterator
    begin()
    {
        return _m_value->begin();
    }

    iterator
    end()
    {
        return _m_value->end();
    }

    const_iterator
    begin() const
    {
        return _m_value->begin();
    }

    const_iterator
    end() const
    {
        return _m_value->end();
    }

    template <indexing::slice_convertible... S>
    auto
    index_select(const S&... slices) requires(sizeof...(slices) == N)
    {
        return shared_tensor_ptr(_m_value->index_select(slices...));
    }

    auto
    expand_dims(std::size_t dim) const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, N + 1>;
        return shared_tensor_ptr<tensor_t>(_m_value->expand_dims(dim));
    }

    template <std::size_t M>
    auto
    view(int (&&dims)[M]) const requires(M > 0)
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return shared_tensor_ptr<tensor_t>(_m_value->view(std::move(dims)));
    }

    template <std::size_t M>
    auto
    view(const std::span<int, M> dims) const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return shared_tensor_ptr<tensor_t>(_m_value->view(dims));
    }

    template <std::size_t M>
    auto
    view(const std::span<std::size_t, M> dims) const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return shared_tensor_ptr<tensor_t>(_m_value->view(dims));
    }

    template <std::size_t M>
    auto
    flatten() const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return shared_tensor_ptr<tensor_t>(_m_value->template flatten<M>());
    }

    shared_tensor_ptr
    narrow(std::size_t dim, std::size_t start, std::size_t length) const
    {
        return shared_tensor_ptr(_m_value->narrow(dim, start, length));
    }

    shared_tensor_ptr
    transpose(const std::size_t (&&dims)[N]) const
    {
        return shared_tensor_ptr(_m_value->transpose(std::move(dims)));
    }

    tensor_layout<N>
    layout() const
    {
        return _m_value->layout();
    }

    template <indexing::size_convertible... S>
    value_type&
    operator[](const S&... sizes) requires(sizeof...(sizes) == N)
    {
        return _m_value->value_select(sizes...);
    }

    template <indexing::size_convertible... S>
    const value_type&
    operator[](const S&... sizes) const requires(sizeof...(sizes) == N)
    {
        return _m_value->value_select(sizes...);
    }

    template <indexing::slice_convertible... S>
    auto
    operator[](const S&... slices) requires(sizeof...(slices) == N)
    {
        return shared_tensor_ptr(_m_value->index_select(slices...));
    }

    auto
    operator[](std::size_t i) requires(N > 1)
    {
        using container_type = reference_memory_container<value_type>;

        using tensor0_t = tensor_type;
        using tensor1_t = change_tensor_dimensions_t<tensor0_t, N - 1>;
        using tensor2_t = change_tensor_container_t<tensor1_t, container_type>;

        return shared_tensor_ptr<tensor2_t>(_m_value->at(i));
    }

private:
    std::shared_ptr<tensor_type> _m_value;
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
