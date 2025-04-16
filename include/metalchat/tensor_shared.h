#pragma once

#include <memory>

#include <metalchat/container.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_concept.h>


namespace metalchat {


template <typename T, std::size_t N, ContiguousContainer Container> class shared_tensor {
public:
    using tensor_type = tensor<T, N, Container>;

    using value_type = tensor_type::value_type;

    using pointer_type = tensor_type::pointer_type;

    using container_type = tensor_type::container_type;

    shared_tensor(const shared_tensor& t) noexcept = default;

    shared_tensor(tensor_type&& t)
    : _m_value(std::make_shared<tensor_type>(std::move(t)))
    {}

    static constexpr std::size_t
    dim()
    {
        return tensor_type::dim();
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

    container_type&
    container() const
    {
        return _m_value->container();
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
    offset(std::size_t dim)
    {
        return _m_value->offset(dim);
    }

    const std::span<std::size_t>
    offsets() const
    {
        return _m_value->offsets();
    }

    template <std::size_t M>
    shared_tensor<T, M, Container>
    view(const int (&&dims)[M]) const requires(M > 0)
    {
        return shared_tensor<T, M, Container>(_m_value->view(std::move(dims)));
    }

    shared_tensor
    narrow(std::size_t dim, std::size_t start, std::size_t length) const
    {
        return shared_tensor(_m_value->narrow(dim, start, length));
    }

    tensor_layout<N>
    layout()
    {
        return _m_value->layout();
    }

    template <indexing::size_convertible... S>
    T&
    operator[](const S&... sizes) requires(sizeof...(sizes) == N)
    {
        return _m_value->value_select(sizes...);
    }

    template <indexing::size_convertible... S>
    const T&
    operator[](const S&... sizes) const requires(sizeof...(sizes) == N)
    {
        return _m_value->value_select(sizes...);
    }

    template <indexing::slice_convertible... S>
    auto
    operator[](const S&... slices) requires(sizeof...(slices) == N)
    {
        return shared_tensor(_m_value->index_select(slices...));
    }

private:
    std::shared_ptr<tensor_type> _m_value;
};


template <typename T, std::size_t N, ContiguousContainer Container>
shared_tensor(tensor<T, N, Container>&&) -> shared_tensor<T, N, Container>;


} // namespace metalchat
