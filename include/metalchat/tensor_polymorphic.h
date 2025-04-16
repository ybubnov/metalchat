#pragma once

#include <metalchat/tensor.h>
#include <metalchat/tensor_concept.h>


namespace metalchat {


class basic_tensor {
public:
    virtual std::size_t
    dim() const
        = 0;

    virtual std::size_t
    size(std::size_t) const
        = 0;

    virtual const std::span<std::size_t>
    sizes() const = 0;

    virtual std::size_t
    stride(std::size_t) const
        = 0;

    virtual const std::span<std::size_t>
    strides() const = 0;

    virtual std::size_t
    offset(std::size_t) const
        = 0;

    virtual const std::span<std::size_t>
    offsets() const = 0;

    virtual std::size_t
    numel() const
        = 0;

    virtual void*
    data_ptr()
        = 0;
};


template <immutable_tensor Tensor> class __tensor : public basic_tensor {
public:
    __tensor(Tensor&& t)
    : _m_tensor(std::make_shared<Tensor>(std::move(t)))
    {}

    inline std::size_t
    dim() const
    {
        return _m_tensor->dim();
    }

    inline std::size_t
    size(std::size_t dim) const
    {
        return _m_tensor->size(dim);
    }

    inline const std::span<std::size_t>
    sizes() const
    {
        return _m_tensor->sizes();
    }

    inline std::size_t
    stride(std::size_t dim) const
    {
        return _m_tensor->stride(dim);
    }

    inline const std::span<std::size_t>
    strides() const
    {
        return _m_tensor->strides();
    }

    inline std::size_t
    offset(std::size_t dim) const
    {
        return _m_tensor->offset(dim);
    }

    inline const std::span<std::size_t>
    offsets() const
    {
        return _m_tensor->offsets();
    }

    std::size_t
    numel() const
    {
        return _m_tensor->numel();
    }

    void*
    data_ptr()
    {
        return _m_tensor->data_ptr();
    }

private:
    std::shared_ptr<Tensor> _m_tensor;
};


class polymorphic_tensor {
public:
    using value_type = void;
    using pointer_type = value_type*;

    template <immutable_tensor Tensor>
    polymorphic_tensor(Tensor&& t)
    : _m_value(std::make_shared<__tensor<Tensor>>(std::move(t)))
    {}

    std::size_t
    dim() const
    {
        return _m_value->dim();
    }

    inline std::size_t
    size(std::size_t dim) const
    {
        return _m_value->size(dim);
    }

    inline const std::span<std::size_t>
    sizes() const
    {
        return _m_value->sizes();
    }

    inline std::size_t
    stride(std::size_t dim) const
    {
        return _m_value->stride(dim);
    }

    inline const std::span<std::size_t>
    strides() const
    {
        return _m_value->strides();
    }

    inline std::size_t
    offset(std::size_t dim) const
    {
        return _m_value->offset(dim);
    }

    inline const std::span<std::size_t>
    offsets() const
    {
        return _m_value->offsets();
    }

    std::size_t
    numel() const
    {
        return _m_value->numel();
    }

    void*
    data_ptr()
    {
        return _m_value->data_ptr();
    }

private:
    std::shared_ptr<basic_tensor> _m_value;
};


} // namespace metalchat
