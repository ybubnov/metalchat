#pragma once

#include <metalchat/format.h>
#include <metalchat/tensor.h>
#include <metalchat/tensor_concept.h>


namespace metalchat {


class polymorphic_tensor {
public:
    using value_type = void;
    using pointer_type = value_type*;

    template <immutable_tensor Tensor>
    polymorphic_tensor(Tensor&& t)
    : _m_value(std::make_shared<Tensor>(std::move(t)))
    {}

    template <immutable_tensor Tensor>
    polymorphic_tensor(std::shared_ptr<Tensor> tensor_ptr)
    : _m_value(tensor_ptr)
    {}

    template <immutable_tensor Tensor>
    void
    emplace(Tensor&& tensor)
    {
        auto tensor_ptr = std::dynamic_pointer_cast<Tensor>(_m_value);
        if (!tensor_ptr) {
            throw std::invalid_argument(
                "polymorphic_tensor::emplace: tensor types are not compatible"
            );
        }

        *tensor_ptr = std::move(tensor);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const polymorphic_tensor& t)
    {
        os << "polymorphic_tensor(" << t._m_value->sizes() << ")" << std::endl;
        return os;
    }

    std::size_t
    dimensions() const
    {
        return _m_value->dimensions();
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

private:
    std::shared_ptr<basic_tensor> _m_value;
};


} // namespace metalchat
