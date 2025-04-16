#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <span>
#include <sstream>
#include <type_traits>
#include <utility>

#include <metalchat/container.h>
#include <metalchat/device.h>
#include <metalchat/format.h>


namespace metalchat {


template <typename T, ContiguousContainer Container> struct tensor_traits {
    using data_type = std::unique_ptr<Container>;
    using size_type = std::unique_ptr<array_ref<std::size_t>>;

    template <typename V, template <typename U> class ref_type>
    static inline std::unique_ptr<ref_type<V>>
    move(V* values)
    {
        return std::move(std::make_unique<ref_type<V>>(values));
    }

    static inline data_type
    weak_move(T* data)
    {
        return move<T, weak_ref>(data);
    }

    static inline size_type
    weak_move(std::size_t* size)
    {
        return move<std::size_t, weak_ref>(size);
    }
};


template <typename T, std::size_t N, ContiguousContainer Container> class tensor_base {
public:
    using traits = tensor_traits<T, Container>;

    tensor_base(T* data, std::size_t* shape, std::size_t* strides)
    : m_data(traits::weak_move(data)),
      m_shape(traits::weak_move(shape)),
      m_strides(traits::weak_move(strides))
    {}

    tensor_base(traits::data_type&& data, traits::size_type&& shape, traits::size_type&& strides)
    : m_data(std::move(data)),
      m_shape(std::move(shape)),
      m_strides(std::move(strides))
    {}

    tensor_base(tensor_base<T, N, Container>&& t)
    : m_data(std::move(t.m_data)),
      m_shape(std::move(t.m_shape)),
      m_strides(std::move(t.m_strides))
    {}

    inline T*
    data_ptr()
    {
        return m_data->data();
    }

    inline const T*
    data_ptr() const
    {
        return m_data->data();
    }

    inline std::span<std::size_t>
    shape() const
    {
        return std::span(m_shape->data(), N);
    }

    inline std::size_t
    stride(std::size_t dim) const
    {
        if (dim >= N) {
            throw std::out_of_range("tensor");
        }
        return m_strides->data()[dim];
    }

    inline constexpr const std::span<std::size_t>
    strides() const noexcept
    {
        return std::span(m_strides->data(), N);
    }

    inline std::size_t
    size(std::size_t dim) const
    {
        if (dim >= N) {
            throw std::out_of_range("tensor");
        }
        return m_shape->data()[dim];
    }

    inline constexpr const std::span<std::size_t>
    sizes() const noexcept
    {
        return std::span(m_shape->data(), N);
    }

    std::size_t
    numel() const
    {
        if (N == 0) {
            return 0;
        }

        std::size_t n = 1;
        for (std::size_t i = 0; i < N; i++) {
            n *= size(i);
        }
        return n;
    }

    virtual void
    data_repr(std::ostream& os, int w) const
    {
        os << "[...]";
    }

    inline const traits::data_type&
    storage() const
    {
        return m_data;
    }

protected:
    traits::data_type m_data = nullptr;
    traits::size_type m_shape = nullptr;
    traits::size_type m_strides = nullptr;
};


template <typename T, std::size_t N, ContiguousContainer Container = weak_ref<T>>
struct tensor_format {
    const tensor_base<T, N, Container>& tensor;
    const int w;

    tensor_format(const tensor_base<T, N, Container>& tensor_, const int w_ = 0)
    : tensor(tensor_),
      w(w_)
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const tensor_format<T, N, Container>& tf)
    {
        tf.tensor.data_repr(os, tf.w);
        return os;
    }
};


template <typename T, std::size_t N, ContiguousContainer Container>
std::ostream&
operator<<(std::ostream& os, const tensor_base<T, N, Container>& t)
{
    os << tensor_format<T, N, Container>(t, 1) << ", shape=(" << t.shape() << ")";
    return os;
}


template <typename T, std::size_t N, ContiguousContainer Container = weak_ref<T>>
class tensor : public tensor_base<T, N, Container> {
public:
    using traits = tensor_traits<T, Container>;

    tensor(T* data, std::size_t* shape, std::size_t* strides)
    : tensor_base<T, N, Container>(data, shape, strides)
    {}

    tensor(traits::data_type&& data, traits::size_type&& shape, traits::size_type&& strides)
    : tensor_base<T, N, Container>(std::move(data), std::move(shape), std::move(strides))
    {}

    tensor<T, N - 1>
    at(std::size_t i)
    {
        auto new_data = this->data_ptr() + this->m_strides->data()[0] * i;
        auto new_shape = this->m_shape->data() + 1;
        auto new_strides = this->m_strides->data() + 1;
        return tensor<T, N - 1>(new_data, new_shape, new_strides);
    }

    const tensor<const T, N - 1>
    at(std::size_t i) const
    {
        auto new_data = this->data_ptr() + this->m_strides->data()[0] * i;
        auto new_shape = this->m_shape->data() + 1;
        auto new_strides = this->m_strides->data() + 1;
        return tensor<const T, N - 1>(new_data, new_shape, new_strides);
    }

    tensor<T, N - 1>
    operator[](std::size_t i)
    {
        return at(i);
    }

    template <typename = void>
        requires(N == 2)
    auto
    t() const
    {
        auto shape = new std::size_t[N];
        auto strides = new std::size_t[N];

        reverse_copy(*this->m_shape, N, shape);
        reverse_copy(*this->m_strides, N, strides);

        return tensor(
            std::move(std::make_unique<weak_ref<T>>(this->m_data->data())),
            std::move(std::make_unique<owned_ref<std::size_t>>(shape)),
            std::move(std::make_unique<owned_ref<std::size_t>>(strides))
        );
    }

    void
    data_repr(std::ostream& os, int w) const override
    {
        auto size = this->size(0);
        auto max_size = fmt::edgeitems * 2 + 1;

        os << "[";
        if (size > max_size) {
            for (std::size_t i = 0; i < fmt::edgeitems; i++) {
                os << tensor_format(at(i), w + 1) << fmt::comma(i, size);
                os << std::endl << std::setw(w) << "";
            }

            os << "..., " << std::endl << std::setw(w) << "";

            for (std::size_t i = size - fmt::edgeitems; i < size; i++) {
                os << tensor_format(at(i), w + 1) << fmt::comma(i, size);
                if (i < size - 1) {
                    os << std::endl << std::setw(w) << "";
                }
            }
        } else {
            for (std::size_t i = 0; i < size; i++) {
                os << tensor_format(at(i), w + 1) << fmt::comma(i, size);
                if (i < size - 1) {
                    os << std::endl << std::setw(w) << "";
                }
            }
        }
        os << "]";
    }
};


template <typename T, ContiguousContainer Container>
class tensor<T, 1, Container> : public tensor_base<T, 1, Container> {
public:
    using traits = tensor_traits<T, Container>;

    tensor(T* data, std::size_t* shape, std::size_t* strides)
    : tensor_base<T, 1, Container>(data, shape, strides)
    {}

    tensor(traits::data_type&& data, traits::size_type&& shape, traits::size_type&& strides)
    : tensor_base<T, 1, Container>(std::move(data), std::move(shape), std::move(strides))
    {}

    T&
    at(std::size_t i)
    {
        const auto n = this->stride(0) * i;
        if (n >= this->size(0)) {
            throw std::out_of_range("tensor");
        }
        return *(this->data_ptr() + n);
    }

    const T&
    at(std::size_t i) const
    {
        const auto n = this->stride(0) * i;
        if (n >= this->size(0)) {
            throw std::out_of_range("tensor");
        }
        return *(this->data_ptr() + n);
    }

    T&
    operator[](std::size_t i)
    {
        return at(i);
    }

    const T&
    operator[](std::size_t i) const
    {
        return at(i);
    }

    auto
    t() const
    {
        return tensor(this->m_data->data(), this->m_shape->data(), this->m_strides->data());
    }

    void
    data_repr(std::ostream& os, int w) const override
    {
        auto size = this->size(0);
        auto max_size = fmt::edgeitems * 2 + 1;

        os << "[";
        if (size > max_size) {
            for (std::size_t i = 0; i < fmt::edgeitems; i++) {
                os << at(i) << fmt::comma(i, size);
            }
            os << ", ..., ";
            for (std::size_t i = size - fmt::edgeitems; i < size; i++) {
                os << at(i) << fmt::comma(i, size);
            }
        } else {
            for (std::size_t i = 0; i < size; i++) {
                os << at(i) << fmt::comma(i, size);
            }
        }
        os << "]";
    }
};


template <typename T, ContiguousContainer Container>
class tensor<T, 0, Container> : public tensor_base<T, 0, Container> {
public:
    using traits = tensor_traits<T, Container>;

    tensor(traits::data_type&& data, traits::size_type&& shape, traits::size_type&& strides)
    : tensor_base<T, 0, Container>(std::move(data), std::move(shape), std::move(strides))
    {}

    void
    data_repr(std::ostream& os, int w) const override
    {
        os << *this->data_ptr();
    }
};


template <typename T, std::size_t N>
    requires(N > 0)
auto
empty(std::size_t (&&sizes)[N])
{
    auto shape_ = std::to_array(sizes);
    auto shape = new std::size_t[N];

    std::size_t numel = 1;
    for (auto i = 0; i < N; i++) {
        numel *= shape_[i];
        shape[i] = shape_[i];
    }

    auto data = new T[numel];
    auto strides = new std::size_t[N];

    strides[N - 1] = 1;
    for (auto i = N - 2; i < N; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    using tensor_type = tensor<T, N, owned_ref<T>>;

    return tensor_type(
        std::move(std::make_unique<owned_ref<T>>(data)),
        std::move(std::make_unique<owned_ref<std::size_t>>(shape)),
        std::move(std::make_unique<owned_ref<std::size_t>>(strides))
    );
}


template <typename T>
auto
scalar(const T& value)
{
    using tensor_type = tensor<T, 0, value_ref<T>>;

    return tensor_type(
        std::move(std::make_unique<value_ref<T>>(value)),
        std::move(std::make_unique<value_ref<std::size_t>>(0)),
        std::move(std::make_unique<value_ref<std::size_t>>(0))
    );
}


template <typename T, std::size_t N>
    requires(N > 0)
auto
empty(std::size_t (&&sizes)[N], device& device)
{
    auto shape_ = std::to_array(sizes);
    auto shape = new std::size_t[N];

    std::size_t numel = 1;
    for (auto i = 0; i < N; i++) {
        numel *= shape_[i];
        shape[i] = shape_[i];
    }

    auto data
        = NS::TransferPtr(device->newBuffer(numel * sizeof(T), MTL::ResourceStorageModeShared));

    auto strides = new std::size_t[N];

    strides[N - 1] = 1;
    for (auto i = N - 2; i < N; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    using tensor_type = tensor<T, N, device_ref<T>>;

    return tensor_type(
        std::move(std::make_unique<device_ref<T>>(data)),
        std::move(std::make_unique<owned_ref<std::size_t>>(shape)),
        std::move(std::make_unique<owned_ref<std::size_t>>(strides))
    );
}


template <typename T, std::size_t N>
    requires(N > 0)
auto
full(std::size_t (&&sizes)[N], const T& fill_value)
{
    auto t = empty<T>(std::move(sizes));
    std::fill_n(t.data_ptr(), t.numel(), fill_value);
    return t;
}


template <typename T, std::size_t N>
    requires(N > 0)
auto
full(std::size_t (&&sizes)[N], const T& fill_value, device& device)
{
    auto t = empty<T>(std::move(sizes), device);
    std::fill_n(t.data_ptr(), t.numel(), fill_value);
    return t;
}


template <typename T, std::size_t N>
    requires(N > 0)
auto
zeros(std::size_t (&&sizes)[N])
{
    return full<T>(std::move(sizes), 0);
}


} //  namespace metalchat
