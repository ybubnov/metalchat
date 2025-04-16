#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <random>
#include <span>
#include <sstream>
#include <type_traits>
#include <utility>

#include <metalchat/container.h>
#include <metalchat/device.h>
#include <metalchat/indexing.h>
#include <metalchat/iterator.h>


namespace metalchat {


template <typename T, ContiguousContainer Container> struct tensor_traits {
    // Here is the twist with the transposition operation: if this method returns a weak
    // reference, then this object cannot outlive the original one (for example, in case
    // of a return from the function), but in case of owning reference, the object should
    // outlive the original tensor, moreover, original tensor should not wipe out the
    // memory it holds, since it could be use by another tensor.
    //
    // All this drama could be solved by forbidding returning transposed tensors, but RVO
    // (return value optimization) erases the control of what is possible to return from
    // a function.
    //
    // So exactly this function dictates the implementation of the tensor: data polymorphism
    // is implemented using a shared pointer (and so could be shared across multiple tensors),
    // while tensor info (shape, strides, offsets) are always unique reference.
    using data_type = std::shared_ptr<Container>;

    using size_type = std::unique_ptr<array_ref<std::size_t>>;
};


template <typename T, std::size_t N, ContiguousContainer Container> class tensor_base {
public:
    using traits = tensor_traits<T, Container>;

    using iterator = tensor_iterator<T, N>;

    using const_iterator = const iterator;

    tensor_base(
        const traits::data_type& data,
        traits::size_type&& shape,
        traits::size_type&& strides,
        traits::size_type&& offsets
    )
    : m_data(data),
      m_shape(std::move(shape)),
      m_strides(std::move(strides)),
      m_offsets(std::move(offsets))
    {}

    tensor_base(T* data, std::size_t* shape, std::size_t* strides, std::size_t* offsets)
    : tensor_base(
          std::make_shared<weak_ref<T>>(data),
          make_weak(shape),
          make_weak(strides),
          make_weak(offsets)
      )
    {}

    tensor_base(
        const traits::data_type& data, traits::size_type&& shape, traits::size_type&& strides
    )
    : tensor_base(data, std::move(shape), std::move(strides), make_owning(new std::size_t[N]()))
    {}

    /// Tensor move constructor
    ///
    /// The newly-created tensor contains the exact contents of the moved instance.
    /// The contents of the moved instance are a valid, but unspecified tensor.
    tensor_base(tensor_base&& t) noexcept = default;

    tensor_base(std::size_t (&&sizes)[N]) requires(std::same_as<Container, owning_ref<T>> && N > 0)
    {
        auto shape = new std::size_t[N]();
        auto strides = new std::size_t[N]();
        auto offsets = new std::size_t[N]();

        std::size_t numel = 1;
        for (auto i = 0; i < N; i++) {
            numel *= sizes[i];
            shape[i] = sizes[i];
        }

        auto data = new T[numel];
        strides[N - 1] = 1;
        for (auto i = N - 2; i < N; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        m_data = std::make_shared<owning_ref<T>>(data);
        m_shape = make_owning(shape);
        m_strides = make_owning(strides);
        m_offsets = make_owning(offsets);
    }

    inline std::size_t
    dim() const noexcept
    {
        return N;
    }

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

    inline auto
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

    inline const auto
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

    inline const auto
    sizes() const noexcept
    {
        return std::span(m_shape->data(), N);
    }

    inline std::size_t
    offset(std::size_t dim) const
    {
        if (dim >= N) {
            throw std::out_of_range("tensor");
        }
        return m_offsets->data()[dim];
    }

    inline const auto
    offsets() const noexcept
    {
        return std::span(m_offsets->data(), N);
    }

    bool
    is_contiguous() const
    {
        for (size_t i = 0; i < N; i++) {
            if (m_offsets->data()[i] != 0) {
                return false;
            }
        }
        return true;
    }

    std::size_t
    numel() const
    {
        std::size_t n = 1;
        for (std::size_t i = 0; i < N; i++) {
            n *= size(i);
        }
        return n;
    }

    inline traits::data_type::element_type&
    container() const
    {
        return *m_data;
    }

    iterator
    begin()
    {
        return iterator(*m_data, *m_shape, *m_strides, *m_offsets);
    }

    const_iterator
    begin() const
    {
        return const_iterator(*m_data, *m_shape, *m_strides, *m_offsets);
    }

    iterator
    end()
    {
        return iterator(*m_data, *m_shape, *m_strides, *m_offsets, numel());
    }

    const_iterator
    end() const
    {
        return const_iterator(*m_data, *m_shape, *m_strides, *m_offsets, numel());
    }

    template <indexing::SliceConvertible... S>
    auto
    operator[](const S&... slices) requires(sizeof...(slices) <= N)
    {
        auto shape = new std::size_t[N]();
        auto offsets = new std::size_t[N]();
        std::size_t i = 0;

        (
            [&] {
                indexing::slice slice(slices);
                auto stop = std::min(slice.stop.value_or(size(i)), size(i));
                auto start = std::min(slice.start.value_or(0), stop);

                shape[i] = stop - start;
                offsets[i] = start;
                i++;
            }(),
            ...
        );

        return tensor_base<T, N, weak_ref<T>>(
            make_weak(data_ptr()), make_owning(shape), make_weak(m_strides->data()),
            make_owning(offsets)
        );
    }

    /// Returns a tensor with dimensions transposed.
    template <indexing::SizeConvertible... Dimensions>
    auto
    transpose(const Dimensions... dims) requires(sizeof...(dims) == N)
    {
        auto shape = new std::size_t[N]{size(static_cast<std::size_t>(dims))...};
        auto strides = new std::size_t[N]{stride(static_cast<std::size_t>(dims))...};
        auto offsets = new std::size_t[N]{offset(static_cast<std::size_t>(dims))...};

        return tensor_base(m_data, make_owning(shape), make_owning(strides), make_owning(offsets));
    }

protected:
    traits::data_type m_data = nullptr;
    traits::size_type m_shape = nullptr;
    traits::size_type m_strides = nullptr;
    traits::size_type m_offsets = nullptr;
};


template <typename T, std::size_t N, ContiguousContainer Container = weak_ref<T>>
class tensor : public tensor_base<T, N, Container> {
private:
    using _Base = tensor_base<T, N, Container>;

public:
    using traits = tensor_traits<T, Container>;

    tensor(_Base&& t)
    : _Base(std::move(t))
    {}

    tensor(T* data, std::size_t* shape, std::size_t* strides, std::size_t* offsets)
    : _Base(data, shape, strides, offsets)
    {}

    tensor(const traits::data_type& data, traits::size_type&& shape, traits::size_type&& strides)
    : _Base(data, std::move(shape), std::move(strides))
    {}

    tensor(
        const traits::data_type& data,
        traits::size_type&& shape,
        traits::size_type&& strides,
        traits::size_type&& offsets
    )
    : _Base(data, std::move(shape), std::move(strides), std::move(offsets))
    {}

    auto
    at(std::size_t i)
    {
        auto new_data = this->data_ptr() + this->stride(0) * (this->offset(0) + i);
        auto new_shape = this->m_shape->data() + 1;
        auto new_strides = this->m_strides->data() + 1;
        auto new_offsets = this->m_offsets->data() + 1;
        return tensor<T, N - 1, weak_ref<T>>(new_data, new_shape, new_strides, new_offsets);
    }

    const auto
    at(std::size_t i) const
    {
        auto new_data = this->data_ptr() + this->stride(0) * (this->offset(0) + i);
        auto new_shape = this->m_shape->data() + 1;
        auto new_strides = this->m_strides->data() + 1;
        auto new_offsets = this->m_offsets->data() + 1;
        return tensor<const T, N - 1, weak_ref<const T>>(
            new_data, new_shape, new_strides, new_offsets
        );
    }

    tensor<T, N - 1>
    operator[](std::size_t i)
    {
        return this->at(i);
    }

    template <indexing::SliceConvertible... S>
    auto
    operator[](const S&... slices)
    {
        return tensor<T, N, weak_ref<T>>(_Base::operator[](slices...));
    }

    template <ContiguousContainer OtherContainer>
    tensor&
    operator=(const tensor<T, N, OtherContainer>& other)
    {
        for (std::size_t i = 0; i < N; i++) {
            assert(other.size(i) == this->size(i));
        }

        std::copy(other.begin(), other.end(), this->begin());
        return *this;
    }

    template <indexing::SizeConvertible... Dimensions>
    auto
    transpose(const Dimensions... dims)
    {
        return tensor(_Base::transpose(dims...));
    }

    auto
    t() requires(N == 2)
    {
        return transpose(0, 1);
    }
};


template <typename T, ContiguousContainer Container>
class tensor<T, 1, Container> : public tensor_base<T, 1, Container> {
private:
    using _Base = tensor_base<T, 1, Container>;

public:
    using traits = tensor_traits<T, Container>;

    tensor(_Base&& t)
    : _Base(std::move(t))
    {}

    tensor(T* data, std::size_t* shape, std::size_t* strides, std::size_t* offsets)
    : _Base(data, shape, strides, offsets)
    {}

    tensor(const traits::data_type& data, traits::size_type&& shape, traits::size_type&& strides)
    : _Base(std::move(data), std::move(shape), std::move(strides))
    {}

    tensor(
        const traits::data_type& data,
        traits::size_type&& shape,
        traits::size_type&& strides,
        traits::size_type&& offsets
    )
    : _Base(data, std::move(shape), std::move(strides), std::move(offsets))
    {}

    T&
    at(std::size_t i)
    {
        if (i >= this->size(0)) {
            throw std::out_of_range("tensor");
        }
        const auto n = this->stride(0) * (i + this->offset(0));
        return *(this->data_ptr() + n);
    }

    const T&
    at(std::size_t i) const
    {
        if (i >= this->size(0)) {
            throw std::out_of_range("tensor");
        }
        const auto n = this->stride(0) * (i + this->offset(0));
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

    template <indexing::SliceConvertible... S>
    auto
    operator[](const S&... slices)
    {
        return tensor<T, 1, weak_ref<T>>(_Base::operator[](slices...));
    }

    auto
    t() const
    {
        return tensor(this->m_data->data(), this->m_shape->data(), this->m_strides->data());
    }
};


template <typename T, ContiguousContainer Container>
class tensor<T, 0, Container> : public tensor_base<T, 0, Container> {
private:
    using _Base = tensor_base<T, 0, Container>;

public:
    using traits = tensor_traits<T, Container>;

    tensor(tensor_base<T, 0, Container>&& t)
    : _Base(std::move(t))
    {}

    tensor(const traits::data_type& data, traits::size_type&& shape, traits::size_type&& strides)
    : _Base(data, std::move(shape), std::move(strides))
    {}

    tensor(
        const traits::data_type& data,
        traits::size_type&& shape,
        traits::size_type&& strides,
        traits::size_type&& offsets
    )
    : _Base(data, std::move(shape), std::move(strides), std::move(offsets))
    {}

    tensor(const T& value)
    : _Base(
          make_value(value),
          make_value<std::size_t>(0),
          make_value<std::size_t>(0),
          make_value<std::size_t>(0)
      )
    {}
};


template <typename T, std::size_t N> requires(N > 0)
auto
empty(std::size_t (&&sizes)[N])
{
    return tensor<T, N, owning_ref<T>>(std::move(sizes));
}


template <typename T>
auto
scalar(const T& value)
{
    using tensor_type = tensor<T, 0, value_ref<T>>;
    auto zero = std::size_t(0);

    return tensor_type(make_value(value), make_value(zero), make_value(zero), make_value(zero));
}


template <typename T, std::size_t N> requires(N > 0)
auto
empty(std::size_t (&&sizes)[N], device& device)
{
    auto shape = new std::size_t[N];

    std::size_t numel = 1;
    for (auto i = 0; i < N; i++) {
        numel *= sizes[i];
        shape[i] = sizes[i];
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
        std::make_shared<device_ref<T>>(data), make_owning(shape), make_owning(strides)
    );
}

template <typename T, std::size_t N, class InputIt> requires(N > 0)
auto
empty(InputIt begin, InputIt end, device& device)
{
    assert((end - begin) == N);

    std::size_t new_shape[N];
    for (std::size_t i = 0; i < N; i++) {
        new_shape[i] = *begin;
        ++begin;
    }

    return empty<T>(std::move(new_shape), device);
}


template <typename T, std::size_t N, ContiguousContainer Container> requires(N > 0)
auto
empty_like(const tensor<T, N, Container>& like, device& device)
{
    auto shape = like.shape();
    return empty<T, N>(shape.begin(), shape.end(), device);
}


template <typename T, std::size_t N> requires(N > 0)
auto
full(std::size_t (&&sizes)[N], const T& fill_value)
{
    auto t = empty<T>(std::move(sizes));
    std::fill_n(t.data_ptr(), t.numel(), fill_value);
    return t;
}


template <typename T, std::size_t N> requires(N > 0)
auto
full(std::size_t (&&sizes)[N], const T& fill_value, device& device)
{
    auto t = empty<T>(std::move(sizes), device);
    std::fill_n(t.data_ptr(), t.numel(), fill_value);
    return t;
}


template <typename T, std::size_t N> requires(N > 0)
auto
zeros(std::size_t (&&sizes)[N])
{
    return full<T>(std::move(sizes), 0);
}


/// Returns a tensor filled with random numbers from a uniform distribution on the
/// interval [0, 1).
///
/// The shape of the tensor is defined by the variable argument `sizes`.
template <typename T, std::size_t N> requires(N > 0)
auto
rand(std::size_t (&&sizes)[N])
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<T> distribution(0.0, 1.0);

    auto t = empty<T>(std::move(sizes));
    std::generate_n(t.data_ptr(), t.numel(), [&]() { return distribution(generator); });

    return t;
}


} //  namespace metalchat
