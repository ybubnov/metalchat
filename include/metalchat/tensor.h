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
#include <metalchat/format.h>
#include <metalchat/indexing.h>
#include <metalchat/iterator.h>


namespace metalchat {


template <typename T, ContiguousContainer Container> struct tensor_traits {
    using data_type = std::unique_ptr<Container>;
    using size_type = std::unique_ptr<array_ref<std::size_t>>;
};


template <typename T, std::size_t N, ContiguousContainer Container> class tensor_base {
public:
    using traits = tensor_traits<T, Container>;

    using iterator = tensor_iterator<T, N>;

    using const_iterator = const iterator;

    tensor_base(T* data, std::size_t* shape, std::size_t* strides, std::size_t* offsets)
    : m_data(make_weak(data)),
      m_shape(make_weak(shape)),
      m_strides(make_weak(strides)),
      m_offsets(make_weak(offsets))
    {}

    tensor_base(traits::data_type&& data, traits::size_type&& shape, traits::size_type&& strides)
    : m_data(std::move(data)),
      m_shape(std::move(shape)),
      m_strides(std::move(strides))
    {
        m_offsets = make_owning(new std::size_t[N]);
    }

    tensor_base(
        traits::data_type&& data,
        traits::size_type&& shape,
        traits::size_type&& strides,
        traits::size_type&& offsets
    )
    : m_data(std::move(data)),
      m_shape(std::move(shape)),
      m_strides(std::move(strides)),
      m_offsets(std::move(offsets))
    {}

    tensor_base(tensor_base<T, N, Container>&& t)
    : m_data(std::move(t.m_data)),
      m_shape(std::move(t.m_shape)),
      m_strides(std::move(t.m_strides)),
      m_offsets(std::move(t.m_offsets))
    {}

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

    inline constexpr const auto
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

    inline constexpr const auto
    sizes() const noexcept
    {
        return std::span(m_shape->data(), N);
    }

    inline constexpr const auto
    offsets() const noexcept
    {
        return std::span(m_offsets->data(), N);
    }

    inline std::size_t
    offset(std::size_t dim) const
    {
        if (dim >= N) {
            throw std::out_of_range("tensor");
        }
        return m_offsets->data()[dim];
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

    virtual void
    data_repr(std::ostream& os, int w) const
    {
        os << "[...]";
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

protected:
    traits::data_type m_data = nullptr;
    traits::size_type m_shape = nullptr;
    traits::size_type m_strides = nullptr;
    traits::size_type m_offsets = nullptr;
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

    tensor(T* data, std::size_t* shape, std::size_t* strides, std::size_t* offsets)
    : tensor_base<T, N, Container>(data, shape, strides, offsets)
    {}

    tensor(traits::data_type&& data, traits::size_type&& shape, traits::size_type&& strides)
    : tensor_base<T, N, Container>(std::move(data), std::move(shape), std::move(strides))
    {}

    tensor(
        traits::data_type&& data,
        traits::size_type&& shape,
        traits::size_type&& strides,
        traits::size_type&& offsets
    )
    : tensor_base<T, N, Container>(
          std::move(data), std::move(shape), std::move(strides), std::move(offsets)
      )
    {}

    tensor<T, N - 1>
    at(std::size_t i)
    {
        auto new_data = this->data_ptr() + this->stride(0) * (this->offset(0) + i);
        auto new_shape = this->m_shape->data() + 1;
        auto new_strides = this->m_strides->data() + 1;
        auto new_offsets = this->m_offsets->data() + 1;
        return tensor<T, N - 1>(new_data, new_shape, new_strides, new_offsets);
    }

    const tensor<const T, N - 1>
    at(std::size_t i) const
    {
        auto new_data = this->data_ptr() + this->stride(0) * (this->offset(0) + i);
        auto new_shape = this->m_shape->data() + 1;
        auto new_strides = this->m_strides->data() + 1;
        auto new_offsets = this->m_offsets->data() + 1;
        return tensor<const T, N - 1>(new_data, new_shape, new_strides, new_offsets);
    }

    tensor<T, N - 1>
    operator[](std::size_t i)
    {
        return at(i);
    }

    template <indexing::SliceConvertible... S>
    auto
    operator[](const S&... slices) requires(sizeof...(slices) <= N)
    {
        constexpr auto slices_size = sizeof...(slices);
        std::array<indexing::slice, slices_size> slices_array
            = {(static_cast<indexing::slice>(slices))...};

        auto shape = new std::size_t[N];
        auto offsets = new std::size_t[N];

        for (std::size_t i = 0; i < slices_size; i++) {
            auto slice = slices_array[i];

            auto stop = std::min(slice.stop.value_or(this->size(i)), this->size(i));
            auto start = std::min(slice.start.value_or(0), stop);

            shape[i] = stop - start;
            offsets[i] = start;
        }

        return tensor<T, N, weak_ref<T>>(
            make_weak(this->data_ptr()), make_owning(shape), make_weak(this->m_strides->data()),
            make_owning(offsets)
        );
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
    transpose(const Dimensions... dims) requires(sizeof...(dims) == N)
    {
        auto shape = new std::size_t[N]{this->size(static_cast<std::size_t>(dims))...};
        auto strides = new std::size_t[N]{this->stride(static_cast<std::size_t>(dims))...};
        auto offsets = new std::size_t[N]{this->offset(static_cast<std::size_t>(dims))...};

        return tensor<T, N, weak_ref<T>>(
            make_weak(this->data_ptr()), make_owning(shape), make_owning(strides),
            make_owning(offsets)
        );
    }

    auto
    t() requires(N == 2)
    {
        return std::move(transpose(0, 1));
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

    tensor(T* data, std::size_t* shape, std::size_t* strides, std::size_t* offsets)
    : tensor_base<T, 1, Container>(data, shape, strides, offsets)
    {}

    tensor(traits::data_type&& data, traits::size_type&& shape, traits::size_type&& strides)
    : tensor_base<T, 1, Container>(std::move(data), std::move(shape), std::move(strides))
    {}

    tensor(
        traits::data_type&& data,
        traits::size_type&& shape,
        traits::size_type&& strides,
        traits::size_type&& offsets
    )
    : tensor_base<T, 1, Container>(
          std::move(data), std::move(shape), std::move(strides), std::move(offsets)
      )
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

    template <indexing::SliceConvertible S>
    auto
    operator[](const S& slice)
    {
        auto shape = new std::size_t[1];
        auto offsets = new std::size_t[1];

        auto stop = std::min(slice.stop.value_or(this->size(0)), this->size(0));
        auto start = std::min(slice.start.value_or(0), stop);

        shape[0] = stop - start;
        offsets[0] = start;

        return tensor<T, 1, weak_ref<T>>(
            make_weak(this->data_ptr()), make_owning(shape), make_weak(this->m_strides->data()),
            make_owning(offsets)
        );
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

    tensor(
        traits::data_type&& data,
        traits::size_type&& shape,
        traits::size_type&& strides,
        traits::size_type&& offsets
    )
    : tensor_base<T, 0, Container>(
          std::move(data), std::move(shape), std::move(strides), std::move(offsets)
      )
    {}

    void
    data_repr(std::ostream& os, int w) const override
    {
        os << *this->data_ptr();
    }
};


template <typename T, std::size_t N> requires(N > 0)
auto
empty(std::size_t (&&sizes)[N])
{
    auto shape = new std::size_t[N];
    auto strides = new std::size_t[N];

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

    using tensor_type = tensor<T, N, owning_ref<T>>;

    return tensor_type(make_owning(data), make_owning(shape), make_owning(strides));
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
        std::make_unique<device_ref<T>>(data), make_owning(shape), make_owning(strides)
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
