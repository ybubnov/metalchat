#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
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
    using data_type = std::shared_ptr<Container>;

    using size_type = std::unique_ptr<array_ref<std::size_t>>;
};


template <uint32_t N> struct tensor_layout {
    uint32_t sizes[N];
    uint32_t strides[N];
    uint32_t offsets[N];
};


template <typename T, std::size_t N, ContiguousContainer Container> class tensor_base {
public:
    using traits = tensor_traits<T, Container>;

    using value_type = T;

    using pointer_type = T*;

    using container_type = Container;

    using iterator = tensor_iterator<T, N>;

    using const_iterator = const iterator;

    /// Tensor move constructor
    ///
    /// The newly-created tensor contains the exact contents of the moved instance.
    /// The contents of the moved instance are a valid, but unspecified tensor.
    tensor_base(tensor_base&& t) noexcept = default;

    tensor_base(const tensor_base& t) = delete;

    tensor_base(const std::size_t (&&sizes)[N])
        requires(std::same_as<Container, owning_ref<T>> && N > 0)
    {
        _m_initialize(std::move(sizes));
        m_data = std::make_shared<owning_ref<T>>(new T[numel()]);
    }

    tensor_base(const std::size_t (&&sizes)[N], const traits::data_type& data)
    : m_data(data)
    {
        _m_initialize(std::move(sizes));
    }

    template <std::forward_iterator ForwardIt>
    tensor_base(ForwardIt first, ForwardIt last, const traits::data_type& data)
    : m_data(data)
    {
        _m_initialize(first, last);
    }

    tensor_base(const std::size_t (&&sizes)[N], device& device)
        requires(std::same_as<Container, device_ref<T>> && N > 0)
    {
        _m_initialize(std::move(sizes));

        auto buf_size = numel() * sizeof(T);
        auto buf = NS::TransferPtr(device->newBuffer(buf_size, MTL::ResourceStorageModeShared));

        m_data = std::make_shared<device_ref<T>>(buf);
    }

    tensor_base(T* data, std::size_t* shape, std::size_t* strides, std::size_t* offsets)
    : tensor_base(
          std::make_shared<weak_ref<T>>(data),
          make_weak(shape),
          make_weak(strides),
          make_weak(offsets)
      )
    {}

    static constexpr std::size_t
    dim()
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

    tensor_layout<N>
    layout() const
    {
        tensor_layout<N> layout;
        std::copy_n(m_shape->data(), N, layout.sizes);
        std::copy_n(m_strides->data(), N, layout.strides);
        std::copy_n(m_offsets->data(), N, layout.offsets);
        return layout;
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
    index_select(const S&... slices) requires(sizeof...(slices) == N)
    {
        tensor_base t(m_data);
        std::size_t i = 0;

        ([&] {
            indexing::slice slice(slices);
            auto stop = std::min(slice.stop.value_or(size(i)), size(i));
            auto start = std::min(slice.start.value_or(0), stop);

            t.set_size(i, stop - start);
            t.set_stride(i, stride(i));
            t.set_offset(i, start);
            i++;
        }(), ...);

        return t;
    }

    template <indexing::size_convertible... S>
    T&
    value_select(const S&... indices) requires(sizeof...(indices) == N)
    {
        std::size_t ptr_offset = 0;
        std::size_t dim = 0;

        ([&] {
            auto i = static_cast<std::size_t>(indices);
            if (i >= size(dim)) {
                throw std::out_of_range(std::format(
                    "tensor::value_select index {} for dimension {} is outside of range {}", i, dim,
                    size(dim)
                ));
            }

            ptr_offset += stride(dim) * (offset(dim) + i);
            dim++;
        }(), ...);

        return *(data_ptr() + ptr_offset);
    }

    template <indexing::size_convertible... S>
    const T&
    value_select(const S&... sizes) const requires(sizeof...(sizes) == N)
    {
        std::size_t ptr_offset = 0;
        std::size_t dim = 0;

        ([&] {
            auto i = static_cast<std::size_t>(sizes);
            ptr_offset += stride(dim) * (offset(dim) + i);
            dim++;
        }(), ...);

        return *(data_ptr() + ptr_offset);
    }

    auto
    narrow(std::size_t dim, std::size_t start, std::size_t length)
    {
        tensor_base t(m_data);
        for (auto i = 0; i < N; i++) {
            t.set_size(i, size(i));
            t.set_stride(i, stride(i));
            t.set_offset(i, offset(i));
        }
        t.set_offset(dim, start);
        t.set_size(dim, length);
        return t;
    }

    tensor_base&
    operator=(const tensor_base& other)
    {
        if (this == &other) {
            return *this;
        }
        for (std::size_t i = 0; i < N; i++) {
            assert(other.size(i) == this->size(i));
        }
        std::copy(other.begin(), other.end(), begin());
        return *this;
    }

    tensor_base&
    operator=(tensor_base&& other)
        = default;

    template <indexing::size_convertible... S>
    T&
    operator[](const S&... sizes)
    {
        return value_select(sizes...);
    }

    /// Returns a tensor with dimensions transposed.
    auto
    transpose(const std::size_t (&&dims)[N])
    {
        tensor_base t(m_data);
        for (auto i = 0; i < N; i++) {
            t.set_size(i, size(dims[i]));
            t.set_stride(i, stride(dims[i]));
            t.set_offset(i, offset(dims[i]));
        }
        return t;
    }

    tensor_base<T, N + 1, Container>
    expand_dims(const std::size_t dim)
    {
        assert(dim <= N);
        assert(is_contiguous());

        std::size_t sizes[N + 1];
        sizes[dim] = 1;

        for (auto i = 0; i < dim; i++) {
            sizes[i] = size(i);
        }
        for (auto i = dim; i < N; i++) {
            sizes[i + 1] = size(i);
        }

        return tensor_base<T, N + 1, Container>(std::move(sizes), m_data);
    }

    template <std::size_t M>
    tensor_base<T, M, Container>
    reshape(const int (&&dims)[M]) const requires(M > 0)
    {
        // So far only reshaping of contiguous tensors is supported.
        assert(is_contiguous());

        auto size = numel();
        auto new_size = 1;

        auto inferred_size = size;
        auto inferred_dim = -1;

        std::size_t sizes[M];

        for (auto i = 0; i < M; i++) {
            if (dims[i] == -1) {
                if (inferred_dim >= 0) {
                    throw std::invalid_argument("only one position can be inferred");
                }
                inferred_dim = i;
            } else {
                sizes[i] = std::size_t(dims[i]);
                new_size *= std::size_t(dims[i]);
                inferred_size = inferred_size / dims[i];
            }
        }
        if (inferred_dim >= 0) {
            sizes[inferred_dim] = inferred_size;
            new_size *= inferred_size;
        }

        if (new_size != size) {
            throw std::invalid_argument(
                std::format("tensor::reshape: shape is invalid for input size {}", size)
            );
        }

        return tensor_base<T, M, Container>(std::move(sizes), m_data);
    }

protected:
    traits::data_type m_data = nullptr;
    traits::size_type m_shape = nullptr;
    traits::size_type m_strides = nullptr;
    traits::size_type m_offsets = nullptr;

    inline void
    set_size(std::size_t dim, std::size_t i)
    {
        m_shape->data()[dim] = i;
    }

    inline void
    set_stride(std::size_t dim, std::size_t i)
    {
        m_strides->data()[dim] = i;
    }

    inline void
    set_offset(std::size_t dim, std::size_t i)
    {
        m_offsets->data()[dim] = i;
    }

    void
    _m_initialize()
    {
        m_shape = make_owning(new std::size_t[N]());
        m_strides = make_owning(new std::size_t[N]());
        m_offsets = make_owning(new std::size_t[N]());
    }

    void
    _m_initialize_strides()
    {
        m_strides->data()[N - 1] = 1;
        for (auto i = N - 2; i < N; --i) {
            m_strides->data()[i] = m_strides->data()[i + 1] * m_shape->data()[i + 1];
        }
    }

    void
    _m_initialize(const std::size_t (&&sizes)[N])
    {
        _m_initialize(sizes, sizes + N);
        _m_initialize_strides();
    }

    template <std::forward_iterator ForwardIt>
    void
    _m_initialize(ForwardIt first, ForwardIt last)
    {
        assert(std::distance(first, last) == N);

        _m_initialize();
        std::copy(first, last, m_shape->data());
        _m_initialize_strides();
    }

    tensor_base(const traits::data_type& data)
    : m_data(data)
    {
        _m_initialize();
    }

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

    tensor(const std::size_t (&&sizes)[N], device& device)
    : _Base(std::move(sizes), device)
    {}

    tensor(T* data, std::size_t* shape, std::size_t* strides, std::size_t* offsets)
    : _Base(data, shape, strides, offsets)
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

    const auto
    operator[](std::size_t i) const
    {
        return this->at(i);
    }

    template <indexing::SliceConvertible... S>
    tensor
    operator[](const S&... slices)
    {
        return tensor(_Base::index_select(slices...));
    }

    template <indexing::size_convertible... S> requires(sizeof...(S) > 1)
    T&
    operator[](const S&... sizes)
    {
        return _Base::value_select(sizes...);
    }

    template <indexing::size_convertible... S> requires(sizeof...(S) > 1)
    const T&
    operator[](const S&... sizes) const
    {
        return _Base::value_select(sizes...);
    }

    auto
    transpose(const std::size_t (&&dims)[N])
    {
        return tensor(_Base::transpose(std::move(dims)));
    }

    auto
    t() requires(N == 2)
    {
        return transpose({1, 0});
    }

    template <std::size_t M>
    tensor<T, M, Container>
    reshape(const int (&&dims)[M]) const requires(M > 0)
    {
        return tensor<T, M, Container>(_Base::reshape(std::move(dims)));
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

    tensor(const std::size_t (&&sizes)[1], device& device)
    : _Base(std::move(sizes), device)
    {}

    tensor(T* data, std::size_t* shape, std::size_t* strides, std::size_t* offsets)
    : _Base(data, shape, strides, offsets)
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
        return tensor(_Base::index_select(slices...));
    }

    auto
    t() const
    {
        return tensor(this->m_data->data(), this->m_shape->data(), this->m_strides->data());
    }

    template <std::size_t M>
    tensor<T, M, Container>
    reshape(const int (&&dims)[M]) const
    {
        return tensor<T, M, Container>(_Base::reshape(std::move(dims)));
    }
};


template <typename T, ContiguousContainer Container>
class tensor<T, 0, Container> : public tensor_base<T, 0, Container> {
private:
    using _Base = tensor_base<T, 0, Container>;

public:
    using traits = tensor_traits<T, Container>;

    tensor(_Base&& t)
    : _Base(std::move(t))
    {}

    tensor(const T& value)
    : _Base(
          std::make_shared<value_ref<T>>(value),
          make_value<std::size_t>(0),
          make_value<std::size_t>(0),
          make_value<std::size_t>(0)
      )
    {}
};


template <typename T, std::size_t N> requires(N > 0)
auto
empty(const std::size_t (&&sizes)[N])
{
    return tensor<T, N, owning_ref<T>>(std::move(sizes));
}


template <typename T>
auto
scalar(const T& value)
{
    return tensor<T, 0, value_ref<T>>(T(value));
}


template <typename T, std::size_t N> requires(N > 0)
auto
empty(const std::size_t (&&sizes)[N], device& device)
{
    return tensor<T, N, device_ref<T>>(std::move(sizes), device);
}


template <typename T, std::size_t N, class InputIt> requires(N > 0)
auto
empty(InputIt begin, InputIt end)
{
    assert((end - begin) == N);

    std::size_t new_shape[N];
    for (std::size_t i = 0; i < N; i++) {
        new_shape[i] = *begin;
        ++begin;
    }

    return empty<T>(std::move(new_shape));
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
empty_like(const tensor<T, N, Container>& like)
{
    auto shape = like.shape();
    return empty<T, N>(shape.begin(), shape.end());
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
full(const std::size_t (&&sizes)[N], const T& fill_value)
{
    auto t = empty<T>(std::move(sizes));
    std::fill_n(t.data_ptr(), t.numel(), fill_value);
    return t;
}


template <typename T, std::size_t N> requires(N > 0)
auto
full(const std::size_t (&&sizes)[N], const T& fill_value, device& device)
{
    auto t = empty<T>(std::move(sizes), device);
    std::fill_n(t.data_ptr(), t.numel(), fill_value);
    return t;
}


template <typename T, std::size_t N> requires(N > 0)
auto
zeros(const std::size_t (&&sizes)[N])
{
    return full<T>(std::move(sizes), 0);
}


/// Returns a tensor filled with random numbers from a uniform distribution on the
/// interval [0, 1).
///
/// The shape of the tensor is defined by the variable argument `sizes`.
template <typename T, std::size_t N> requires(N > 0)
auto
rand(const std::size_t (&&sizes)[N])
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<T> distribution(0.0, 1.0);

    auto t = empty<T>(std::move(sizes));
    std::generate_n(t.data_ptr(), t.numel(), [&]() { return distribution(generator); });

    return t;
}


template <typename T, std::size_t N, std::forward_iterator ForwardIt> requires(N > 0)
auto
to_tensor(const std::size_t (&&sizes)[N], ForwardIt first, ForwardIt last)
{
    auto t = empty<T>(std::move(sizes));
    auto distance = std::distance(first, last);

    if (distance != t.numel()) {
        throw std::invalid_argument(std::format(
            "tensor: iterators differences ({}) should be equal to tensor numel ({})", distance,
            t.numel()
        ));
    }

    std::copy(first, last, t.data_ptr());
    return t;
}


template <typename T, std::size_t N, ContiguousContainer Container>
auto
to_tensor(const tensor<T, N, Container>& t)
{
    auto tt = empty_like(t);
    std::copy_n(t.data_ptr(), t.numel(), tt.data_ptr());
    return tt;
}


} //  namespace metalchat
