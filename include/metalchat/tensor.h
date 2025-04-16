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


template <typename T, std::size_t N, ContiguousContainer Container> class tensor {
public:
    using traits_type = tensor_traits<T, Container>;

    using value_type = T;

    using pointer_type = T*;

    using container_type = Container;

    using iterator = tensor_iterator<T, N>;

    using const_iterator = const iterator;

    /// Tensor move constructor
    ///
    /// The newly-created tensor contains the exact contents of the moved instance.
    /// The contents of the moved instance are a valid, but unspecified tensor.
    tensor(tensor&& t) noexcept = default;

    tensor(const tensor& t) noexcept = delete;

    tensor(const T& value) requires(std::same_as<Container, value_ref<T>> && N == 0)
    : m_data(std::make_shared<Container>(value)),
      m_shape(make_value<std::size_t>(0)),
      m_strides(make_value<std::size_t>(0)),
      m_offsets(make_value<std::size_t>(0))
    {}

    tensor(const std::size_t (&&sizes)[N]) requires(std::same_as<Container, owning_ref<T>> && N > 0)
    {
        _m_initialize(std::move(sizes));
        m_data = std::make_shared<owning_ref<T>>(new T[numel()]);
    }

    tensor(const std::size_t (&&sizes)[N], const traits_type::data_type& data)
    : m_data(data)
    {
        _m_initialize(std::move(sizes));
    }

    template <std::forward_iterator ForwardIt>
    tensor(ForwardIt first, ForwardIt last, const traits_type::data_type& data)
    : m_data(data)
    {
        _m_initialize(first, last);
    }

    tensor(const std::size_t (&&sizes)[N], MTL::Device* device)
        requires(std::same_as<Container, device_ref<T>> && N > 0)
    {
        _m_initialize(std::move(sizes));

        auto buf_size = numel() * sizeof(T);
        auto buf = NS::TransferPtr(device->newBuffer(buf_size, MTL::ResourceStorageModeShared));

        m_data = std::make_shared<device_ref<T>>(buf);
    }

    tensor(const std::size_t (&&sizes)[N], device& device)
        requires(std::same_as<Container, device_ref<T>> && N > 0)
    : tensor(std::move(sizes), (*device))
    {}

    tensor(T* data, std::size_t* shape, std::size_t* strides, std::size_t* offsets)
    : tensor(
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

    inline T*
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
            throw std::out_of_range(
                std::format("tensor::stride: dim {} exceeds tensor dimensionality {}", dim, N)
            );
        }
        return m_strides->data()[dim];
    }

    inline void
    set_stride(std::size_t dim, std::size_t i)
    {
        m_strides->data()[dim] = i;
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
            throw std::out_of_range(
                std::format("tensor::size: dim {} exceeds tensor dimensionality {}", dim, N)
            );
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
            throw std::out_of_range(
                std::format("tensor::offset: dim {} exceed tensor dimensionality {}", dim, N)
            );
        }
        return m_offsets->data()[dim];
    }

    inline void
    set_offset(std::size_t dim, std::size_t i)
    {
        m_offsets->data()[dim] = i;
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

    inline container_type&
    container() const
    {
        return *m_data;
    }

    std::size_t
    container_offset() const
    {
        std::size_t off = 0;
        for (std::size_t dim = 0; dim < N; dim++) {
            off += offset(dim);
        }
        return off;
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

    auto
    at(std::size_t i) const
    {
        if (auto size0 = size(0); i >= size0) {
            std::out_of_range(std::format("tensor::at: index {} is out of tensor size {}", i, size0)
            );
        }

        auto data = data_ptr() + stride(0) * i + offset(0);
        auto sizes = m_shape->data() + 1;
        auto strides = m_strides->data() + 1;
        auto offsets = m_offsets->data() + 1;
        return tensor<T, N - 1, weak_ref<T>>(data, sizes, strides, offsets);
    }

    auto
    at(std::size_t i)
    {
        using tensor_type = tensor<T, N - 1, weak_ref<T>>;
        return const_cast<tensor const&>(*this).at(i);
    }

    template <indexing::slice_convertible... S>
    auto
    index_select(const S&... slices) requires(sizeof...(slices) == N)
    {
        tensor t(m_data);
        std::size_t dim = 0;

        ([&] {
            indexing::slice slice(slices);
            auto stop = std::min(slice.stop.value_or(size(dim)), size(dim));
            auto start = std::min(slice.start.value_or(0), stop);

            t.set_size(dim, stop - start);
            t.set_stride(dim, stride(dim));
            t.set_offset(dim, start * t.stride(dim));
            dim++;
        }(), ...);

        return t;
    }

    template <indexing::size_convertible... S>
    const T&
    value_select(const S&... indices) const requires(sizeof...(indices) == N)
    {
        std::size_t ptr_offset = 0;
        std::size_t dim = 0;

        ([&] {
            auto i = static_cast<std::size_t>(indices);
            if (auto size_d = size(dim); i >= size_d) {
                throw std::out_of_range(std::format(
                    "tensor::value_select index {} for dimension {} is outside of range {}", i, dim,
                    size_d
                ));
            }

            ptr_offset += stride(dim) * i + offset(dim);
            dim++;
        }(), ...);

        return *(data_ptr() + ptr_offset);
    }

    template <indexing::size_convertible... S>
    T&
    value_select(const S&... sizes) requires(sizeof...(sizes) == N)
    {
        return const_cast<T&>(const_cast<tensor const&>(*this).value_select(sizes...));
    }

    auto
    narrow(std::size_t dim, std::size_t start, std::size_t length) const
    {
        tensor t(m_data);
        for (auto i = 0; i < N; i++) {
            t.set_size(i, size(i));
            t.set_stride(i, stride(i));
            t.set_offset(i, offset(i));
        }
        t.set_offset(dim, t.stride(dim) * start);
        t.set_size(dim, length);
        return t;
    }

    tensor&
    operator=(const tensor& other)
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

    tensor&
    operator=(tensor&& other)
        = default;

    template <indexing::size_convertible... S>
    T&
    operator[](const S&... sizes) requires(sizeof...(sizes) == N)
    {
        return value_select(sizes...);
    }

    template <indexing::size_convertible... S>
    const T&
    operator[](const S&... sizes) const requires(sizeof...(sizes) == N)
    {
        return value_select(sizes...);
    }

    template <indexing::slice_convertible... S>
    auto
    operator[](const S&... slices) requires(sizeof...(slices) == N)
    {
        return index_select(slices...);
    }

    auto
    operator[](std::size_t i) requires(N > 1)
    {
        return at(i);
    }

    /// Returns a tensor with dimensions transposed.
    auto
    transpose(const std::size_t (&&dims)[N]) const
    {
        tensor t(m_data);
        for (auto i = 0; i < N; i++) {
            t.set_size(i, size(dims[i]));
            t.set_stride(i, stride(dims[i]));
            t.set_offset(i, offset(dims[i]));
        }
        return t;
    }

    auto
    t() const requires(N == 2)
    {
        return transpose({1, 0});
    }

    tensor<T, N + 1, Container>
    expand_dims(std::size_t dim) const
    {
        assert(dim <= N);

        int sizes[N + 1];
        sizes[dim] = 1;

        for (auto i = 0; i < dim; i++) {
            sizes[i] = size(i);
        }
        for (auto i = dim; i < N; i++) {
            sizes[i + 1] = size(i);
        }

        return view(std::move(sizes));
    }

    template <std::size_t M>
    tensor<T, M, Container>
    view(const int (&&dims)[M]) const requires(M > 0)
    {
        auto tensor_numel = numel();
        auto view_numel = 1;

        auto inferred_size = tensor_numel;
        auto inferred_dim = -1;

        std::size_t view_sizes[M];
        std::size_t view_strides[M];

        for (auto i = 0; i < M; i++) {
            if (dims[i] == -1) {
                if (inferred_dim >= 0) {
                    throw std::invalid_argument("tensor::view: only one position can be inferred");
                }
                inferred_dim = i;
            } else {
                view_sizes[i] = std::size_t(dims[i]);
                view_numel *= std::size_t(dims[i]);
                inferred_size = inferred_size / dims[i];
            }
        }
        if (inferred_dim >= 0) {
            view_sizes[inferred_dim] = inferred_size;
            view_numel *= inferred_size;
        }

        if (view_numel != tensor_numel) {
            throw std::invalid_argument(std::format(
                "tensor::view: view numel is not the same as tensor numel {} != {}", view_numel,
                tensor_numel
            ));
        }

        tensor_numel = 1;
        view_numel = 1;
        int view_i = M - 1;
        auto base_stride = stride(N - 1);

        for (int i = N - 1; i >= 0; i--) {
            tensor_numel *= size(i);

            // When tensor stride is not equal to the "default" stride (which could happen
            // in case of slicing or narrowing a tensor), try computing new strides according
            // to the layout of the original tensor.
            //
            // A new shape of a view might break the contiguous layout of memory, in this
            // case throw an `invalid_argument` exception to the caller.
            if (i == 0 || stride(i - 1) != tensor_numel * base_stride) {
                while (view_i >= 0 && (view_numel < tensor_numel || view_sizes[view_i] == 1)) {
                    view_strides[view_i] = view_numel * base_stride;
                    view_numel *= view_sizes[view_i];
                    view_i--;
                }

                if (view_numel != tensor_numel) {
                    throw std::invalid_argument(std::format(
                        ("tensor::view: shape is invalid for input of size {}, "
                         "considering copying the tensor"),
                        numel()
                    ));
                }

                if (i > 0) {
                    base_stride = stride(i - 1);
                    tensor_numel = 1;
                    view_numel = 1;
                }
            }
        }

        auto t = tensor<T, M, Container>(std::move(view_sizes), m_data);
        t.set_offset(0, container_offset());
        for (std::size_t dim = 0; dim < M; dim++) {
            t.set_stride(dim, view_strides[dim]);
        }

        return t;
    }

protected:
    traits_type::data_type m_data = nullptr;
    traits_type::size_type m_shape = nullptr;
    traits_type::size_type m_strides = nullptr;
    traits_type::size_type m_offsets = nullptr;

    inline void
    set_size(std::size_t dim, std::size_t i)
    {
        m_shape->data()[dim] = i;
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

    tensor(const traits_type::data_type& data)
    : m_data(data)
    {
        _m_initialize();
    }

    tensor(
        const traits_type::data_type& data,
        traits_type::size_type&& shape,
        traits_type::size_type&& strides,
        traits_type::size_type&& offsets
    )
    : m_data(data),
      m_shape(std::move(shape)),
      m_strides(std::move(strides)),
      m_offsets(std::move(offsets))
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


template <typename T, std::size_t N> requires(N > 0)
auto
empty(const std::size_t (&&sizes)[N], MTL::Device* device)
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
