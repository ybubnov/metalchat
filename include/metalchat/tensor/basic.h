#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <format>
#include <random>
#include <span>
#include <utility>

#include <metalchat/accelerator.h>
#include <metalchat/allocator.h>
#include <metalchat/container.h>
#include <metalchat/tensor/concept.h>
#include <metalchat/tensor/indexing.h>
#include <metalchat/tensor/iterator.h>


namespace metalchat {


class basic_tensor {
public:
    virtual std::size_t
    dimensions() const
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

    virtual ~basic_tensor() {}

protected:
    void
    deduce_view_sizes(const int* begin, std::size_t count, std::size_t* result) const;

    void
    deduce_view_strides(const std::size_t* sizes, std::size_t count, std::size_t* strides) const;
};


template <typename T, std::size_t N, contiguous_container Container = random_memory_container<T>>
class tensor : public basic_tensor {
public:
    /// Alias of the tensor type.
    using value_type = T;

    /// Pointer to the tensor type.
    using pointer_type = T*;

    using container_type = Container;

    using container_pointer = std::shared_ptr<container_type>;

    using iterator = tensor_iterator<T, N>;

    using const_iterator = const iterator;

    using descriptor_type = memory_container<std::size_t>;

    using descriptor_pointer = std::shared_ptr<descriptor_type>;

    tensor() { initialize(); }

    tensor(tensor&& t) noexcept = default;

    tensor(const tensor& t) noexcept = default;

    /// Constructs a new tensor with a scalar value. By default, method is using a \ref
    /// scalar_memory_allocator, but an arbitrary typed allocator could be used.
    ///
    /// \param value the value to initialize a tensor with.
    /// \param alloc allocator to use for all memory allocations of this container.
    ///
    /// ```c++
    /// auto T = tensor<float, 0, scalar_memory_container<float>>(3.0f);
    /// // Same as:
    /// auto S = scalar<float>(3.0f);
    /// ```
    template <allocator_t<T> Allocator = scalar_memory_allocator<T>>
    tensor(const T& value, Allocator alloc = Allocator())
    : _M_data(alloc.allocate(1)),
      _M_sizes(make_scalar_container<std::size_t>(0)),
      _M_strides(make_scalar_container<std::size_t>(0)),
      _M_offsets(make_scalar_container<std::size_t>(0))
    {
        *data_ptr() = value;
    }

    /// Constructs a new empty tensor with the size defined by the content of the range
    /// [`first`, `last`). Each iterator in [`first`, `last`) is dereferenced exactly once.
    ///
    /// The distance `std::distance(first, last)` between iterators should be equal to the
    /// tensor dimensionality `N`.
    ///
    /// \param first, last the pair of iterators defining the range of dimensions of the tensor
    ///     to copy from.
    /// \param alloc allocator to use for all memory allocations of this container.
    ///
    /// ```c++
    /// auto sizes = std::vector({4, 3, 6, 7});
    /// auto T = tensor<float, 4>(sizes.begin(), sizes.end(), random_memory_allocator<float>());
    /// ```
    template <std::forward_iterator ForwardIt, allocator_t<T> Allocator> requires(N > 0)
    tensor(ForwardIt first, ForwardIt last, Allocator alloc)
    {
        initialize(first, last);
        _M_data = alloc.allocate(numel());
    }

    template <
        std::forward_iterator ForwardIt,
        allocator_t<T> Allocator = random_memory_allocator<T>>
    tensor(ForwardIt first, ForwardIt last, T* data, Allocator alloc = Allocator())
    {
        initialize(first, last);
        _M_data = alloc.allocate(data, numel());
    }

    template <std::forward_iterator ForwardIt>
    tensor(ForwardIt first, ForwardIt last, const container_pointer& data)
    {
        initialize(first, last);
        _M_data = data;
    }

    template <allocator_t<T> Allocator = random_memory_allocator<T>>
    tensor(const std::span<std::size_t, N> sizes, Allocator alloc = Allocator())
    : tensor(sizes.begin(), sizes.end(), alloc)
    {}

    tensor(const std::span<std::size_t, N> sizes, const container_pointer& data)
    : tensor(sizes.begin(), sizes.end(), data)
    {}

    template <allocator_t<T> Allocator = random_memory_allocator<T>>
    tensor(std::size_t (&&sizes)[N], Allocator alloc = Allocator())
    : tensor(std::span<std::size_t, N>(sizes, N), alloc)
    {}

    tensor(std::size_t (&&sizes)[N], const container_pointer& data)
    : tensor(std::span<std::size_t, N>(sizes, N), data)
    {}

    tensor(
        const std::span<std::size_t>& sizes,
        const std::span<std::size_t>& strides,
        const std::span<std::size_t>& offsets,
        const container_pointer& data
    )
    {
        initialize();
        std::copy(sizes.begin(), sizes.end(), _M_sizes->data());
        std::copy(strides.begin(), strides.end(), _M_strides->data());
        std::copy(offsets.begin(), offsets.end(), _M_offsets->data());
        _M_data = data;
    }

    /// Returns the number of dimension of the tensor. This is a const expression.
    ///
    /// See also \ref tensor::dimensions.
    static constexpr std::size_t
    dim()
    {
        return N;
    }

    /// Returns the number of dimension of the tensor.
    std::size_t
    dimensions() const
    {
        return dim();
    }

    /// Returns a pointer to the first element of the tensor.
    pointer_type
    data_ptr() noexcept
    {
        return const_cast<tensor const&>(*this).data_ptr();
    }

    /// Returns a pointer to the first element of the tensor.
    const pointer_type
    data_ptr() const noexcept
    {
        if (!_M_data) {
            return nullptr;
        }
        return _M_data->data();
    }

    /// Returns the stride of the specified tensor dimension.
    ///
    /// \param dim the dimension for which to retrieve the stride.
    ///
    /// ```c++
    /// auto T = empty<float>({2, 5});
    /// std::cout << T.stride(0) << std::endl;
    /// // out: 5
    /// std::cout << T.stride(1) << std::endl;
    /// // out: 1
    /// ```
    ///
    /// See also \ref tensor::strides.
    std::size_t
    stride(std::size_t dim) const
    {
        if (dim >= N) {
            throw std::out_of_range(
                std::format("tensor::stride: dim {} exceeds tensor dimensionality {}", dim, N)
            );
        }
        return _M_strides->data()[dim];
    }

    void
    set_stride(std::size_t dim, std::size_t i)
    {
        _M_strides->data()[dim] = i;
    }

    /// Returns strides of the tensor.
    ///
    /// ```c++
    /// auto T = empty<float>({2, 5});
    /// std::cout << T.strides() << std::endl;
    /// // out: 5, 1
    /// ```
    ///
    /// See also \ref tensor::stride.
    const std::span<std::size_t>
    strides() const noexcept
    {
        return std::span<std::size_t, N>(_M_strides->data(), N);
    }

    /// Returns the size of the specified tensor dimension.
    ///
    /// \param dim the dimension for which to retrieve the size.
    ///
    /// ```c++
    /// auto T = empty<float>({3, 4, 5});
    /// std::cout << T.size(1) << std::endl;
    /// // out: 4
    /// ```
    std::size_t
    size(std::size_t dim) const
    {
        if (dim >= N) {
            throw std::out_of_range(
                std::format("tensor::size: dim {} exceeds tensor dimensionality {}", dim, N)
            );
        }
        return _M_sizes->data()[dim];
    }

    /// Returns the sizes of the tensor.
    ///
    /// ```c++
    /// auto T = empty<float>({3, 4, 5});
    /// std::cout << T.sizes() << std::endl;
    /// // out: 3, 4, 5
    /// ```
    ///
    /// See also \ref tensor::shape.
    const std::span<std::size_t>
    sizes() const noexcept
    {
        return std::span<std::size_t, N>(_M_sizes->data(), N);
    }

    /// Returns the sizes of the tensor. Unlike \ref tensor::sizes method, this method returns
    /// a span with the fixed extent.
    ///
    /// See also \ref tensor::sizes.
    const std::span<std::size_t, N>
    shape() const noexcept
    {
        return std::span<std::size_t, N>(_M_sizes->data(), N);
    }

    /// Returns the container offset of the specified tensor dimension. The offset is always in
    /// units of \ref value_type.
    ///
    /// \param dim the dimension for which to retrieve the offset.
    ///
    /// ```c++
    /// auto T = empty<float>({3, 4, 5});
    /// auto S = T[slice(), slice(1, 3), slice()];
    /// std::cout << S.offset(1) << std::endl;
    /// // out: 1
    /// ```
    std::size_t
    offset(std::size_t dim) const
    {
        if (dim >= N) {
            throw std::out_of_range(
                std::format("tensor::offset: dim {} exceed tensor dimensionality {}", dim, N)
            );
        }
        return _M_offsets->data()[dim];
    }

    void
    set_offset(std::size_t dim, std::size_t i)
    {
        _M_offsets->data()[dim] = i;
    }

    /// Returns the offsets of the tensor container.
    ///
    /// ```c++
    /// auto T = empty<float>({3, 4, 5});
    /// std::cout << T.offsets() << std::endl;
    /// // out: 0, 0, 0
    /// auto S = T[slice(1, 2), slice(2, 3), slice(2, 4)];
    /// std::cout << T.offsets() << std::endl;
    /// // out: 1, 2, 2
    /// ```
    const std::span<std::size_t>
    offsets() const noexcept
    {
        return std::span<std::size_t, N>(_M_offsets->data(), N);
    }

    bool
    is_contiguous() const
    {
        for (size_t i = 0; i < N; i++) {
            if (_M_offsets->data()[i] != 0) {
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

    container_type&
    container() const
    {
        if (!_M_data) {
            throw std::logic_error("tensor::container: empty container cannot be accessed");
        }
        return *_M_data;
    }

    container_pointer
    container_ptr() const
    {
        return _M_data;
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
        std::copy_n(_M_sizes->data(), N, layout.sizes);
        std::copy_n(_M_strides->data(), N, layout.strides);
        std::copy_n(_M_offsets->data(), N, layout.offsets);
        return layout;
    }

    iterator
    begin()
    {
        return iterator(*this);
    }

    const_iterator
    begin() const
    {
        return const_iterator(*this);
    }

    iterator
    end()
    {
        return iterator(*this, numel());
    }

    const_iterator
    end() const
    {
        return const_iterator(*this, numel());
    }

    auto
    at(std::size_t i) const
    {
        if (auto size0 = size(0); i >= size0) {
            std::out_of_range(std::format("tensor::at: index {} is out of tensor size {}", i, size0)
            );
        }

        auto data = data_ptr() + stride(0) * i + offset(0);
        auto sizes = _M_sizes->data() + 1;
        auto strides = _M_strides->data() + 1;
        auto offsets = _M_offsets->data() + 1;
        return tensor<T, N - 1, reference_memory_container<T>>(data, sizes, strides, offsets);
    }

    auto
    at(std::size_t i)
    {
        return const_cast<tensor const&>(*this).at(i);
    }

    /// Return a tensor minor from the current tensor. The returned tensor and input tensor share
    /// the same underlying container.
    ///
    /// \param slices the slices of the tensor minor.
    ///
    /// ```c++
    /// auto T = rand<float>({3, 4});
    /// const auto M = T[slice(0, 1), slice(1, 3)];
    /// ```
    template <convertible_to_slice... SliceTypes>
    tensor
    index_select(const SliceTypes&... slices) requires(sizeof...(slices) == N)
    {
        return const_cast<tensor const&>(*this).index_select(slices...);
    }

    /// Return a constant tensor minor from the current tensor. The returned tensor and input
    /// tensor share the same underlying container.
    ///
    /// \param slices the slices of the tensor minor.
    ///
    /// See also \ref tensor::index_select.
    template <convertible_to_slice... SliceTypes>
    const tensor
    index_select(const SliceTypes&... slices) const requires(sizeof...(slices) == N)
    {
        tensor t(_M_data);
        std::size_t dim = 0;

        ([&] {
            auto s = slice(slices);
            auto stop = std::min(s.stop.value_or(size(dim)), size(dim));
            auto start = std::min(s.start.value_or(0), stop);

            t.set_size(dim, stop - start);
            t.set_stride(dim, stride(dim));
            t.set_offset(dim, start * t.stride(dim));
            dim++;
        }(), ...);

        return t;
    }

    /// Returns a reference to the `indices`-th element of the tensor. The returned tensor and
    /// input tensor share the same underlying container.
    ///
    /// \param indices the indices of the element to access.
    ///
    /// ```c++
    /// auto T = rand<float>({3, 4});
    /// std::cout << T.value_select(0, 2) << std::endl;
    /// ```
    ///
    /// See also \ref tensor::operator[]
    template <convertible_to_index... IndexTypes>
    value_type&
    value_select(const IndexTypes&... indices) requires(sizeof...(indices) == N)
    {
        return const_cast<T&>(const_cast<tensor const&>(*this).value_select(indices...));
    }

    /// Return a constant reference to the `indices`-th element of the tensor. The returned tensor
    /// and input tensor shared the same underlying container.
    ///
    /// \param indices the indices of the element to access.
    ///
    /// See also \ref tensor::value_select.
    template <convertible_to_index... IndexTypes>
    const value_type&
    value_select(const IndexTypes&... indices) const requires(sizeof...(indices) == N)
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

    /// Return a new tensor that is a narrowed version of the current tensor. The returned
    /// tensor and input tensor share the same underlying container.
    ///
    /// \param dim the dimension along which to narrow.
    /// \param start index of the element to start the narrowed dimension from.
    /// \param length length of the narrowed dimension.
    ///
    /// ```c++
    /// auto T = rand<float>({3, 3});
    /// std::cout << T.narrow(0, 0, 2).sizes() << std::endl;
    /// // out: 2, 3
    /// ```
    tensor
    narrow(std::size_t dim, std::size_t start, std::size_t length) const
    {
        tensor t(_M_data);
        for (std::size_t i = 0; i < N; i++) {
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

    /// Returns a reference to the `indices`-th element of the tensor.
    ///
    /// \param indices the indices of the element to access.
    ///
    /// ```c++
    /// auto T = rand<float>({3, 4});
    /// std::cout << T[0, 2] << std::endl;
    /// ```
    template <convertible_to_index... IndexTypes>
    value_type&
    operator[](const IndexTypes&... indices) requires(sizeof...(indices) == N)
    {
        return value_select(indices...);
    }

    /// Returns a constant reference to the `indices`-th element of the tensor.
    template <convertible_to_index... IndexTypes>
    const value_type&
    operator[](const IndexTypes&... indices) const requires(sizeof...(indices) == N)
    {
        return value_select(indices...);
    }

    /// Returns a constant slice of the tensor.
    ///
    /// \param slices the minors of the tensor to access.
    ///
    /// ```c++
    /// auto T = rand<float>({10, 20, 3});
    /// std::cout << T[slice(1, 4), slice(2, 4), slice(0, 2)] << std::endl;
    /// ```
    template <convertible_to_slice... SliceTypes>
    const tensor
    operator[](const SliceTypes&... slices) const requires(sizeof...(slices) == N)
    {
        return index_select(slices...);
    }

    /// Returns a slice of the tensor.
    ///
    /// \param slices the minors of the tensor to access.
    template <convertible_to_slice... SliceTypes>
    tensor
    operator[](const SliceTypes&... slices) requires(sizeof...(slices) == N)
    {
        return index_select(slices...);
    }

    auto
    operator[](std::size_t i) requires(N > 1)
    {
        return at(i);
    }

    /// Returns a tensor with dimensions transposed. The dimension values should not exceed
    /// the dimensionality of the tensor you transpose.
    ///
    /// \param dims dimensions to transpose
    ///
    /// ```c++
    /// auto T = rand<float>({10, 4, 8, 128});
    /// std::cout << T.transpose({1, 0, 3, 2}) << std::endl;
    /// ```
    tensor
    transpose(const std::size_t (&&dims)[N]) const requires(N > 0)
    {
        tensor t(_M_data);
        for (std::size_t i = 0; i < N; i++) {
            t.set_size(i, size(dims[i]));
            t.set_stride(i, stride(dims[i]));
            t.set_offset(i, offset(dims[i]));
        }
        return t;
    }

    /// Returns a tensor with dimension transposed, expects 2-D tensor.
    ///
    /// See also `tensor::transpose`.
    tensor
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

        for (std::size_t i = 0; i < dim; i++) {
            sizes[i] = size(i);
        }
        for (std::size_t i = dim; i < N; i++) {
            sizes[i + 1] = size(i);
        }

        return view(std::move(sizes));
    }

    template <std::size_t M>
    tensor<T, M, Container>
    view(int (&&dims)[M]) const requires(M > 0)
    {
        const std::span<int, M> sizes(dims, M);
        return view(sizes);
    }

    template <std::size_t M>
    tensor<T, M, Container>
    view(const std::span<int, M> dims) const requires(M > 0)
    {
        std::size_t view_sizes[M];
        deduce_view_sizes(dims.data(), M, view_sizes);
        return view(std::span<std::size_t, M>(view_sizes));
    }

    template <std::size_t M>
    tensor<T, M, Container>
    view(const std::span<std::size_t, M> view_sizes) const requires(M > 0)
    {
        std::size_t view_strides[M];
        deduce_view_strides(view_sizes.data(), M, view_strides);

        auto t = tensor<T, M, Container>(view_sizes, _M_data);
        t.set_offset(0, container_offset());
        for (std::size_t dim = 0; dim < M; dim++) {
            t.set_stride(dim, view_strides[dim]);
        }

        return t;
    }

    /// Flattens the tensor by reshaping (creating a view) it into lower-dimensionality tensor.
    ///
    /// The resulting tensor dimensionality should be lesser then the flattenting tensor
    /// dimensionality (M ≤ N). The resulting tensor is always a view of the original tensor data,
    /// therefore method raises an exception if it is not possible to create a view for the tensor.
    ///
    /// ```c++
    /// auto T = rand<float>({2, 4, 8, 10});
    /// std::cout << T.flatten<2>().sizes() << std::endl;
    /// // out: 64, 10
    /// ```
    template <std::size_t M>
    tensor<T, M, Container>
    flatten() const requires(M <= N)
    {
        std::size_t sizes[M]{numel()};
        for (std::size_t i = 1; i <= M - 1; i++) {
            sizes[M - i] = size(N - i);
            sizes[0] /= sizes[M - i];
        }
        return view(std::span<std::size_t, M>(sizes, M));
    }

protected:
    container_pointer _M_data = nullptr;
    descriptor_pointer _M_sizes = nullptr;
    descriptor_pointer _M_strides = nullptr;
    descriptor_pointer _M_offsets = nullptr;

    // Make all specialization of the tensor friends to the current specialization.
    template <typename FriendT, std::size_t FriendN, contiguous_container FriendContainer>
    friend class tensor;

    void
    set_size(std::size_t dim, std::size_t i)
    {
        _M_sizes->data()[dim] = i;
    }

    void
    initialize()
    {
        auto allocator = random_memory_allocator<std::size_t>();
        _M_sizes = allocator.allocate(N);
        _M_strides = allocator.allocate(N);
        _M_offsets = allocator.allocate(N);
    }

    void
    initialize_strides()
    {
        _M_strides->data()[N - 1] = 1;
        for (std::size_t i = N - 2; i < N; --i) {
            _M_strides->data()[i] = _M_strides->data()[i + 1] * _M_sizes->data()[i + 1];
        }
    }

    void
    initialize(const std::span<std::size_t, N> sizes)
    {
        initialize(sizes.begin(), sizes.end());
        initialize_strides();
    }

    void
    initialize(std::size_t (&&sizes)[N])
    {
        initialize(std::span<std::size_t, N>(sizes, N));
    }

    template <std::forward_iterator ForwardIt>
    void
    initialize(ForwardIt first, ForwardIt last)
    {
        assert(std::distance(first, last) == N);

        initialize();
        std::copy(first, last, _M_sizes->data());
        initialize_strides();
    }

    tensor(const container_pointer& data)
    : _M_data(data)
    {
        initialize();
    }

    tensor(
        const container_pointer& data,
        descriptor_pointer&& shape,
        descriptor_pointer&& strides,
        descriptor_pointer&& offsets
    )
    : _M_data(data),
      _M_sizes(std::move(shape)),
      _M_strides(std::move(strides)),
      _M_offsets(std::move(offsets))
    {}

    tensor(T* data, std::size_t* shape, std::size_t* strides, std::size_t* offsets)
    : tensor(
          make_reference_container(data),
          make_reference_container(shape),
          make_reference_container(strides),
          make_reference_container(offsets)
      )
    {}
};


template <typename T, std::size_t N, contiguous_container Container, std::size_t M>
struct change_tensor_dimensions<tensor<T, N, Container>, M> {
    using type = tensor<T, M, Container>;
};


template <
    typename T,
    std::size_t N,
    contiguous_container ContainerIn,
    contiguous_container ContainerOut>
struct change_tensor_container<tensor<T, N, ContainerIn>, ContainerOut> {
    using type = tensor<T, N, ContainerOut>;
};


template <std::size_t N, contiguous_container Container>
tensor(std::size_t (&&)[N], const std::shared_ptr<Container>&)
    -> tensor<typename Container::value_type, N, Container>;


template <std::size_t N, immutable_tensor Tensor>
auto
flatten(Tensor tensor)
{
    return tensor.template flatten<N>();
}


template <typename T>
auto
scalar(const T& value)
{
    return tensor<T, 0, scalar_memory_container<T>>(T(value));
}


template <typename T, std::size_t N, allocator_t<T> Allocator>
auto
empty(std::size_t (&&sizes)[N], Allocator alloc)
{
    return tensor<T, N, typename Allocator::container_type>(std::move(sizes), alloc);
}


template <typename T, std::size_t N, hardware_allocator_t<void> Allocator>
auto
empty(std::size_t (&&sizes)[N], Allocator alloc)
{
    auto rebind_alloc = rebind_hardware_allocator<T, Allocator>(alloc);
    return empty<T>(std::move(sizes), rebind_alloc);
}


template <typename T, std::size_t N>
auto
empty(std::size_t (&&sizes)[N], hardware_accelerator& accelerator)
{
    using container_type = hardware_memory_container<T>;
    return empty<T, N, container_type>(std::move(sizes), accelerator.get_allocator());
}


template <typename T, std::size_t N> requires(N > 0)
auto
empty(std::size_t (&&sizes)[N])
{
    return empty<T>(std::move(sizes), random_memory_allocator<T>());
}


template <typename T, std::size_t N, std::forward_iterator InputIt> requires(N > 0)
auto
empty(InputIt begin, InputIt end)
{
    return tensor<T, N, random_memory_container<T>>(begin, end);
}


template <typename T, immutable_tensor Tensor, allocator_t<T> Allocator>
auto
empty_like(const Tensor& like, Allocator alloc)
{
    using container_type = Allocator::container_type;

    auto sizes = like.sizes();
    return tensor<T, Tensor::dim(), container_type>(sizes.begin(), sizes.end(), alloc);
}


template <typename T, immutable_tensor Tensor, hardware_allocator_t<void> Allocator>
auto
empty_like(const Tensor& like, Allocator alloc)
{
    auto rebind_alloc = rebind_hardware_allocator<T, Allocator>(alloc);
    return empty_like<T>(like, rebind_alloc);
}


template <immutable_tensor Tensor> requires(Tensor::dim() > 0)
auto
empty_like(const Tensor& like)
{
    using value_type = Tensor::value_type;

    auto sizes = like.sizes();
    return empty<value_type, Tensor::dim()>(sizes.begin(), sizes.end());
}


template <typename T, std::size_t N> requires(N > 0)
auto
full(std::size_t (&&sizes)[N], const T& fill_value)
{
    auto t = empty<T>(std::move(sizes));
    std::fill_n(t.data_ptr(), t.numel(), fill_value);
    return t;
}


template <typename T, std::size_t N, hardware_allocator_t<void> Allocator>
auto
full(std::size_t (&&sizes)[N], const T& fill_value, Allocator alloc)
{
    auto t = empty<T>(std::move(sizes), alloc);
    std::fill_n(t.data_ptr(), t.numel(), fill_value);
    return t;
}


template <typename T, std::size_t N> requires(N > 0)
auto
full(std::size_t (&&sizes)[N], const T& fill_value, hardware_accelerator& accelerator)
{
    auto t = empty<T>(std::move(sizes), accelerator);
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


template <typename T, std::size_t N, std::forward_iterator ForwardIt> requires(N > 0)
auto
to_tensor(std::size_t (&&sizes)[N], ForwardIt first, ForwardIt last)
{
    auto t = empty<T>(std::move(sizes));
    auto distance = std::distance(first, last);

    if (std::size_t(distance) != t.numel()) {
        throw std::invalid_argument(std::format(
            "tensor: iterators differences ({}) should be equal to tensor numel ({})", distance,
            t.numel()
        ));
    }

    std::copy(first, last, t.data_ptr());
    return t;
}


template <typename T, std::size_t N, contiguous_container Container>
auto
to_tensor(const tensor<T, N, Container>& t)
{
    auto tt = empty_like(t);
    std::copy_n(t.data_ptr(), t.numel(), tt.data_ptr());
    return tt;
}


template <immutable_tensor Tensor, hardware_allocator_t<void> Allocator>
auto
move(const Tensor& t, Allocator alloc)
{
    using T = typename Tensor::value_type;
    constexpr auto N = Tensor::dim();

    using allocator_type = rebind_hardware_allocator<T, Allocator>;
    using container_type = allocator_type::container_type;
    using tensor_type = tensor<T, N, container_type>;

    auto allocator = allocator_type(alloc);
    auto container = allocator.allocate(t.data_ptr(), t.numel());

    return tensor_type(t.sizes(), t.strides(), t.offsets(), container);
}


} //  namespace metalchat
