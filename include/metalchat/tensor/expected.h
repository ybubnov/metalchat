// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025-2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <exception>
#include <expected>
#include <format>
#include <source_location>
#include <sstream>

#include <metalchat/tensor/concept.h>
#include <metalchat/tensor/format.h>


namespace metalchat {


class matching_dim {
private:
    std::size_t _M_dim;
    std::size_t _M_size;
    std::source_location _M_source_location;

public:
    matching_dim(
        std::size_t dim, std::size_t size, std::source_location s = std::source_location::current()
    )
    : _M_dim(dim),
      _M_size(size),
      _M_source_location(s)
    {}

    template <immutable_tensor Tensor>
    std::exception_ptr
    operator()(const Tensor& t) const
    {
        if (const auto actual = t.size(_M_dim); actual != _M_size) {
            const auto source_text = format_source_location(_M_source_location);
            const auto actual_sizes = format_memory_container(t.accessor().sizes());

            const auto error_text = std::vformat(
                "{} unexpected dimension ({}) of tensor [{}] ({} != {})",
                std::make_format_args(source_text, _M_dim, actual_sizes, actual, _M_size)
            );
            return std::make_exception_ptr(std::invalid_argument(error_text));
        }

        return std::exception_ptr();
    }
};


class matching_last_dim {
private:
    std::size_t _M_dim;
    tensor_accessor _M_access;
    std::source_location _M_source_location;

public:
    template <immutable_tensor Tensor>
    matching_last_dim(const Tensor& t, std::source_location s = std::source_location::current())
    : _M_dim(t.dim() - 1),
      _M_access(t.accessor()),
      _M_source_location(s)
    {}

    template <immutable_tensor Tensor>
    std::exception_ptr
    operator()(const Tensor& t) const
    {
        auto expect = _M_access.size(_M_dim);

        if (const auto actual = t.sizes().back(); actual != expect) {
            const auto source_text = format_source_location(_M_source_location);
            const auto actual_sizes = format_memory_container(_M_access.sizes());
            const auto expect_sizes = format_memory_container(t.accessor().sizes());

            const auto error_text = std::vformat(
                "{} last dimensions of tensor1 [{}] and tensor2 [{}] are different ({} != {})",
                std::make_format_args(
                    source_text, _M_dim, actual_sizes, expect_sizes, actual, expect
                )
            );
            return std::make_exception_ptr(std::invalid_argument(error_text));
        }

        return std::exception_ptr();
    }
};


class matching_shape {
private:
    std::size_t _M_dim;
    tensor_accessor _M_access;
    std::source_location _M_source_location;

public:
    template <immutable_tensor Tensor>
    matching_shape(const Tensor& t, std::source_location s = std::source_location::current())
    : _M_dim(t.dim()),
      _M_access(t.accessor()),
      _M_source_location(s)
    {}

    template <immutable_tensor Tensor>
    std::exception_ptr
    operator()(const Tensor& t) const
    {
        if (t.dim() != _M_dim) {
            const auto source_text = format_source_location(_M_source_location);
            const auto sizes1 = format_memory_container(_M_access.sizes());
            const auto sizes2 = format_memory_container(t.accessor().sizes());

            const auto error_text = std::format(
                "{} tensor1 [{}] and tensor2 [{}] dimensions are different: ({}) != ({})",
                source_text, sizes1, sizes2, t.dim(), _M_dim
            );
            return std::make_exception_ptr(std::invalid_argument(error_text));
        }

        for (std::size_t i = 0; i < _M_dim; i++) {
            auto actual = t.size(i);
            auto expect = _M_access.size(i);

            if (actual != expect) {
                const auto source_text = format_source_location(_M_source_location);
                const auto sizes1 = format_memory_container(_M_access.sizes());
                const auto sizes2 = format_memory_container(t.accessor().sizes());

                const auto error_text = std::format(
                    "{} tensor shapes are different: [{}] != [{}] at the dimension ({})",
                    source_text, sizes1, sizes2, i
                );
                return std::make_exception_ptr(std::invalid_argument(error_text));
            }
        }

        return std::exception_ptr();
    }
};


class matching_numel {
private:
    std::size_t _M_dim;
    tensor_accessor _M_access;
    std::source_location _M_source_location;

public:
    template <immutable_tensor Tensor>
    matching_numel(const Tensor& t, std::source_location s = std::source_location::current())
    : _M_dim(t.dim()),
      _M_access(t.accessor()),
      _M_source_location(s)
    {}

    template <immutable_tensor Tensor>
    std::exception_ptr
    operator()(const Tensor& t) const
    {
        std::size_t expect = 1;
        for (std::size_t i = 0; i < _M_dim; i++) {
            expect *= _M_access.size(i);
        }

        if (const auto actual = t.numel(); actual != expect) {
            const auto source_text = format_source_location(_M_source_location);
            const auto sizes1 = format_memory_container(_M_access.sizes());
            const auto sizes2 = format_memory_container(t.accessor().sizes());

            const auto error_text = std::format(
                "{} tensor1 [{}] and tensor2 [{}] numels are different: ({}) != ({})", source_text,
                sizes1, sizes2, actual, expect
            );
            return std::make_exception_ptr(std::invalid_argument(error_text));
        }

        return std::exception_ptr();
    }
};


/// The tensor coupled with expectations about the shape, size, dimensionality, etc. This
/// type is used to chain tensor assertions into \ref immutable_tensor concept, so it could
/// be used just a regular tensor in the kernel or nn library.
///
/// \tparam Tensor a wrapped tensor type.
///
/// Consider the following example where a tensor `T2` is wrapped into an expected tensor:
/// ```cpp
/// auto T1 = full<float>({2, 4}, 1.0f);
/// auto T2 = full<float>({3, 4}, 2.0f);
///
/// auto E = expected_tensor(T)
///     .expect(matching_dim(1, T2.size(1)))
///     .expect(matching_numel(T2));
///
/// // throws `std::invalid_argument` exception since T1 and T2
/// // tensors have different number of elements (numel).
/// E.value();
/// ```
template <immutable_tensor Tensor> class expected_tensor {
    using expected_type = std::expected<Tensor, std::exception_ptr>;
    using unexpected_type = std::unexpected<std::exception_ptr>;

public:
    using tensor_type = Tensor;
    using error_type = std::exception_ptr;

    static constexpr std::size_t N = tensor_type::dim();

    using value_type = tensor_type::value_type;

    using pointer_type = tensor_type::pointer_type;

    using accessor_type = tensor_accessor;

    using container_type = tensor_type::container_type;

    using container_pointer = tensor_type::container_pointer;

    using iterator = tensor_type::iterator;

    using const_iterator = tensor_type::const_iterator;

    /// Create an \ref expected_tensor by acquiring ownership of another tensor instance.
    expected_tensor(tensor_type&& t)
    : _M_value(std::move(t))
    {}

    /// Create an \ref expected_tensor by copying another tensor instance.
    expected_tensor(const tensor_type& t)
    : _M_value(t)
    {}

    /// Add a new expectation to the tensor.
    ///
    /// The expectation is executed immediately if the tensor has value,
    /// otherwise the expectation is never executed.
    ///
    /// \tparam Expectation a type of the tensor expectation.
    /// \param expectation a tensor expectation to assert.
    template <typename Expectation>
    expected_tensor&
    expect(const Expectation& expectation)
    {
        if (!_M_value.has_value()) {
            return *this;
        }
        if (auto eptr = expectation(*_M_value); eptr) {
            _M_value = unexpected_type(eptr);
        }
        return *this;
    }

    /// See \ref tensor::dim.
    static constexpr std::size_t
    dim()
    {
        return tensor_type::dim();
    }

    explicit
    operator bool() const noexcept
    {
        return _M_value;
    }

    tensor_type&
    value()
    {
        if (!_M_value.has_value()) {
            std::rethrow_exception(_M_value.error());
        }
        return _M_value.value();
    }

    const tensor_type&
    value() const
    {
        if (!_M_value.has_value()) {
            std::rethrow_exception(_M_value.error());
        }
        return _M_value.value();
    }

    tensor_type&
    operator*()
    {
        return value();
    }

    const tensor_type&
    operator*() const
    {
        return value();
    }

    /// See \ref tensor::numel.
    std::size_t
    numel() const
    {
        return value().numel();
    }

    const accessor_type&
    accessor() const
    {
        return value().accessor();
    }

    accessor_type&
    accessor()
    {
        return value().accessor();
    }

    container_type&
    container() const
    {
        return value().container();
    }

    std::shared_ptr<basic_container>
    container_ptr() const
    {
        return value().container_ptr();
    }

    void
    set_container(const std::shared_ptr<basic_container>& ptr)
    {
        value().set_container(ptr);
    }

    /// See \ref tensor::data_ptr.
    pointer_type
    data_ptr()
    {
        return value().data_ptr();
    }

    /// See \ref tensor::data_ptr.
    const pointer_type
    data_ptr() const
    {
        return value().data_ptr();
    }

    /// See \ref tensor::size.
    std::size_t
    size(std::size_t dim) const
    {
        return value().size(dim);
    }

    /// See \ref tensor::sizes.
    const std::span<std::size_t>
    sizes() const
    {
        return value().sizes();
    }

    /// See \ref tensor::shape.
    const std::span<std::size_t, N>
    shape() const
    {
        return value().shape();
    }

    /// See \ref tensor::stride.
    std::size_t
    stride(std::size_t dim) const
    {
        return value().stride(dim);
    }

    /// See \ref tensor::strides.
    const std::span<std::size_t>
    strides() const
    {
        return value().strides();
    }

    /// See \ref tensor::offset.
    std::size_t
    offset(std::size_t dim) const
    {
        return value().offset(dim);
    }

    /// See \ref tensor::offsets.
    const std::span<std::size_t>
    offsets() const
    {
        return value().offsets();
    }

    /// See \ref tensor::begin.
    iterator
    begin()
    {
        return value().begin();
    }

    /// See \ref tensor::end.
    iterator
    end()
    {
        return value().end();
    }

    /// See \ref tensor::begin.
    const_iterator
    begin() const
    {
        return value().begin();
    }

    /// See \ref tensor::end.
    const_iterator
    end() const
    {
        return value().end();
    }

    /// See \ref tensor::index_select.
    template <convertible_to_slice... SliceTypes>
    auto
    index_select(const SliceTypes&... slices) requires(sizeof...(slices) == N)
    {
        return expected_tensor(value().index_select(slices...));
    }

    /// See \ref tensor::expand_dims.
    auto
    expand_dims(std::size_t dim) const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, N + 1>;
        return expected_tensor<tensor_t>(value().expand_dims(dim));
    }

    /// See \ref tensor::view.
    template <std::size_t M>
    auto
    view(int (&&dims)[M]) const requires(M > 0)
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return expected_tensor<tensor_t>(value().view(std::move(dims)));
    }

    /// See \ref tensor::view.
    template <std::size_t M>
    auto
    view(const std::span<int, M> dims) const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return expected_tensor<tensor_t>(value().view(dims));
    }

    /// See \ref tensor::view.
    template <std::size_t M>
    auto
    view(const std::span<std::size_t, M> dims) const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return expected_tensor<tensor_t>(value().view(dims));
    }

    /// See \ref tensor::flatten.
    template <std::size_t M>
    auto
    flatten() const
    {
        using tensor_t = change_tensor_dimensions_t<tensor_type, M>;
        return expected_tensor<tensor_t>(value().template flatten<M>());
    }

    /// See \ref tensor::transpose.
    expected_tensor
    narrow(std::size_t dim, std::size_t start, std::size_t length) const
    {
        return expected_tensor(value().narrow(dim, start, length));
    }

    /// See \ref tensor::transpose.
    expected_tensor
    transpose(const std::size_t (&&dims)[N]) const
    {
        return expected_tensor(value().transpose(std::move(dims)));
    }

    /// See \ref tensor::layout.
    tensor_layout<N>
    layout() const
    {
        return value().layout();
    }

private:
    std::expected<tensor_type, error_type> _M_value;
};


} // namespace metalchat
