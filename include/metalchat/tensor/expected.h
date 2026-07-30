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


template <immutable_tensor Tensor> class expected_tensor {
    using expected_type = std::expected<Tensor, std::exception_ptr>;
    using unexpected_type = std::unexpected<std::exception_ptr>;

public:
    using tensor_type = Tensor;
    using error_type = std::exception_ptr;

    expected_tensor(tensor_type&& t)
    : _M_value(std::move(t))
    {}

    expected_tensor(const tensor_type& t)
    : _M_value(t)
    {}

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

    tensor_type&&
    value()
    {
        if (!_M_value.has_value()) {
            std::rethrow_exception(_M_value.error());
        }

        return std::move(_M_value.value());
    }

private:
    std::expected<tensor_type, error_type> _M_value;
};


} // namespace metalchat
