// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <expected>
#include <filesystem>
#include <format>
#include <source_location>
#include <sstream>

#include <metalchat/tensor/concept.h>
#include <metalchat/tensor/format.h>


namespace metalchat {


template <immutable_tensor Tensor, typename Error = std::invalid_argument> class expected_tensor {
    using expected_type = std::expected<Tensor, Error>;
    using unexpected_type = std::unexpected<Error>;

public:
    using tensor_type = Tensor;
    using error_type = Error;

    expected_tensor(tensor_type&& t)
    : _M_value(std::move(t))
    {}

    expected_tensor(const tensor_type& t)
    : _M_value(t)
    {}

    template <immutable_tensor Other>
    expected_tensor&
    same_dim(
        const Other& other,
        std::size_t lhs,
        std::size_t rhs,
        std::source_location source_location = std::source_location::current()
    )
    {
        if (!_M_value.has_value()) {
            return *this;
        }

        const auto actual = _M_value->size(lhs);
        const auto expect = other.size(rhs);

        if (actual != expect) {
            const auto source_text = format_source_location(source_location);
            const auto error_text = std::vformat(
                "{} dim ({}) of tensor1 is different from dim ({}) of tensor2 ({} != {})",
                std::make_format_args(source_text, lhs, rhs, actual, expect)
            );

            _M_value = unexpected_type(error_type(error_text));
        }

        return *this;
    }

    template <immutable_tensor Other>
    expected_tensor&
    same_shape(
        const Other& other, std::source_location source_location = std::source_location::current()
    )
    {
        if (!_M_value.has_value()) {
            return *this;
        }

        auto actual = _M_value->sizes();
        auto expect = other.sizes();

        if (actual.size() != expect.size()) {
            const auto source_text = format_source_location(source_location);
            const auto error_text = std::format(
                "{} tensor shapes are different: ({}) != ({})", source_text, format_span(actual),
                format_span(expect)
            );

            _M_value = unexpected_type(error_type(error_text));
            return *this;
        }

        for (std::size_t i = 0; i < expect.size(); i++) {
            if (actual[i] != expect[i]) {
                const auto source_text = format_source_location(source_location);
                const auto error_text = std::format(
                    "{} tensors sizes are different: ({}) != ({})", source_text,
                    format_span(actual), format_span(expect)
                );

                _M_value = unexpected_type(error_type(error_text));
                return *this;
            }
        }

        return *this;
    }

    template <immutable_tensor Other>
    expected_tensor&
    same_dim(
        const Other& other,
        std::size_t dim,
        std::source_location source_location = std::source_location::current()
    )
    {
        return same_dim(other, dim, dim, source_location);
    }


    template <immutable_tensor Other>
    expected_tensor&
    same_last_dim(
        const Other& other, std::source_location source_location = std::source_location::current()
    )
    {
        return same_dim(other, _M_value->dim() - 1, other.dim() - 1, source_location);
    }

    template <immutable_tensor Other>
    expected_tensor&
    same_first_dim(
        const Other& other, std::source_location source_location = std::source_location::current()
    )
    {
        return same_dim(other, 0, 0, source_location);
    }

    expected_tensor&
    same_dim(
        std::size_t lhs,
        std::size_t expect,
        std::source_location source_location = std::source_location::current()
    )
    {
        if (!_M_value.has_value()) {
            return *this;
        }

        if (_M_value->size(lhs) != expect) {
            const auto source_text = format_source_location(source_location);
            const auto error_text = std::format(
                "{} tensor shape is not as expected: ({}) != ({})", source_text,
                _M_value->size(lhs), expect
            );

            _M_value = unexpected_type(error_type(error_text));
        }

        return *this;
    }

    template <immutable_tensor Other>
    expected_tensor&
    same_numel(
        const Other& other, std::source_location source_location = std::source_location::current()
    )
    {
        if (!_M_value.has_value()) {
            return *this;
        }

        const auto actual = _M_value->numel();
        const auto expect = other.numel();

        if (actual != expect) {
            const auto source_text = format_source_location(source_location);
            const auto error_text = std::vformat(
                "{} numel of tensors are different ({}!={})",
                std::make_format_args(source_text, actual, expect)
            );

            _M_value = unexpected_type(error_type(error_text));
        }

        return *this;
    }

    tensor_type&&
    value()
    {
        if (!_M_value.has_value()) {
            throw _M_value.error();
        }

        return std::move(_M_value.value());
    }

private:
    std::string
    format_source_location(const std::source_location& source_location) const
    {
        const auto source_filepath = std::filesystem::path(source_location.file_name());
        const auto source_file = source_filepath.filename().string();
        const auto source_line = source_location.line();

        return std::vformat("{}#{}:", std::make_format_args(source_file, source_line));
    }

    std::string
    format_span(const std::span<std::size_t>& span) const
    {
        std::stringstream formatted_span;
        formatted_span << span;
        return formatted_span.str();
    }

    std::expected<Tensor, Error> _M_value;
};


} // namespace metalchat
