// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <expected>
#include <format>

#include <metalchat/tensor/concept.h>


namespace metalchat {


template <immutable_tensor Tensor, typename Error = std::invalid_argument> class expected_tensor {
    using expected_type = std::expected<Tensor, Error>;
    using unexpected_type = std::unexpected<Error>;

public:
    using tensor_type = Tensor;
    using error_type = Error;

    expected_tensor(tensor_type&& t)
    : _M_value(std::move(t)),
      _M_origin()
    {}

    expected_tensor(const tensor_type& t)
    : _M_value(t),
      _M_origin()
    {}

    expected_tensor&
    origin(const std::string& o)
    {
        _M_origin = o;
        return *this;
    }

    template <immutable_tensor Other>
    expected_tensor&
    expect_same_dimension(
        const Other& other, std::size_t lhs, std::size_t rhs, const std::string& message
    )
    {
        const auto expected_dim = other.size(rhs);
        const auto origin = _M_origin;

        _M_value = _M_value.and_then([=](tensor_type& value) -> expected_type {
            const auto actual_dim = value.size(lhs);

            if (actual_dim != expected_dim) {
                auto error_text = std::format("{}: {}!={}", origin, actual_dim, expected_dim);
                return unexpected_type(error_type(error_text));
            }

            return expected_type(value);
        });

        return *this;
    }

    tensor_type&&
    value()
    {
        return std::move(_M_value.value());
    }

private:
    std::expected<Tensor, Error> _M_value;
    std::string _M_origin;
};


} // namespace metalchat
