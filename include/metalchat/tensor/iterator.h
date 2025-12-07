// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <iterator>
#include <optional>

#include <metalchat/container.h>
#include <metalchat/tensor/concept.h>


namespace metalchat {


template <typename It, typename T>
concept forward_tensor_iterator_t = std::forward_iterator<It> && requires(It it) {
    typename std::iterator_traits<It>::value_type;

    // Ensure that iterator yields tensors.
    requires immutable_tensor_t<typename std::iterator_traits<It>::value_type, T>;
};


template <typename T, std::size_t N> class tensor_iterator {
public:
    using value_type = T;

    using iterator_category = std::forward_iterator_tag;

    using iterator = tensor_iterator<T, N>;

    using reference = value_type&;

    using pointer = value_type*;

    using difference_type = std::ptrdiff_t;

    using container_pointer = std::shared_ptr<basic_container>;

    template <immutable_tensor Tensor>
    tensor_iterator(const Tensor& tensor, std::optional<std::size_t> start = std::nullopt)
    : _M_data(tensor.container_ptr()),
      _M_sizes(tensor.sizes().begin(), tensor.sizes().end()),
      _M_strides(tensor.strides().begin(), tensor.strides().end()),
      _M_offsets(tensor.offsets().begin(), tensor.offsets().end()),
      _M_index(0),
      _M_num(0),
      _M_indices({})
    {
        if (start) {
            auto start_num = start.value() - 1;
            _M_num = start_num + 1;

            // Calculate the total number of elements in the given tensor.
            std::size_t numel = 1;
            for (std::size_t i = 0; i < N; i++) {
                numel *= _M_sizes[i];
            }

            // Generate the index of the element in multidimensional tensor, so
            // that increment operator could start from the correct position.
            for (std::size_t i = 0; i < N; i++) {
                numel = numel / _M_sizes[i];
                _M_indices[i] = start_num / numel;
                start_num = start_num % numel;
            }
        } else {
            _M_index = next();
        }
    }

    tensor_iterator(const iterator& it) = default;

    iterator&
    operator++()
    {
        _M_index = next();
        _M_num++;
        return *this;
    }

    reference
    operator*()
    {
        return data(_M_index);
    }

    pointer
    operator->()
    {
        return &data(_M_index);
    }

    bool
    operator==(const iterator& rhs) const
    {
        return (
            (_M_data == nullptr && rhs._M_data == nullptr) ||
            ((_M_data != nullptr) && (rhs._M_data != nullptr) &&
             (_M_data->data_ptr() == rhs._M_data->data_ptr()) && (_M_num == rhs._M_num))
        );
    }

    bool
    operator!=(const iterator& rhs) const
    {
        return !operator==(rhs);
    }

private:
    container_pointer _M_data;
    std::vector<std::size_t> _M_sizes;
    std::vector<std::size_t> _M_strides;
    std::vector<std::size_t> _M_offsets;

    std::size_t _M_index;
    std::size_t _M_num;

    std::array<std::size_t, N> _M_indices;

    inline reference
    data(std::size_t index)
    {
        return static_cast<value_type*>(_M_data->data_ptr())[index];
    }

    std::size_t
    next()
    {
        std::size_t index = 0;
        for (std::size_t i = 0; i < N; i++) {
            index = index + _M_strides[i] * _M_indices[i] + _M_offsets[i];
        }

        // Update indices in the array.
        std::size_t carry = 1;
        for (std::size_t i = N - 1; i < N; i--) {
            auto sum = _M_indices[i] + carry;
            _M_indices[i] = sum % _M_sizes[i];
            carry = sum / _M_sizes[i];
        }

        return index;
    }
};


} // namespace metalchat
