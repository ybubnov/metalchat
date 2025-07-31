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
private:

public:
    using value_type = T;

    using iterator_category = std::forward_iterator_tag;

    using iterator = tensor_iterator<T, N>;

    using reference = value_type&;

    using pointer = value_type*;

    using difference_type = std::ptrdiff_t;

    using container_pointer = std::shared_ptr<memory_container<value_type>>;

    template <immutable_tensor Tensor>
    tensor_iterator(const Tensor& tensor, std::optional<std::size_t> start = std::nullopt)
    : _m_data(tensor.container_ptr()),
      _m_sizes(tensor.sizes().begin(), tensor.sizes().end()),
      _m_strides(tensor.strides().begin(), tensor.strides().end()),
      _m_offsets(tensor.offsets().begin(), tensor.offsets().end()),
      _m_index(0),
      _m_num(0),
      _m_indices({})
    {
        if (start) {
            auto start_num = start.value() - 1;
            _m_num = start_num + 1;

            // Calculate the total number of elements in the given tensor.
            std::size_t numel = 1;
            for (std::size_t i = 0; i < N; i++) {
                numel *= _m_sizes[i];
            }

            // Generate the index of the element in multidimensional tensor, so
            // that increment operator could start from the correct position.
            for (std::size_t i = 0; i < N; i++) {
                numel = numel / _m_sizes[i];
                _m_indices[i] = start_num / numel;
                start_num = start_num % numel;
            }
        } else {
            _m_index = next();
        }
    }

    tensor_iterator(const iterator& it)
    : _m_data(it._m_data),
      _m_sizes(it._m_sizes),
      _m_strides(it._m_strides),
      _m_offsets(it._m_offsets),
      _m_index(it._m_index),
      _m_num(it._m_num),
      _m_indices(it._m_indices)
    {}

    iterator&
    operator++()
    {
        _m_index = next();
        _m_num++;
        return *this;
    }

    reference
    operator*()
    {
        return data(_m_index);
    }

    pointer
    operator->()
    {
        return &data(_m_index);
    }

    bool
    operator==(const iterator& rhs)
    {
        return (
            (_m_data == nullptr && rhs._m_data == nullptr)
            || ((_m_data != nullptr) && (rhs._m_data != nullptr)
                && (_m_data->data() == rhs._m_data->data()) && (_m_num == rhs._m_num))
        );
    }

    bool
    operator!=(const iterator& rhs)
    {
        return !operator==(rhs);
    }

private:
    container_pointer _m_data;
    std::vector<std::size_t> _m_sizes;
    std::vector<std::size_t> _m_strides;
    std::vector<std::size_t> _m_offsets;

    std::size_t _m_index;
    std::size_t _m_num;

    std::array<std::size_t, N> _m_indices;

    inline reference
    data(std::size_t index)
    {
        return _m_data->data()[index];
    }

    std::size_t
    next()
    {
        std::size_t index = 0;
        for (std::size_t i = 0; i < N; i++) {
            index = index + _m_strides[i] * _m_indices[i] + _m_offsets[i];
        }

        // Update indices in the array.
        std::size_t carry = 1;
        for (std::size_t i = N - 1; i < N; i--) {
            auto sum = _m_indices[i] + carry;
            _m_indices[i] = sum % _m_sizes[i];
            carry = sum / _m_sizes[i];
        }

        return index;
    }
};


} // namespace metalchat
