#pragma once


#include <optional>


#include <metalchat/container.h>


namespace metalchat {


template <typename T, std::size_t N> class tensor_iterator {
public:
    using iterator = tensor_iterator<T, N>;

    using value_type = T;

    using reference = value_type&;

    using pointer = value_type*;

    tensor_iterator(
        array_ref<T>& data,
        array_ref<std::size_t>& sizes,
        array_ref<std::size_t>& strides,
        array_ref<std::size_t>& offsets,
        std::optional<std::size_t> start = std::nullopt
    )
    : m_data(data),
      m_sizes(sizes),
      m_strides(strides),
      m_offsets(offsets),
      m_index(0),
      m_num(0),
      m_indices({0})
    {
        if (start) {
            auto start_num = start.value() - 1;
            m_num = start_num + 1;

            // Calculate the total number of elements in the given tensor.
            std::size_t numel = 1;
            for (std::size_t i = 0; i < N; i++) {
                numel *= size(i);
            }

            // Generate the index of the element in multidimensional tensor, so
            // that increment operator could start from the correct position.
            for (std::size_t i = 0; i < N; i++) {
                numel = numel / size(i);
                m_indices[i] = start_num / numel;
                start_num = start_num % numel;
            }
        } else {
            m_index = next();
        }
    }

    tensor_iterator(const iterator& it)
    : m_data(it.m_data),
      m_sizes(it.m_sizes),
      m_strides(it.m_strides),
      m_offsets(it.m_offsets),
      m_index(it.m_index),
      m_num(it.m_num),
      m_indices(it.m_indices)
    {}

    iterator&
    operator++()
    {
        m_index = next();
        m_num++;
        return *this;
    }

    reference
    operator*()
    {
        return data(m_index);
    }

    pointer
    operator->()
    {
        return &data(m_index);
    }

    bool
    operator!=(const iterator& rhs)
    {
        return !((m_data.data() == rhs.m_data.data()) && (m_num == rhs.m_num));
    }

private:
    array_ref<T>& m_data;
    array_ref<std::size_t>& m_sizes;
    array_ref<std::size_t>& m_strides;
    array_ref<std::size_t>& m_offsets;

    std::size_t m_index;
    std::size_t m_num;

    std::array<std::size_t, N> m_indices;

    inline std::size_t
    size(std::size_t dim)
    {
        return m_sizes.data()[dim];
    }

    inline std::size_t
    stride(std::size_t dim)
    {
        return m_strides.data()[dim];
    }

    inline std::size_t
    offset(std::size_t dim)
    {
        return m_offsets.data()[dim];
    }

    inline reference
    data(std::size_t index)
    {
        return m_data.data()[index];
    }

    std::size_t
    next()
    {
        std::size_t index = 0;
        for (std::size_t i = 0; i < N; i++) {
            index = index + stride(i) * (offset(i) + m_indices[i]);
        }

        // Update indices in the array.
        std::size_t carry = 1;
        for (std::size_t i = N - 1; i < N; i--) {
            auto sum = m_indices[i] + carry;
            m_indices[i] = sum % size(i);
            carry = sum / size(i);
        }

        return index;
    }
};


} // namespace metalchat
