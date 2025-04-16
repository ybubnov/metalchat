#pragma once

#include <iterator>
#include <optional>

#include <metalchat/container.h>
#include <metalchat/tensor_concept.h>


namespace metalchat {


template <typename It, typename T>
concept forward_tensor_iterator_t = std::forward_iterator<It> && requires(It it) {
    typename std::iterator_traits<It>::value_type;

    // Ensure that iterator yields tensors.
    requires immutable_tensor_t<typename std::iterator_traits<It>::value_type, T>;
};


template <typename T, std::size_t N> class tensor_iterator {
public:
    using iterator_category = std::forward_iterator_tag;

    using iterator = tensor_iterator<T, N>;

    using value_type = T;

    using reference = value_type&;

    using pointer = value_type*;

    using difference_type = std::ptrdiff_t;

    tensor_iterator(
        memory_container<T>& data,
        memory_container<std::size_t>& sizes,
        memory_container<std::size_t>& strides,
        memory_container<std::size_t>& offsets,
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
    operator==(const iterator& rhs)
    {
        return ((m_data.get().data() == rhs.m_data.get().data()) && (m_num == rhs.m_num));
    }

    bool
    operator!=(const iterator& rhs)
    {
        return !operator==(rhs);
    }

private:
    std::reference_wrapper<memory_container<T>> m_data;
    std::reference_wrapper<memory_container<std::size_t>> m_sizes;
    std::reference_wrapper<memory_container<std::size_t>> m_strides;
    std::reference_wrapper<memory_container<std::size_t>> m_offsets;

    std::size_t m_index;
    std::size_t m_num;

    std::array<std::size_t, N> m_indices;

    inline std::size_t
    size(std::size_t dim)
    {
        return m_sizes.get().data()[dim];
    }

    inline std::size_t
    stride(std::size_t dim)
    {
        return m_strides.get().data()[dim];
    }

    inline std::size_t
    offset(std::size_t dim)
    {
        return m_offsets.get().data()[dim];
    }

    inline reference
    data(std::size_t index)
    {
        return m_data.get().data()[index];
    }

    std::size_t
    next()
    {
        std::size_t index = 0;
        for (std::size_t i = 0; i < N; i++) {
            index = index + stride(i) * m_indices[i] + offset(i);
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
