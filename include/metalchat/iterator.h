#pragma once


#include <metalchat/container.h>


namespace metalchat {


template <typename T, std::size_t N>
    requires(N > 0)
class tensor_iterator {
public:
    using iterator = tensor_iterator<T, N>;

    using value_type = T;

    using reference = value_type&;

    using pointer = value_type*;

    tensor_iterator(
        array_ref<T>& data,
        array_ref<std::size_t>& sizes,
        array_ref<std::size_t>& strides,
        array_ref<std::size_t>& offsets
    )
    : m_data(data),
      m_sizes(sizes),
      m_strides(strides),
      m_offsets(offsets),
      m_index(0),
      m_numel(1),
      m_num(0),
      m_indices({0})
    {
        for (std::size_t i = 0; i < N; i++) {
            m_numel *= size(i);
        }
        m_index = next();
    }

    iterator&
    operator++()
    {
        if (m_num >= m_numel) {
            return *this;
        }

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
        return m_num != rhs.m_numel;
    }

    iterator
    end()
    {
        m_num = m_numel;
        return *this;
    }

private:
    array_ref<T>& m_data;
    array_ref<std::size_t>& m_sizes;
    array_ref<std::size_t>& m_strides;
    array_ref<std::size_t>& m_offsets;

    std::size_t m_index;
    std::size_t m_numel;
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
        for (std::size_t i = N - 1; i < N; i--) {
            if (m_indices[i] + 1 < size(i)) {
                m_indices[i] += 1;
                break;
            }
            m_indices[i] = 0;
        }

        return index;
    }
};


} // namespace metalchat
