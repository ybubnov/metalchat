// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <iterator>
#include <optional>

#include <metalchat/container.h>
#include <metalchat/tensor/accessor.h>
#include <metalchat/tensor/concept.h>


namespace metalchat {


template <typename It, typename T>
concept forward_tensor_iterator_t = std::forward_iterator<It> && requires(It it) {
    typename std::iterator_traits<It>::value_type;

    // Ensure that iterator yields tensors.
    requires immutable_tensor_t<typename std::iterator_traits<It>::value_type, T>;
};


/// An iterator of multi-dimensional matrix returning elements in a row-major (lexicographic)
/// order.
///
/// \tparam T data type of the tensor elements (float, int, etc.)
/// \tparam N dimensionality of the underlying data container.
///
/// Usually \ref tensor_iterator is constructed using methods \ref tensor::begin and
/// \ref tensor::end. But could also be used as a standalone entity.
///
/// ```c++
/// auto T = rand<float>({3, 4});
/// for (auto it = T.begin(); it != T.end(); ++it) {
///     std::cout << (*it) << std::endl;
/// }
/// ```
template <typename T, std::size_t N> class tensor_iterator {
public:
    /// Alias of the tensor type.
    using value_type = T;

    /// Declares that \ref tensor_iterator is a forward iterator.
    using iterator_category = std::forward_iterator_tag;

    /// Iterator type.
    using iterator = tensor_iterator<T, N>;

    /// Reference type of the tensor element.
    using reference = value_type&;

    /// Pointer type of the tensor element.
    using pointer = value_type*;

    /// Specifies a distance type between two iterators.
    using difference_type = std::ptrdiff_t;

    /// Pointer to the container type storing the data of the tensor.
    using container_pointer = std::shared_ptr<basic_container>;

    /// Creates a new \ref tensor_iterator instance.
    ///
    /// Initializes iterator with a pointer to the tensor's container. Therefore tensor iterator
    /// is still valid even after destruction of the original tensor.
    template <immutable_tensor Tensor>
    tensor_iterator(const Tensor& tensor, difference_type start)
    : _M_data(tensor.container_ptr()),
      _M_access(tensor.accessor().copy()),
      _M_index(0),
      _M_overflow(0),
      _M_numel(tensor.numel()),
      _M_indices({})
    {
        start = std::min(start, static_cast<difference_type>(_M_numel));
        std::tie(_M_index, _M_overflow) = advance(start);
    }

    /// Creates a new \ref tensor_iterator that points to the first element of the tensor.
    template <immutable_tensor Tensor>
    tensor_iterator(const Tensor& tensor)
    : tensor_iterator(tensor, 0)
    {}

    /// The default constructor of \ref tensor_iterator.
    tensor_iterator(const iterator& it) = default;

    /// Advances the iterator forward.
    iterator&
    operator++()
    {
        if (_M_overflow == 0) {
            std::tie(_M_index, _M_overflow) = next();
        }
        return *this;
    }

    /// Advances the iterator forward by `n` elements.
    iterator&
    operator+(difference_type n)
    {
        std::tie(_M_index, _M_overflow) = advance(n);
        return *this;
    }

    /// Returns a reference to the current tensor element.
    ///
    /// The end-of-data iterator dereference always returns the first element of the tensor.
    reference
    operator*()
    {
        return data(_M_index);
    }

    /// Returns a pointer to the current tensor element.
    ///
    /// The end-of-data iterator dereference always returns the first element of the tensor.
    pointer
    operator->()
    {
        return &data(_M_index);
    }

    /// Compares two tensor_iterators.
    ///
    /// Method compares data pointers of the underlying tensor containers and current position
    /// of the iterator. Note that iterators are considered equal even when they are traversing
    /// tensor data with a different view (different strides and offsets).
    bool
    operator==(const iterator& rhs) const
    {
        if ((_M_data == nullptr) & (rhs._M_data == nullptr)) {
            return true;
        }
        if ((_M_data == nullptr) | (rhs._M_data == nullptr)) {
            return false;
        }

        bool same = true;
        same &= (_M_data->data_ptr() == rhs._M_data->data_ptr());
        same &= (_M_data->size() == rhs._M_data->size());
        same &= (_M_index == rhs._M_index);
        same &= (_M_overflow == rhs._M_overflow);
        same &= (_M_numel == rhs._M_numel);

        return same;
    }

    /// Compares two tensor_iterators.
    ///
    /// The implementation is negation of the \ref tensor_iterator::operator==.
    bool
    operator!=(const iterator& rhs) const
    {
        return !operator==(rhs);
    }

private:
    container_pointer _M_data;
    tensor_accessor _M_access;
    difference_type _M_index;
    difference_type _M_overflow;
    std::size_t _M_numel;

    std::array<std::size_t, N> _M_indices;

    inline reference
    data(std::size_t index)
    {
        return static_cast<value_type*>(_M_data->data_ptr())[index];
    }

    std::tuple<difference_type, difference_type>
    next()
    {
        return advance(_M_overflow + 1);
    }

    difference_type
    index() const
    {
        difference_type ind = 0;
        for (std::size_t i = 0; i < N; i++) {
            ind = ind + _M_access.stride(i) * _M_indices[i] + _M_access.offset(i);
        }
        return ind;
    }

    std::tuple<difference_type, difference_type>
    advance(difference_type distance)
    {
        for (std::size_t i = N - 1; i < N; i--) {
            auto sum = _M_indices[i] + distance;
            _M_indices[i] = sum % _M_access.size(i);
            distance = sum / _M_access.size(i);
        }
        return std::make_tuple(index(), std::min(distance, difference_type(1)));
    }
};


template <immutable_tensor Tensor>
tensor_iterator(const Tensor&, std::ptrdiff_t)
    -> tensor_iterator<typename Tensor::value_type, Tensor::dim()>;


} // namespace metalchat
