#pragma once

#include <array>
#include <cstddef>
#include <iomanip>
#include <sstream>


#include <metalama/format.h>


template <typename T>
struct unmanaged_ptr_traits {
    using ptr_type = T*;

    ptr_type data;

    unmanaged_ptr_traits(T* data_)
    : data(data_)
    {}
};

template <typename T>
struct managed_ptr_traits {
    using ptr_type = T*;

    ptr_type data;
    managed_ptr_traits(T* data_)
    : data(data_)
    {}

    ~managed_ptr_traits()
    {
        delete[] data;
        data = nullptr;
    }
};


template <typename T, std::size_t N, template <typename U> class ptr_traits = unmanaged_ptr_traits>
class tensor_base {
protected:
    ptr_traits<T> ptr;
    ptr_traits<const std::size_t> shape_ptr;
    ptr_traits<const std::size_t> strides_ptr;

public:
    tensor_base(T* data, const std::size_t* shape, const std::size_t* strides)
    : ptr(data),
      shape_ptr(shape),
      strides_ptr(strides)
    {}

    tensor_base(tensor_base&& t)
    {
        ptr = t.ptr;
        shape_ptr = t.shape_ptr;
        strides_ptr = t.strides_ptr;
        t.ptr.data = nullptr;
        t.shape_ptr.data = nullptr;
        t.strides_ptr.data = nullptr;
    }

    inline T*
    data_ptr()
    {
        return ptr.data;
    }

    inline const T*
    data_ptr() const
    {
        return ptr.data;
    }

    inline std::vector<std::size_t>
    shape() const
    {
        return std::vector(shape_ptr.data, shape_ptr.data + N);
    }

    inline std::size_t
    size(std::size_t dim) const
    {
        return shape_ptr.data[dim];
    }

    virtual void
    data_repr(std::ostream& os, int w) const
    {
        os << "[...]";
    }
};


template <typename T, std::size_t N, template <typename U> class ptr_traits = unmanaged_ptr_traits>
struct tensor_format {
    const tensor_base<T, N, ptr_traits>& tensor;
    const int w;

    tensor_format(const tensor_base<T, N, ptr_traits>& tensor_, const int w_ = 0)
    : tensor(tensor_),
      w(w_)
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const tensor_format<T, N, ptr_traits>& tf)
    {
        tf.tensor.data_repr(os, tf.w);
        return os;
    }
};


template <typename T, std::size_t N, template <typename U> class ptr_traits = unmanaged_ptr_traits>
std::ostream&
operator<<(std::ostream& os, const tensor_base<T, N, ptr_traits>& t)
{
    os << tensor_format<T, N, ptr_traits>(t, 1) << ", shape=(" << t.shape() << ")";
    return os;
}


template <typename T, std::size_t N, template <typename U> class ptr_traits = unmanaged_ptr_traits>
class tensor : public tensor_base<T, N, ptr_traits> {
public:
    tensor(T* data, const std::size_t* shape, const std::size_t* strides)
    : tensor_base<T, N, ptr_traits>(data, shape, strides)
    {}

    tensor<T, N - 1>
    at(std::size_t i)
    {
        auto new_data = this->data_ptr() + this->strides_ptr.data[0] * i;
        auto new_shape = this->shape_ptr.data + 1;
        auto new_strides = this->strides_ptr.data + 1;
        return tensor(new_data, new_shape, new_strides);
    }

    const tensor<const T, N - 1>
    at(std::size_t i) const
    {
        auto new_data = this->data_ptr() + this->strides_ptr.data[0] * i;
        auto new_shape = this->shape_ptr.data + 1;
        auto new_strides = this->strides_ptr.data + 1;
        return tensor<const T, N - 1>(new_data, new_shape, new_strides);
    }

    tensor<T, N - 1>
    operator[](std::size_t i)
    {
        return at(i);
    }

    void
    data_repr(std::ostream& os, int w) const override
    {
        auto size = this->size(0);
        auto max_size = fmt::edgeitems * 2 + 1;

        os << "[";
        if (size > max_size) {
            for (std::size_t i = 0; i < fmt::edgeitems; i++) {
                os << tensor_format(at(i), w + 1) << fmt::comma(i, size);
                os << std::endl << std::setw(w) << "";
            }

            os << "..., " << std::endl << std::setw(w) << "";

            for (std::size_t i = size - fmt::edgeitems; i < size; i++) {
                os << tensor_format(at(i), w + 1) << fmt::comma(i, size);
                if (i < size - 1) {
                    os << std::endl << std::setw(w) << "";
                }
            }
        } else {
            for (std::size_t i = 0; i < size; i++) {
                os << tensor_format(at(i), w + 1) << fmt::comma(i, size);
                if (i < size - 1) {
                    os << std::endl << std::setw(w) << "";
                }
            }
        }
        os << "]";
    }
};


template <typename T, template <typename U> class ptr_traits>
class tensor<T, 1, ptr_traits> : public tensor_base<T, 1, ptr_traits> {
public:
    tensor(T* data, const std::size_t* shape, const std::size_t* strides)
    : tensor_base<T, 1, ptr_traits>(data, shape, strides)
    {}

    T&
    operator[](std::size_t i)
    {
        return this->data_ptr[i];
    }

    void
    data_repr(std::ostream& os, int w) const override
    {
        auto size = this->size(0);
        auto max_size = fmt::edgeitems * 2 + 1;

        os << "[";
        if (size > max_size) {
            os << std::vector<T>(this->data_ptr(), this->data_ptr() + fmt::edgeitems);
            os << ", ..., ";
            os << std::vector<T>(this->data_ptr() + size - fmt::edgeitems, this->data_ptr() + size);
        } else {
            os << std::vector<T>(this->data_ptr(), this->data_ptr() + size);
        }

        os << "]";
    }
};


template <typename T, std::size_t N>
    requires(N > 0)
tensor<T, N, managed_ptr_traits>
rand(std::array<std::size_t, N> shape)
{
    auto strides = new std::size_t[N];
    strides[N - 1] = 1;

    for (auto i = N - 2; i < N; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    std::size_t size = 1;
    auto sizes = new std::size_t[N];
    for (auto i = 0; i < N; i++) {
        sizes[i] = shape[i];
        size *= shape[i];
    }

    T* data = new T[size];
    return tensor<T, N, managed_ptr_traits>(data, sizes, strides);
}


using bfloat_tensor1d = tensor<__fp16, 1>;
using bfloat_tensor2d = tensor<__fp16, 2>;
using int32_tensor1d = tensor<int32_t, 1>;
using int32_tensor2d = tensor<int32_t, 2>;
