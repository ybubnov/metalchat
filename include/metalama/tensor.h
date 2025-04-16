#pragma once

#include <cstddef>
#include <iomanip>
#include <sstream>


#include <metalama/format.h>


template <typename T, std::size_t N>
class tensor_base {
protected:
    T* data;
    const std::size_t* sizes;
    const std::size_t* strides;

public:
    tensor_base(T* data_, const std::size_t* sizes_, const std::size_t* strides_)
    : data(data_),
      sizes(sizes_),
      strides(strides_)
    {}

    const T*
    data_ptr() const
    {
        return data;
    }

    std::vector<std::size_t>
    shape() const
    {
        return std::vector(sizes, sizes + N);
    }

    std::size_t
    size(std::size_t dim)
    {
        return sizes[dim];
    }

    virtual void
    data_repr(std::ostream& os, int w) const
    {
        os << std::setw(w) << "" << "[...]";
    }
};


template <typename T, std::size_t N>
struct tensor_format {
    const tensor_base<T, N>& tensor;
    const int w;

    tensor_format(const tensor_base<T, N>& tensor_, const int w_ = 0)
    : tensor(tensor_),
      w(w_)
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const tensor_format<T, N>& tf)
    {
        os << std::setw(tf.w) << "";
        tf.tensor.data_repr(os, tf.w);
        return os;
    }
};


template <typename T, std::size_t N>
std::ostream&
operator<<(std::ostream& os, const tensor_base<T, N>& t)
{
    os << tensor_format<T, N>(t, 0) << ", shape=(" << t.shape() << ")";
    return os;
}


template <typename T, std::size_t N>
class tensor : public tensor_base<T, N> {
public:
    tensor(T* data_, const std::size_t* sizes_, const std::size_t* strides_)
    : tensor_base<T, N>(data_, sizes_, strides_)
    {}

    tensor<T, 1>
    at(std::size_t i)
    {
        return tensor<T, 1>(this->data + this->strides[0] * i, this->sizes + 1, this->strides + 1);
    }

    const tensor<T, 1>
    at(std::size_t i) const
    {
        return tensor<T, 1>(this->data + this->strides[0] * i, this->sizes + 1, this->strides + 1);
    }

    tensor<T, N - 1>
    operator[](std::size_t i)
    {
        return at(i);
    }

    void
    data_repr(std::ostream& os, int w) const override
    {
        auto size = this->sizes[0];
        auto max_size = fmt::edgeitems * 2 + 1;
        w += 1;

        os << "[";
        if (size > max_size) {
            for (std::size_t i = 0; i < fmt::edgeitems; i++) {
                auto W = i > 0 ? w : 0;
                os << tensor_format(at(i), W) << fmt::comma(i, size) << std::endl;
            }

            os << std::setw(w) << "" << "..., " << std::endl;

            for (std::size_t i = size - fmt::edgeitems; i < size; i++) {
                os << tensor_format(at(i), w) << fmt::comma(i, size);
                if (i < size - 1) {
                    os << std::endl;
                }
            }
        } else {
            for (std::size_t i = 0; i < size; i++) {
                os << tensor_format(at(i), w) << fmt::comma(i, size);
                if (i < size - 1) {
                    os << std::endl;
                }
            }
        }
        os << "]";
    }
};


template <typename T>
class tensor<T, 1> : public tensor_base<T, 1> {
public:
    tensor(T* data_, const std::size_t* sizes_, const std::size_t* strides_)
    : tensor_base<T, 1>(data_, sizes_, strides_)
    {}

    T&
    operator[](std::size_t i)
    {
        return this->data[i];
    }

    void
    data_repr(std::ostream& os, int w) const override
    {
        auto size = this->sizes[0];
        auto max_size = fmt::edgeitems * 2 + 1;

        os << "[";
        if (size > max_size) {
            os << std::vector<T>(this->data, this->data + fmt::edgeitems);
            os << ", ..., ";
            os << std::vector<T>(this->data + size - fmt::edgeitems, this->data + size);
        } else {
            os << std::vector<T>(this->data, this->data + size);
        }

        os << "]";
    }
};


using bfloat_tensor1d = tensor<__fp16, 1>;
using bfloat_tensor2d = tensor<__fp16, 2>;
