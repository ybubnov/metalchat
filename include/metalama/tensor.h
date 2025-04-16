#pragma once

#include <cstddef>
#include <iomanip>
#include <sstream>


#include <metalama/stream.h>


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

    std::size_t
    size(std::size_t dim)
    {
        return sizes[dim];
    }

    virtual std::string
    repr() const
    {
        return "[...]";
    }

    friend std::ostream&
    operator<<(std::ostream& os, const tensor_base& t)
    {
        auto sizes = std::vector<std::size_t>(t.sizes, t.sizes + N);
        os << "tensor(" << t.repr() << ", shape=[" << sizes << "])";
        return os;
    }
};


template <typename T, std::size_t N>
class tensor : public tensor_base<T, N> {
public:
    tensor(T* data_, const std::size_t* sizes_, const std::size_t* strides_)
    : tensor_base<T, N>(data_, sizes_, strides_)
    {}

    tensor<T, N - 1>
    operator[](std::size_t i)
    {
        return tensor(this->data + this->strides[0] * i, this->sizes + 1, this->strides + 1);
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

    std::string
    repr() const override
    {
        std::stringstream ss;
        ss << "[";

        auto size = this->sizes[0];
        if (size > 10) {
            ss << std::vector<T>(this->data, this->data + 3);
            ss << ", ..., ";
            ss << std::vector<T>(this->data + size - 3, this->data + size);
        } else {
            ss << std::vector<T>(this->data, this->data + size);
        }

        ss << "]";
        return ss.str();
    }
};


template <typename T>
class tensor<T, 2> : public tensor_base<T, 2> {
public:
    tensor(T* data_, const std::size_t* sizes_, const std::size_t* strides_)
    : tensor_base<T, 2>(data_, sizes_, strides_)
    {}

    tensor<T, 1>
    operator[](std::size_t i)
    {
        return tensor<T, 1>(this->data + this->strides[0] * i, this->sizes + 1, this->strides + 1);
    }

    const tensor<T, 1>
    operator[](std::size_t i) const
    {
        return tensor<T, 1>(this->data + this->strides[0] * i, this->sizes + 1, this->strides + 1);
    }

    std::string
    repr() const override
    {
        auto size = this->sizes[0];

        std::stringstream ss;
        ss << "[";
        if (size > 10) {
            for (std::size_t i = 0; i < 3; i++) {
                if (i > 0) {
                    ss << std::setw(8) << "";
                }
                ss << this->operator[](i).repr() << "," << std::endl;
            }

            ss << std::setw(8) << "" << "..., " << std::endl;

            for (std::size_t j = size - 3; j < size; j++) {
                ss << std::setw(8) << "";
                ss << this->operator[](j).repr();
                if (j < size - 1) {
                    ss << ", " << std::endl;
                }
            }
        } else {
            for (std::size_t i = 0; i < size; i++) {
                if (i > 0) {
                    ss << std::setw(8) << "";
                }
                ss << this->operator[](i).repr();
                if (i < size - 1) {
                    ss << ", " << std::endl;
                }
            }
        }
        ss << "]";
        return ss.str();
    }
};


using bfloat_tensor1d = tensor<__fp16, 1>;
using bfloat_tensor2d = tensor<__fp16, 2>;
