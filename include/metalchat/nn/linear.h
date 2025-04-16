#pragma once

#include <iostream>

#include <metalchat/functional.h>


namespace metalchat {
namespace nn {


/*
template <typename T, std::size_t N>
struct parameter {

    parameter()
    : data()
    {}

    parameter(shared_tensor<T, N, Container> _data)
    : data(_data)
    {}

    std::optional<shared_tensor<T, N, Container>> data;
};


struct function {
    using value_type = std::reference_wrapper<parameter>;

    void
    register_parameter(std::string name)
    {
    }

    void
    get_paramater(std::string name)
    {
    }

    template <immutable_tensor Tensor>
    void
    set_parameter(std::string name, Tensor t)
    {
    }

    // get_parameters();

protected:
    std::unordered_map<std::string, value_type> _m_parameters;
};
*/


/// Applies an affine linear transformation to the incoming data.
///
/// This module does not support bias adjustment to the input tensor, and only multiplies
/// it (input) by the specified weight tensor. Meaning it effectively works as matrix
/// multiplication operation.
template <typename T, contiguous_container WeightContainer> class linear {
private:
    // parameter<T, 2> _m_weight;
    //
    // register_parameter("weight", _m_weight);
    // register_function("", _m_bmm);

    shared_tensor<T, 2, WeightContainer> _m_weight;
    hardware_accelerator& _m_accelerator;

public:
    linear(shared_tensor<T, 2, WeightContainer> weight, hardware_accelerator& gpu)
    : _m_weight(weight.transpose({1, 0})),
      _m_accelerator(gpu)
    {}

    linear(tensor<T, 2, WeightContainer>&& weight, hardware_accelerator& gpu)
    : linear(shared_tensor(std::move(weight)), gpu)
    {}

    template <immutable_tensor_t<T> Input>
    auto
    operator()(Input input)
    {
        return matmul(input, _m_weight, _m_accelerator);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const linear& l)
    {
        os << "nn::linear<" << type_traits<T>::name() << ">";
        os << "(" << l._m_weight.size(0) << ", " << l._m_weight.size(1) << ")";
        return os;
    }
};


} // namespace nn
} // namespace metalchat
