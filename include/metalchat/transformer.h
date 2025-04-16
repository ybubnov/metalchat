#pragma once


#include "metalchat/tensor.h"


namespace metalchat {


template <typename T>
struct attention {
    tensor<T, 2> q;
    tensor<T, 2> k;
    tensor<T, 2> v;
    tensor<T, 2> o;
};


} // namespace metalchat
