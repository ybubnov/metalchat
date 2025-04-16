// vi: set filetype=cpp :

#include <metal_common>

#include "kernel.h"
#include "tensor.h"


using namespace metal;


template <typename T> struct __gt_parameters {
    constant tensor_layout<2>& output_layout;
    device bool* output;
    constant tensor_layout<2>& input1_layout;
    device const T* input1;
    constant tensor_layout<2>& input2_layout;
    device const T* input2;
};


template <typename T, uint BlockSize>
kernel void
gt(__gt_parameters<T> params,
   uint gid [[threadgroup_position_in_grid]],
   uint tid [[thread_index_in_threadgroup]])
{
    tensor<const T, 2> in1{params.input1, params.input1_layout};
    tensor<const T, 2> in2{params.input2, params.input2_layout};
    tensor<bool, 2> out{params.output, params.output_layout};

    const uint dim_size = in1.size(1);
    const uint i = gid;

    const uint begin = tid * BlockSize;
    const uint end = begin + BlockSize;

    for (uint k = 0; k < end && k < dim_size; k++) {
        out.at(i, k) = in1.at(i, k) > in2.at(i, k);
    }
}


__lib_metalchat_kernel(gt, bfloat, 8);
__lib_metalchat_kernel(gt, bfloat, 16);
__lib_metalchat_kernel(gt, bfloat, 32);

__lib_metalchat_kernel(gt, float, 8);
__lib_metalchat_kernel(gt, float, 16);
__lib_metalchat_kernel(gt, float, 32);
