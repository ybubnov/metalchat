// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once


template <typename T> T inline __ceil_div(T a, T b) { return (a + b - 1) / b; }


#define __lib_metalchat_stringify(x) #x


#define __lib_metalchat_concatenate2(s1, s2) \
    __lib_metalchat_stringify(s1##_##s2)


#define __lib_metalchat_concatenate3(s1, s2, s3) \
    __lib_metalchat_stringify(s1##_##s2##_##s3)


#define __lib_metalchat_concatenate4(s1, s2, s3, s4) \
    __lib_metalchat_stringify(s1##_##s2##_##s3##_##s4)


#define __lib_metalchat_kernel(function_name, type)                             \
    template [[host_name(__lib_metalchat_concatenate2(function_name, type))]]   \
    kernel void                                                                 \
    function_name<type>(__##function_name##_parameters<type>, uint, uint, uint)


/// A macro renders the function prototype of a metal kernel with 2-dimensional execution grid.
///
/// Use this macro for kernels that do not use block-tiling.
#define __lib_metalchat_kernel2(function_name, type)                               \
    template [[host_name(__lib_metalchat_concatenate2(function_name, type))]]      \
    kernel void                                                                    \
    function_name<type>(__##function_name##_parameters<type>, uint2, uint2, uint2)


/// A macro renders the function prototype of a metal kernel with support of block-tiling and
/// 2-dimensional execution grid.
#define __lib_metalchat_kernel2_tiled(function_name, block_size, type)                    \
    template [[host_name(__lib_metalchat_concatenate3(function_name, block_size, type))]] \
    kernel void                                                                           \
    function_name<type, block_size>(                                                      \
        __##function_name##_parameters<type>, uint2, uint2, uint2                         \
    )


/// A macro renders the function prototype of a metal kernel with 3 mixed precision
/// arguments and 2-dimensional execution grid. The macro also emits a host name
/// as a concatenation of a function name and function types.
#define __lib_metalchat_kernel2_mixed3(function_name, type1, type2, type3)                   \
    template [[host_name(__lib_metalchat_concatenate4(function_name, type1, type2, type3))]] \
    kernel void                                                                              \
    function_name<type1, type2, type3>(                                                      \
        __##function_name##_parameters<type1, type2, type3>, uint2, uint2, uint2             \
    )


/// A macro renders the function prototype of a metal kernel with 3-dimensional execution grid.
///
/// Use this macro for kernels that do not use block-tiling.
#define __lib_metalchat_kernel3(function_name, type, block_size)                          \
    template [[host_name(__lib_metalchat_concatenate3(function_name, block_size, type))]] \
    kernel void                                                                           \
    function_name<type, block_size>(                                                      \
        __##function_name##_parameters<type>, uint3, uint3, uint3                         \
    )


/// A macro renders the function prototype of a metal kernel with support of block-tiling and
/// 3-dimensional execution grid.
#define __lib_metalchat_kernel3_tiled(function_name, block_size, type)                    \
    template [[host_name(__lib_metalchat_concatenate3(function_name, block_size, type))]] \
    kernel void                                                                           \
    function_name<type, block_size>(                                                      \
        __##function_name##_parameters<type>, uint3, uint3, uint3                         \
    )
