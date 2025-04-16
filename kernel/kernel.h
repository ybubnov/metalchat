#pragma once

#define __lib_metalchat_stringify(x) #x
#define __lib_metalchat_concatenate(a, b, c) __lib_metalchat_stringify(a##_##b##_##c)


#define __lib_metalchat_kernel(function_name, type, block_size)                          \
    template [[host_name(__lib_metalchat_concatenate(function_name, block_size, type))]] \
    kernel void                                                                          \
    function_name<type, block_size>(__##function_name##_parameters<type>, uint, uint)


#define __lib_metalchat_kernel2(function_name, type, block_size)                               \
    template [[host_name(__lib_metalchat_concatenate(function_name, block_size, type))]]       \
    kernel void                                                                                \
    function_name<type, block_size>(__##function_name##_parameters<type>, uint2, uint2, uint2)


#define __lib_metalchat_kernel3(function_name, type, block_size)                         \
    template [[host_name(__lib_metalchat_concatenate(function_name, block_size, type))]] \
    kernel void                                                                          \
    function_name<type, block_size>(__##function_name##_parameters<type>, uint3, uint3, uint3)
