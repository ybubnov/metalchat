// vi: set filetype=cpp :
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metal_common>
#include <metal_stdlib>


#define StructuredBuffer(T, N, name)                      \
    using name##_ValueType = const T;                     \
    static constexpr constant uint name##_Dimensions = N; \
    constant tensor_layout<N>& name##_layout;             \
    device const T* name##_data


#define RWStructuredBuffer(T, N, name)                    \
    using name##_ValueType = T;                           \
    static constexpr constant uint name##_Dimensions = N; \
    constant tensor_layout<N>& name##_layout;             \
    device T* name##_data


#define Tensor(params, name)                         \
    tensor<                                          \
        typename decltype(params)::name##_ValueType, \
        decltype(params)::name##_Dimensions>         \
    name{params.name##_data, params.name##_layout}


template <uint N> struct tensor_layout {
    uint sizes[N];
    uint strides[N];
    uint offsets[N];
};


template <typename T, uint N> struct tensor {
    device T* data;
    constant tensor_layout<N>& layout;

    inline device T&
    at(uint i0)
    {
        auto ptr_offset = 0;
        ptr_offset += layout.strides[0] * i0 + layout.offsets[0];
        return *(data + ptr_offset);
    }

    inline device T&
    at(uint i0, uint i1)
    {
        auto ptr_offset = 0;
        ptr_offset += layout.strides[0] * i0 + layout.offsets[0];
        ptr_offset += layout.strides[1] * i1 + layout.offsets[1];
        return *(data + ptr_offset);
    }

    inline device T&
    at(uint i0, uint i1, uint i2)
    {
        auto ptr_offset = 0;
        ptr_offset += layout.strides[0] * i0 + layout.offsets[0];
        ptr_offset += layout.strides[1] * i1 + layout.offsets[1];
        ptr_offset += layout.strides[2] * i2 + layout.offsets[2];
        return *(data + ptr_offset);
    }

    inline uint
    size(uint dim)
    {
        return layout.sizes[dim];
    }

    inline uint
    stride(uint dim)
    {
        return layout.strides[dim];
    }

    inline uint
    offset(uint dim)
    {
        return layout.offsets[dim];
    }
};
