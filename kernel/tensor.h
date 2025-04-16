// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>


using namespace metal;


template <uint N> struct tensor_layout {
    uint sizes[N];
    uint strides[N];
    uint offsets[N];
};


template <typename T, uint N> struct tensor {
    device T* data;
    constant tensor_layout<N>& layout;

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
