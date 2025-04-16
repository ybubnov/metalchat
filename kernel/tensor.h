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
    at(uint i, uint j)
    {
        auto ptr_offset = 0;
        ptr_offset += layout.strides[0] * i + layout.offsets[0];
        ptr_offset += layout.strides[1] * j + layout.offsets[1];
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
