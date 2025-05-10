// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>


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


using layout1 = tensor_layout<1>;
using layout2 = tensor_layout<2>;
using layout3 = tensor_layout<3>;


template <typename T> struct tensor1 {
    constant layout1& layout;
    device T* data;

    tensor1(constant layout1& _layout, device T* _data)
    : layout(_layout),
      data(_data)
    {}

    inline device T&
    at(uint i0)
    {
        auto ptr_offset = layout.strides[0] * i0 + layout.offsets[0];
        return *(data + ptr_offset);
    }

    inline const device T&
    at(uint i0) const
    {
        return const_cast<thread tensor1&>(*this).at(i0);
    }

    inline uint
    size(uint dim) const
    {
        return layout.sizes[dim];
    }

    inline uint
    stride(uint dim) const
    {
        return layout.strides[dim];
    }

    inline uint
    offset(uint dim) const
    {
        return layout.offsets[dim];
    }
};


template <typename T> struct tensor2 {
    constant layout2& layout;
    device T* data;

    tensor2(constant layout2& _layout, device T* _data)
    : layout(_layout),
      data(_data)
    {}

    inline device T&
    at(uint i0, uint i1)
    {
        auto ptr_offset = 0;
        ptr_offset += layout.strides[0] * i0 + layout.offsets[0];
        ptr_offset += layout.strides[1] * i1 + layout.offsets[1];
        return *(data + ptr_offset);
    }

    inline const device T&
    at(uint i0, uint i1) const
    {
        return const_cast<thread tensor2&>(*this).at(i0, i1);
    }

    inline uint
    size(uint dim) const
    {
        return layout.sizes[dim];
    }

    inline uint
    stride(uint dim) const
    {
        return layout.strides[dim];
    }

    inline uint
    offset(uint dim) const
    {
        return layout.offsets[dim];
    }
};


template <typename T> struct tensor3 {
    constant layout3& layout;
    device T* data;

    tensor3(constant layout3& _layout, device T* _data)
    : layout(_layout),
      data(_data)
    {}

    inline device T&
    at(uint i0, uint i1, uint i2)
    {
        auto ptr_offset = 0;
        ptr_offset += layout.strides[0] * i0 + layout.offsets[0];
        ptr_offset += layout.strides[1] * i1 + layout.offsets[1];
        ptr_offset += layout.strides[2] * i2 + layout.offsets[2];
        return *(data + ptr_offset);
    }

    inline const device T&
    at(uint i0, uint i1, uint i2) const
    {
        return const_cast<thread tensor3&>(*this).at(i0, i1, i2);
    }

    inline uint
    size(uint dim) const
    {
        return layout.sizes[dim];
    }

    inline uint
    stride(uint dim) const
    {
        return layout.strides[dim];
    }

    inline uint
    offset(uint dim) const
    {
        return layout.offsets[dim];
    }
};
