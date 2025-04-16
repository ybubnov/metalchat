// vi: set filetype=cpp :

#include <metal_common>
#include <metal_stdlib>


using namespace metal;


template <uint64_t N> struct tensor_layout {
    uint64_t sizes[N];
    uint64_t strides[N];
    uint64_t offsets[N];
};


template <typename T, uint64_t N> struct tensor {
    device T* data;
    constant tensor_layout<N>& layout;

    template <typename... I>
    device T&
    at(const I... indices)
    {
        static_assert(
            sizeof...(indices) == N,
            "matrix::at expects the same number of indices as matrix dimensionality"
        );

        uint64_t ptr_offset = 0;
        uint64_t i = 0;

        ((ptr_offset += layout.strides[i] * (layout.offsets[i] + indices), ++i), ...);

        return *(data + ptr_offset);
    }

    inline uint64_t
    size(uint64_t dim)
    {
        return layout.sizes[dim];
    }

    inline uint64_t
    stride(uint64_t dim)
    {
        return layout.strides[dim];
    }

    inline uint64_t
    offset(uint64_t dim)
    {
        return layout.offsets[dim];
    }
};
