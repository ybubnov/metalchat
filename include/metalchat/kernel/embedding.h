#pragma once

#include <cmath>
#include <concepts>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/format.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {


template <typename T> class embedding : public kernel {
private:
    inline static const std::string operation_name = "embedding";

public:
    embedding(device& device)
    : kernel(operation_name, type_traits<T>::name(), device)
    {}

    template <
        integral IndexType,
        ContiguousContainer InputContainer,
        ContiguousContainer WeightContainer>
    auto
    operator()(
        const tensor<IndexType, 2, InputContainer>& input,
        const tensor<T, 2, WeightContainer>& weight
    )
    {
        auto output = empty<T>({input.size(0), input.size(1), weight.size(1)}, m_device);

        auto data_size = input.numel();
        auto emb_size = weight.sizes().back();
        auto dim_size = input.sizes().back();
        auto num_rows = data_size / dim_size;

        constexpr std::size_t block_size = 4;
        constexpr std::size_t eblock_size = 128;

        auto x_size = ceil_div(dim_size, block_size);
        auto y_size = ceil_div(emb_size, eblock_size);
        auto thread = dim3(x_size, y_size);
        auto threads = dim3(x_size * num_rows, y_size);

        blocking(threads, thread)(
            scalar<IndexType>(dim_size), scalar<IndexType>(emb_size), input, weight, output
        );
        return output;
    }
};


template <typename T> class rope : public kernel {
private:
    inline static const std::string operation_name = "rope";

    std::size_t m_dim;
    float m_theta;
    float m_scale;

public:
    rope(device& device, std::size_t dim, float theta = 500000.0, float scale = 1.0)
    : kernel(operation_name, type_traits<T>::name(), device),
      m_dim(dim),
      m_theta(theta),
      m_scale(scale)
    {}

    // Implements the rotary positional encoding.
    //
    // Shape:
    //   - Input: :math:`(B, N, L, D)`, where :math:`B` is a batch size, :math:`N` is a number
    //            of attention heads, :math:`L` is a sequence length, and :math:`D` is a head
    //            dimension.
    //   - Output: :math:`(B, N, L, D)`, same as input.
    template <ContiguousContainer InputContainer, ContiguousContainer MultiplierContainer>
    auto
    operator()(
        const tensor<T, 4, InputContainer>& input,
        const tensor<int32_t, 0, MultiplierContainer>& offset
    )
    {
        std::cout << "ROPE input=" << input.sizes() << std::endl;
        assert((input.is_contiguous()));
        assert((input.size(3) % 2 == 0));

        if (input.size(3) != m_dim) {
            throw std::invalid_argument(std::format(
                "kernel::rope: the last dimension of the input should be {}, but received {}",
                m_dim, input.size(3)
            ));
        }

        constexpr std::size_t block_size = 4;

        auto input_strides = empty<uint32_t>({3});
        auto output_strides = empty<uint32_t>({3});

        auto output = empty_like(input, m_device);
        auto mat_size = input.size(2) * input.size(3);

        input_strides[0] = mat_size;
        input_strides[1] = input.stride(2);
        input_strides[2] = input.stride(3);

        output_strides[0] = mat_size;
        output_strides[1] = output.stride(2);
        output_strides[2] = output.stride(3);

        auto n_batch = input.numel() / mat_size;

        auto dim0 = m_dim / 2;
        auto dim1 = input.size(2);
        auto dim2 = ceil_div(n_batch, block_size);

        auto threads = dim3(dim0, dim1, dim2);
        auto thread = dim3(1, 1, block_size);

        blocking(threads, thread)(
            input, input_strides, output, output_strides, offset, scalar<float>(m_scale),
            scalar<float>(m_theta), scalar<uint32_t>(n_batch)
        );

        return output;
    }

    template <ContiguousContainer InputContainer>
    auto
    operator()(const tensor<T, 4, InputContainer>& input, std::size_t offset = 0)
    {
        return operator()(input, scalar<int32_t>(offset));
    }
};


} // namespace metalchat
