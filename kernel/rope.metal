// vi: set filetype=cpp :


#include <metal_common>
#include <metal_simdgroup>
#include <metal_stdlib>


kernel void
rope_bf16(
    const device bfloat* input [[buffer(0)]],
    const device uint* input_strides [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    const device uint* output_strides [[buffer(3)]],
    constant const uint& offset [[buffer(4)]],
    constant const float& scale [[buffer(5)]],
    constant const float& base [[buffer(6)]],
    constant const uint& n_batch [[buffer(7)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]
)
{
    constexpr uint BLOCK_SIZE = 4;

    float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
    float inv_freq = metal::exp2(-d * base);

    float L = scale * static_cast<float>(pos.y + offset);

    float theta = L * inv_freq;
    float costheta = metal::fast::cos(theta);
    float sintheta = metal::fast::sin(theta);

    uint output_index_1
        = (pos.x * output_strides[2] + pos.y * output_strides[1]
           + BLOCK_SIZE * pos.z * output_strides[0]);
    uint input_index_1
        = (pos.x * input_strides[2] + pos.y * input_strides[1]
           + BLOCK_SIZE * pos.z * input_strides[0]);

    uint output_index_2 = output_index_1 + grid.x * output_strides[2];
    uint input_index_2 = input_index_1 + grid.x * input_strides[2];

    for (uint i = 0; i < BLOCK_SIZE && pos.z * BLOCK_SIZE + i < n_batch; ++i) {
        float x1 = static_cast<float>(input[input_index_1]);
        float x2 = static_cast<float>(input[input_index_2]);
        float re_x1 = x1 * costheta - x2 * sintheta;
        float im_x2 = x1 * sintheta + x2 * costheta;

        output[output_index_1] = static_cast<bfloat>(re_x1);
        output[output_index_2] = static_cast<bfloat>(im_x2);

        input_index_1 += input_strides[0];
        input_index_2 += input_strides[0];
        output_index_1 += output_strides[0];
        output_index_2 += output_strides[0];
    }
}
