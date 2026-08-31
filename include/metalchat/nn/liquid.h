// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once


#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel/activation.h>
#include <metalchat/nn/attention.h>
#include <metalchat/nn/embedding.h>
#include <metalchat/nn/layer_array.h>
#include <metalchat/nn/rmsnorm.h>
#include <metalchat/nn/transformer.h>


namespace metalchat {
namespace nn {


struct lfm2_options {};


template <typename T, contiguous_container Container = hardware_memory_container<T>>
class lfm2 : public basic_layer {
    using MultiheadAttention = multihead_attention<T, Container>;
    using RecurrentAttention = recurrent_attention<T, Container>;
    using Transformer = transformer<T, Container, kernel::silu<T>>;
    using TransformerArray = nn::layer_array<Transformer>;
    using Embedding = embedding<T, Container>;
    using RotaryPositionalEmbedding = rope<T>;
    using RMSNorm = rmsnorm<T, Container>;
    using Linear = linear<T, Container>;

    indirect_layer<Embedding> _M_embedding;
    indirect_layer<Linear> _M_output;

    indirect_layer<RMSNorm> _M_norm;
    indirect_layer<TransformerArray> _M_transforms;

    lfm2_options _M_options;

public:
    lfm2(const lfm2_options& options, hardware_accelerator& accelerator)
    : basic_layer(accelerator),
      _M_options(options)
    {}
};


} // namespace nn
} // namespace metalchat
