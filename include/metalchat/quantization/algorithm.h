// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <iterator>

#include <metalchat/nn/layer.h>


namespace metalchat {
namespace quantization {


/// Replaces all matches of the layers with the specified layer by copying it.
///
/// ```c++
/// using namespace metalchat;
///
/// auto llm = nn::llama3(/* ... */);
///
/// using SourceLayer = nn::linear<bf16>;
/// using OutputLayer = quantization::qlora_linear<bf16>;
///
/// quantization::replace<SourceLayer>(llm, OutputLayer());
/// ```
template <typename Layer, typename OutputLayer> requires std::derived_from<Layer, nn::basic_layer>
void
replace(nn::basic_layer& input, const OutputLayer& new_value)
{
    auto replace = [&](nn::named_layer layer) {
        auto layer_ptr = dynamic_pointer_cast<Layer>(layer.ptr);
        if (layer_ptr != nullptr) {
            *layer_ptr = new_value;
        }
    };

    input.apply(replace);
}


} // namespace quantization
} // namespace metalchat
