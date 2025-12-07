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
template <typename Layer, typename OutputLayer>
requires std::derived_from<Layer, nn::basic_layer> &&
         std::derived_from<OutputLayer, nn::basic_layer> && std::copy_constructible<OutputLayer>
void
replace(nn::basic_layer& input, const OutputLayer& new_value)
{
    std::vector<nn::named_layer> candidates;

    auto find_candidates = [&](nn::named_layer layer) {
        if (dynamic_pointer_cast<Layer>(layer.ptr) != nullptr) {
            candidates.push_back(layer);
        }
    };

    input.apply(find_candidates);

    for (auto& layer : candidates) {
        auto& layer_parent = input.get_parent_layer(layer.path);

        OutputLayer layer_value = new_value;
        layer_parent.register_layer(layer.name, std::move(layer_value));
    }
}


} // namespace quantization
} // namespace metalchat
