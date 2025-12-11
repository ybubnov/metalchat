// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/nn.h>
#include <metalchat/safetensor.h>


namespace metalchat {


struct llama3_traits {
    using layer_type = nn::llama3<bf16>;
    using layer_options = nn::llama3_options;

    // The original implementation of Llama 3.2 shares the weight of token embeddings
    // and the output layer, use a shared tensor in order to reduce memory footprint.
    //
    // This adaptor implement \ref safetensor_document_adaptor concept and creates an
    // alias between output and embedding layers.
    struct reference_document_adaptor {
        void
        adapt(safetensor_document& document) const;
    };
};


/// ```c++
/// using Layer = metalchat::llama3_traits;
/// using Autoloader = metalchat::huggingface_autoloader<Layer>;
/// Autoloader autoloader("Llama-3.1-1B-Instruct");
/// auto layer = autoloader.load();
/// ```
template <typename LayerTraits> struct huggingface_autoloader {
    using layer_type = LayerTraits::layer_type;
    using layer_options = LayerTraits::layer_options;

    huggingface_autoloader(const std::filesystem::path& local_path);

    // void merge_options(layer_options options);
    // void override_options(layer_options options);

    nn::indirect_layer<layer_type>
    load(hardware_accelerator& accelerator) const;
    nn::indirect_layer<layer_type>
    load() const;
};


} // namespace metalchat
