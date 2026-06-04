// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <vector>

#include <metalchat/accelerator.h>
#include <metalchat/nn/layer.h>


namespace metalchat {
namespace nn {


/// Sequential container of layers.
///
/// \ref layer_array can be indexed like a random access container, but layers it contains are
/// properly registered, and will be visible by all \ref basic_layer methods.
///
/// \tparam Layer a type of the layers this module stores.
///
/// ```cpp
/// struct my_layer : public basic_layer {
///     // Step 2. Define helper types.
///     using Linear = nn::linear<float>;
///     using LinearArray = nn::layer_array<Linear>;
///
///     // Step 2. Create a layer array as a type member.
///     nn::indirect_layer<LinearArray> linears;
///
///     // Step 3. Register a layer array as a sub-layer.
///     my_layer(hardware_accelerator& accelerator)
///     : layer(accelerator)
///     {
///         linears = register_layer<LinearArray>("linears");
///
///         for (std::size_t i = 0; i < 10; i++) {
///             // Step 3. Initialize layers within an array.
///             linears.emplace_back(10, 10, accelerator);
///         }
///     }
///
///     template<immutable_tensor2_t<float> Input>
///     auto
///     operator()(Input input)
///     {
///         for (std::size_t i = 0; i < 10; i++) {
///             // Step 4. Use layers as a regular random-access array.
///             input = linears[i / 2](input) + linears[i](input);
///         }
///         return input;
///     }
/// };
/// ```
///
/// \note You can access elements of the layer array through
/// \ref basic_layer::parameter(const std::string&) const
/// method by using the following syntax: `array.0`.
template <mutable_layer Layer> class layer_array : public basic_layer {
public:
    using size_type = std::size_t;
    using pointer = indirect_layer<Layer>;
    using reference = Layer&;
    using const_reference = const Layer&;

    /// The layer array constructor.
    layer_array(const hardware_accelerator& accelerator)
    : basic_layer(accelerator),
      _M_pointers()
    {}

    /// Returns a reference to the `pos`-element of the layer array.
    ///
    /// \param pos the position of a layer in the array.
    reference
    operator[](size_type pos)
    {
        return *_M_pointers[pos];
    }

    /// Returns a reference to the `pos`-element of the layer array.
    ///
    /// \param pos the position of a layer in the array.
    reference
    at(size_type pos)
    {
        return *_M_pointers[pos];
    }

    /// Returns a reference to the last element in the container.
    reference
    back()
    {
        return *_M_pointers.back();
    }

    /// Returns a const reference to the last element in the container.
    const_reference
    back() const
    {
        return *_M_pointers.back();
    }

    /// Returns a constant reference to the `pos`-element of the layer array.
    ///
    /// \param pos the position of a layer in the array.
    const_reference
    operator[](size_type pos) const
    {
        return *_M_pointers[pos];
    }

    /// Returns a constant reference to the `pos`-element of the layer array.
    ///
    /// \param pos the position of a layer in the array.
    const_reference
    at(size_type pos) const
    {
        return *_M_pointers[pos];
    }

    /// Appends an existing layer to the end of the container.
    ///
    /// \param layer the layer to append.
    void
    push_back(const indirect_layer<Layer>& layer)
    {
        auto name = std::format("{}", _M_pointers.size());
        auto ptr = register_layer(name, layer);
        _M_pointers.push_back(ptr);
    }

    /// Appends a new layer to the end of the container. The arguments `args...` are forwarded
    /// to the layer constructor as `std::forward<Args>(args)...`.
    ///
    /// \tparam Args argument types to forward to the constructor of the layer.
    /// \param args arguments to forward to the constructor of the layer.
    template <typename... Args>
    void
    emplace_back(Args&&... args)
    {
        // TODO: Does it make sense to pass in an accelerator, like in layer::register_layer?
        push_back(indirect_layer<Layer>(std::forward<Args>(args)...));
    }

    /// Returns the number of elements in the container.
    size_type
    size() const
    {
        return _M_pointers.size();
    }

private:
    std::vector<pointer> _M_pointers;
};


} // namespace nn
} // namespace metalchat
