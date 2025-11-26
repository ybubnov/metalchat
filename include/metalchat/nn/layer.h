// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <optional>

#include <metalchat/accelerator.h>
#include <metalchat/kernel.h>
#include <metalchat/tensor.h>


namespace metalchat {
namespace nn {


/// A Wrapper around a shared pointer for arbitrary layer implementation provides invocable
/// functionality for `Layer` implementations.
template <typename Layer> class shared_layer_ptr {
public:
    /// Construct a shared layer with no managed layer, i.e. empty `shared_ptr`.
    shared_layer_ptr()
    : _M_value(nullptr)
    {}

    /// Construct a shared layer that takes ownership from the specified `Layer` instance.
    shared_layer_ptr(Layer&& layer)
    : _M_value(std::move(layer))
    {}

    /// Construct a shared layer which shares ownership of the layer managed by `r`.
    shared_layer_ptr(const std::shared_ptr<Layer>& r)
    : _M_value(r)
    {}

    /// Invoke the stored layer target with the parameters `args`.
    ///
    /// Effectively does `f(std::forward<Args>(args)...);`, where `f` is the target layer.
    template <class... Args>
    auto
    operator()(Args&&... args)
    {
        return (*_M_value)(std::forward<Args>(args)...);
    }

    /// Return the raw shared pointer to the layer.
    std::shared_ptr<Layer>
    get() const
    {
        return _M_value;
    }

    /// Dereference the stored pointer to the Layer.
    Layer*
    operator->() noexcept
    {
        return _M_value.get();
    }

    /// Dereference the stored pointer to the Layer.
    Layer&
    operator*() noexcept
    {
        return (*_M_value);
    }

    const Layer&
    operator*() const noexcept
    {
        return (*_M_value);
    }

    template <typename DerivedLayer> requires std::derived_from<DerivedLayer, Layer>
    shared_layer_ptr<Layer>&
    operator=(const shared_layer_ptr<DerivedLayer>& derived)
    {
        _M_value = derived.get();
        return *this;
    }

private:
    std::shared_ptr<Layer> _M_value;
};


class basic_layer;


struct named_layer {
    std::string name;
    std::shared_ptr<basic_layer> ptr;
};


struct named_parameter {
    std::string name;
    std::shared_ptr<basic_tensor> ptr;
};


/// Layer is a basic building block of neural networks in MetalChat. A layer specifies a set of
/// (trainable) parameters it uses for computation and a set of upstream layers, used within a
/// layer computation logic.
class basic_layer {
public:
    /// A shared pointer to the basic tensor.
    using parameter_pointer = std::shared_ptr<basic_tensor>;
    /// A shared pointer to the basic layer.
    using layer_pointer = std::shared_ptr<basic_layer>;

    using parameter_container = std::unordered_map<std::string, parameter_pointer>;
    using layer_container = std::unordered_map<std::string, layer_pointer>;

    /// Construct a layer that is a associated with the specified hardware accelerator.
    basic_layer(const hardware_accelerator& accelerator);

    /// Get a constant reference to the hardware accelerator.
    const hardware_accelerator&
    accelerator() const;

    /// Get a reference to the hardware accelerator.
    hardware_accelerator&
    accelerator();

    /// Register an upstream layer for the current layer. The layer could be accessed using
    /// the given name using `basic_layer::get_layer` method.
    ///
    /// The registry of layers owns the upstream layer, and the method returns a object pointing
    /// to that owned layer.
    ///
    /// \note You can explore a variety of different layers in
    /// \verbatim embed:rst:inline :doc:`nn` \endverbatim.
    ///
    /// A common practice is registering upstream layers within a downstream layer constructor
    /// like in the example below.
    ///
    /// ```c++
    /// using namespace metalchat;
    ///
    /// struct custom_layer : public basic_layer {
    ///     // Declare upstream layers here.
    ///     nn::linear<float>::layer_pointer linear1;
    ///     nn::linear<float>::layer_pointer linear2;
    ///
    ///    custom_layer(hardware_accelerator accelerator)
    ///    : basic_layer(accelerator)
    ///    {
    ///       // Register layers here.
    ///       linear1 = register_layer("linear1", nn::linear<float>(accelerator));
    ///       linear2 = register_layer("linear2", nn::linear<float>(accelerator));
    ///    }
    /// };
    /// ```
    template <typename Layer>
    shared_layer_ptr<Layer>
    register_layer(const std::string& name, Layer&& l)
    {
        auto layer_ptr = std::make_shared<Layer>(std::move(l));
        _M_layers.emplace(name, layer_ptr);
        return shared_layer_ptr(layer_ptr);
    }

    template <typename Layer>
    shared_layer_ptr<Layer>
    register_layer(const std::string& name, const shared_layer_ptr<Layer>& layer_ptr)
    {
        _M_layers.emplace(name, layer_ptr.get());
        return layer_ptr;
    }

    /// Get upstream layer by name. This method does not perform recursive lookup and only
    /// returns layers registered at the current layer. If layer is not registered, method
    /// throws exception.
    const basic_layer&
    get_layer(const std::string& name) const;

    /// Add a parameter to the layer.
    ///
    /// The parameter can be accessed using `basic_layer::get_parameter` method and updated with
    /// `basic_layer::set_parameter` method respectively.
    ///
    /// A common practice is registering parameters of the layers that could be updated
    /// externally (loaded from a file, or stored after inference):
    ///
    /// ```c++
    /// using namespace metalchat;
    ///
    /// struct custom_layer : public basic_layer {
    ///     // Declare parameters here.
    ///     shared_tensor<float, 3> weight;
    ///
    ///     custom_layer(hardware_accelerator accelerator)
    ///     : basic_layer(accelerator)
    ///     {
    ///         weight = register_parameter("weight", empty<float>({10, 4, 3}, accelerator));
    ///     }
    /// };
    /// ```
    template <immutable_tensor Tensor>
    shared_tensor_ptr<Tensor>
    register_parameter(const std::string& name, Tensor&& tensor)
    {
        auto tensor_ptr = shared_tensor_ptr(std::move(tensor));
        return register_parameter(name, tensor_ptr);
    }

    /// Add a parameter to the layer.
    ///
    /// This method shared ownership of the tensor (parameter) with the caller. Consider the
    /// following example, where the parameter is constructed with the basic layer using
    /// delegated constructors, and then registered in the body of the constructor:
    /// ```c++
    /// using namespace metalchat;
    ///
    /// struct custom_layer : public basic_layer {
    ///     // Declare parameters here.
    ///     shared_tensor<float, 3> weight;
    ///
    ///     custom_layer(hardware_accelerator accelerator)
    ///     : basic_layer(accelerator),
    ///       weight(full<float>({5, 4, 2}, 4.0, accelerator))
    ///     {
    ///         register_parameter("weight", weight);
    ///     }
    /// };
    /// ```
    template <immutable_tensor Tensor>
    shared_tensor_ptr<Tensor>
    register_parameter(const std::string& name, const shared_tensor_ptr<Tensor>& tensor_ptr)
    {
        _M_params.insert_or_assign(name, tensor_ptr.get());
        return tensor_ptr;
    }

    /// Set value to the registered layer parameter.
    ///
    /// When the specified parameter is not found, the method throws an exception. The method
    /// supports assignment of the nested parameters.
    ///
    /// Example:
    ///
    /// ```c++
    /// using namespace metalchat;
    ///
    /// auto accelerator = hardware_accelerator(32);
    /// auto linear = nn::linear<float>(accelerator);
    ///
    /// linear.set_parameter("weight", empty<float>({4, 4}, accelerator));
    /// ```
    template <immutable_tensor Tensor>
    void
    set_parameter(const std::string& name, Tensor&& tensor)
    {
        auto param_ptr = get_parameter(name);
        move_tensor_to_pointer(param_ptr, std::move(tensor));
    }

    /// Return a pointer to the registered parameter by the specified name.
    ///
    /// This method also supports recursive lookup of the parameter within children layers
    /// if the name contains a dot ('.') delimiter.
    parameter_pointer
    get_parameter(const std::string& name) const;

    /// Return a set of parameters with fully-qualified names. Parameters of different layers
    /// are separated using dot (".") delimiter symbol.
    ///
    /// If you want to return only parameters of the current layer and drop upstream parameters,
    /// you could call this method with `recurse = false`.
    const parameter_container
    get_parameters(bool recurse = true) const;

    /// Apply a function to every parameters of the layer.
    ///
    /// This method traverses all parameters in breadth-first way when `recurse` parameter is set
    /// to `true`. Otherwise, only parameters of the current layer are visited.
    template <std::invocable<named_parameter> Function>
    void
    apply(Function fn, bool recurse = true) const
    {
        for (const auto& [full_name, param] : _M_params) {
            fn(named_parameter{full_name, param});
        }

        if (!recurse) {
            return;
        }

        using layer_type = layer_container::value_type;
        std::deque<layer_type> layers(_M_layers.begin(), _M_layers.end());

        while (!layers.empty()) {
            auto [name, layer_ptr] = layers.front();
            layers.pop_front();

            // Iterate over the downstream layers, and push them back to the queue.
            for (auto [child_name, child_layer_ref] : layer_ptr->_M_layers) {
                auto full_name = name + "." + child_name;
                layers.emplace_back(full_name, child_layer_ref);
            }

            for (auto [param_name, param] : layer_ptr->_M_params) {
                auto full_name = name + "." + param_name;
                fn(named_parameter{full_name, param});
            }
        }
    }

    template <std::invocable<named_layer> Function>
    void
    apply(Function fn)
    {
        using layer_type = layer_container::value_type;
        std::deque<layer_type> layers(_M_layers.begin(), _M_layers.end());

        while (!layers.empty()) {
            auto [name, layer_ptr] = layers.front();
            layers.pop_front();

            // Iterate over the downstream layers, and push them back to the queue.
            for (auto [child_name, child_layer_ref] : layer_ptr->_M_layers) {
                auto full_name = name + "." + child_name;
                layers.emplace_back(full_name, child_layer_ref);
            }

            fn(named_layer{name, layer_ptr});
        }
    }

    virtual ~basic_layer() {}

private:
    parameter_container _M_params;
    layer_container _M_layers;
    hardware_accelerator _M_accelerator;

    template <immutable_tensor Tensor>
    void
    move_tensor_to_pointer(std::shared_ptr<basic_tensor>& ptr, Tensor&& tensor)
    {
        auto tensor_ptr = std::dynamic_pointer_cast<Tensor>(ptr);
        if (!tensor_ptr) {
            throw std::invalid_argument("basic_layer::move_tensor: tensor types are not compatible"
            );
        }
        *tensor_ptr = std::move(tensor);
    }
};


/// Sequential container of layers.
///
/// \ref layer_array can be indexed like a random access container, but layers it contains are
/// properly registered, and will be visible by all \ref basic_layer methods.
///
/// \tparam Layer a type of the layers this module stores.
///
/// ```c++
/// struct my_layer : public basic_layer {
///     // Step 1. Create a layer array as a type member.
///     layer_array<nn::linear<float>> linears;
///
///     // Step 2. Register a layer array as a sub-layer.
///     my_layer(const hardware_accelerator& accelerator)
///     : basic_layer(accelerator),
///       linears(*register_layer("linears", layer_array<nn::linear<float>>(accelerator)))
///     {
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
/// \note You can access elements of the layer array through \ref basic_layer::get_parameter
/// method by using the following syntax: `array.0`.
template <typename Layer> class layer_array : public basic_layer {
public:
    using size_type = std::size_t;
    using pointer = shared_layer_ptr<Layer>;
    using reference = Layer&;
    using const_reference = const Layer&;

    using layer_type = layer_array<Layer>;
    using layer_pointer = shared_layer_ptr<layer_type>;

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
    push_back(Layer&& layer)
    {
        auto name = std::format("{}", _M_pointers.size());
        auto ptr = register_layer(name, std::move(layer));
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
        push_back(Layer(std::forward<Args>(args)...));
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
