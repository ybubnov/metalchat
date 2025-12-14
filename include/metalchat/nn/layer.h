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


class basic_layer;


struct named_layer {
    std::string path;
    std::string name;
    std::shared_ptr<basic_layer> ptr;
};


struct named_parameter {
    std::string path;
    std::string name;
    std::shared_ptr<basic_tensor> ptr;
};


/// A Wrapper around a shared pointer for arbitrary layer implementation provides invocable
/// functionality for `Layer` implementations.
template <typename Layer> class indirect_layer {
public:
    using layer_type = std::remove_cvref_t<Layer>;
    using layer_pointer = layer_type::layer_pointer;
    using layer_container = layer_type::layer_container;
    using parameter_pointer = layer_type::parameter_pointer;
    using parameter_container = layer_type::parameter_container;

    /// Construct a shared layer with no managed layer, i.e. empty `shared_ptr`.
    indirect_layer()
    : _M_value(nullptr)
    {}

    indirect_layer(indirect_layer&& other) noexcept = default;
    indirect_layer(const indirect_layer& other) noexcept = default;

    /// Construct a shared layer which shares ownership of the layer managed by `r`.
    indirect_layer(std::shared_ptr<layer_type> r)
    : _M_value(r)
    {}

    template <typename... Args> requires std::constructible_from<Layer, Args...>
    indirect_layer(Args&&... args)
    : indirect_layer(std::make_shared<layer_type>(std::forward<Args>(args)...))
    {
        _M_value->initialize();
    }

    /// Invoke the stored layer target with the parameters `args`.
    ///
    /// Effectively does `f(std::forward<Args>(args)...);`, where `f` is the target layer.
    template <typename... Args>
    auto
    operator()(Args&&... args)
    {
        return (*_M_value)(std::forward<Args>(args)...);
    }

    /// Return the raw shared pointer to the layer.
    std::shared_ptr<layer_type>
    get() const
    {
        return _M_value;
    }

    /// Dereference the stored pointer to the Layer.
    layer_type*
    operator->() noexcept
    {
        return _M_value.get();
    }

    /// Dereference the stored pointer to the Layer.
    layer_type&
    operator*() noexcept
    {
        return (*_M_value);
    }

    const layer_type&
    operator*() const noexcept
    {
        return (*_M_value);
    }

    indirect_layer&
    operator=(const indirect_layer& other) = default;

    indirect_layer&
    operator=(indirect_layer&& other) = default;

    const hardware_accelerator&
    accelerator() const;

    hardware_accelerator&
    accelerator();

    basic_layer&
    get_layer(const std::string& name) const;

    basic_layer&
    get_parent_layer(const std::string& name) const;

    template <immutable_tensor Tensor>
    void
    set_parameter(const std::string& name, Tensor&& tensor);

    parameter_pointer
    get_parameter(const std::string& name) const;

    const parameter_container
    get_parameters(bool recurse = true) const;

    template <std::invocable<named_parameter> Function>
    void
    apply(Function fn, bool recurse = true) const;

    template <std::invocable<named_layer> Function>
    void
    apply(Function fn) const;

    virtual ~indirect_layer() = default;

private:
    std::shared_ptr<layer_type> _M_value;
};


template <typename Layer> indirect_layer(Layer&&) -> indirect_layer<Layer>;


template <typename Layer> class polymorphic_layer {
public:
    polymorphic_layer(const std::string& name, const std::weak_ptr<basic_layer>& ptr)
    : _M_layer(ptr),
      _M_name(name)
    {}

    polymorphic_layer()
    : _M_layer(),
      _M_name()
    {}

    template <class... Args>
    auto
    operator()(Args&&... args);

    template <typename DerivedLayer> requires std::derived_from<DerivedLayer, Layer>
    polymorphic_layer<Layer>&
    operator=(indirect_layer<DerivedLayer>&& derived);

private:
    std::weak_ptr<basic_layer> _M_layer;
    std::string _M_name;
};


/// Layer is a basic building block of neural networks in MetalChat. A layer specifies a set of
/// (trainable) parameters it uses for computation and a set of upstream layers, used within a
/// layer computation logic.
class basic_layer : public std::enable_shared_from_this<basic_layer> {
public:
    /// A shared pointer to the basic tensor.
    using parameter_pointer = std::shared_ptr<basic_tensor>;
    /// A shared pointer to the basic layer.
    using layer_pointer = std::shared_ptr<basic_layer>;

    using parameter_container = std::unordered_map<std::string, parameter_pointer, _StringHash>;
    using layer_container = std::unordered_map<std::string, layer_pointer, _StringHash>;

    basic_layer(char delimiter, const hardware_accelerator& accelerator);

    /// Construct a layer that is a associated with the specified hardware accelerator.
    basic_layer(const hardware_accelerator& accelerator);

    /// Get a constant reference to the hardware accelerator.
    const hardware_accelerator&
    accelerator() const;

    /// Get a reference to the hardware accelerator.
    hardware_accelerator&
    accelerator();

    virtual void
    initialize();

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
    ///     indirect_layer<nn::linear<float>> linear1;
    ///     indirect_layer<nn::linear<float>> linear2;
    ///
    ///    custom_layer(hardware_accelerator accelerator)
    ///    : basic_layer(accelerator)
    ///    {
    ///       // Register layers here.
    ///       linear1 = register_layer<nn::linear<float>>("linear1");
    ///       linear2 = register_layer<nn::linear<float>>("linear2");
    ///    }
    /// };
    /// ```
    template <typename Layer, typename... Args>
    requires std::constructible_from<Layer, Args..., hardware_accelerator&> &&
             std::derived_from<Layer, basic_layer>
    indirect_layer<Layer>
    register_layer(const std::string& name, Args&&... args)
    {
        indirect_layer<Layer> layer(std::forward<Args>(args)..., _M_accelerator);
        _M_layers.insert_or_assign(name, layer.get());
        return layer;
    }

    template <typename Layer>
    indirect_layer<Layer>
    register_layer(const std::string& name, const indirect_layer<Layer>& layer_ptr)
    {
        layer_pointer ptr = layer_ptr.get();
        _M_layers.insert_or_assign(name, ptr);
        return layer_ptr;
    }

    template <typename Base, typename Layer, typename... Args>
    requires std::constructible_from<Layer, Args..., hardware_accelerator&> &&
             std::derived_from<Layer, Base> && std::derived_from<Base, basic_layer>
    polymorphic_layer<Base>
    register_polymorphic_layer(const std::string& name, Args&&... args)
    {
        // TODO: move the initialize call to polymorphic_layer?
        auto layer_ptr = std::make_shared<Layer>(std::forward<Args>(args)..., _M_accelerator);
        layer_ptr->initialize();

        _M_layers.insert_or_assign(name, layer_ptr);
        return polymorphic_layer<Base>(name, weak_from_this());
    }

    template <typename Base> requires std::derived_from<Base, basic_layer>
    polymorphic_layer<Base>
    register_polymorphic_layer(const std::string& name)
    {
        _M_layers.insert_or_assign(name, nullptr);
        return polymorphic_layer<Base>(name, weak_from_this());
    }

    /// Get upstream layer by name. This method does not perform recursive lookup and only
    /// returns layers registered at the current layer. If layer is not registered, method
    /// throws exception.
    basic_layer&
    get_layer(const std::string& name) const;

    basic_layer&
    get_parent_layer(const std::string& name) const;

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
    /// are separated using dot ('.') delimiter symbol.
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
        for (const auto& [param_name, param] : _M_params) {
            fn(named_parameter{param_name, param_name, param});
        }

        if (!recurse) {
            return;
        }

        using layer_type = layer_container::value_type;
        std::deque<layer_type> layers(_M_layers.begin(), _M_layers.end());

        while (!layers.empty()) {
            auto [layer_path, layer_ptr] = layers.front();
            layers.pop_front();

            // Iterate over the downstream layers, and push them back to the queue.
            for (auto [child_name, child_layer_ptr] : layer_ptr->_M_layers) {
                auto child_path = layer_path + _M_delimiter + child_name;
                layers.emplace_back(child_path, child_layer_ptr);
            }

            for (auto [param_name, param] : layer_ptr->_M_params) {
                auto param_path = layer_path + _M_delimiter + param_name;
                fn(named_parameter{param_path, param_name, param});
            }
        }
    }

    template <std::invocable<named_layer> Function>
    void
    apply(Function fn) const
    {
        using layer_type = layer_container::value_type;
        std::deque<layer_type> layers(_M_layers.begin(), _M_layers.end());

        while (!layers.empty()) {
            auto [layer_path, layer_ptr] = layers.front();
            layers.pop_front();

            // Iterate over the downstream layers, and push them back to the queue.
            for (auto [child_name, child_layer_ptr] : layer_ptr->_M_layers) {
                auto child_path = layer_path + _M_delimiter + child_name;
                layers.emplace_back(child_path, child_layer_ptr);
            }

            auto delim_pos = layer_path.rfind(_M_delimiter);
            auto layer_name = layer_path;
            if (delim_pos != std::string::npos) {
                layer_name = layer_name.substr(delim_pos + 1);
            }
            fn(named_layer{layer_path, layer_name, layer_ptr});
        }
    }

    virtual ~basic_layer() = default;

private:
    parameter_container _M_params;
    layer_container _M_layers;
    hardware_accelerator _M_accelerator;
    char _M_delimiter;

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


template <typename Layer>
const hardware_accelerator&
indirect_layer<Layer>::accelerator() const
{
    return _M_value->accelerator();
}


template <typename Layer>
hardware_accelerator&
indirect_layer<Layer>::accelerator()
{
    return _M_value->accelerator();
}


template <typename Layer>
basic_layer&
indirect_layer<Layer>::get_layer(const std::string& name) const
{
    return _M_value->get_layer(name);
}


template <typename Layer>
basic_layer&
indirect_layer<Layer>::get_parent_layer(const std::string& name) const
{
    return _M_value->get_parent_layer(name);
}


template <typename Layer>
template <immutable_tensor Tensor>
void
indirect_layer<Layer>::set_parameter(const std::string& name, Tensor&& tensor)
{
    _M_value->set_parameter(name, std::move(tensor));
}


template <typename Layer>
indirect_layer<Layer>::parameter_pointer
indirect_layer<Layer>::get_parameter(const std::string& name) const
{
    return _M_value->get_parameter(name);
}


template <typename Layer>
const indirect_layer<Layer>::parameter_container
indirect_layer<Layer>::get_parameters(bool recurse) const
{
    return _M_value->get_parameters(recurse);
}


template <typename Layer>
template <std::invocable<named_parameter> Function>
void
indirect_layer<Layer>::apply(Function fn, bool recurse) const
{
    _M_value->apply(fn, recurse);
}


template <typename Layer>
template <std::invocable<named_layer> Function>
void
indirect_layer<Layer>::apply(Function fn) const
{
    _M_value->apply(fn);
}


template <typename Layer>
template <class... Args>
auto
polymorphic_layer<Layer>::operator()(Args&&... args)
{
    auto layer_ptr = _M_layer.lock();
    Layer& layer_impl = dynamic_cast<Layer&>(layer_ptr->get_layer(_M_name));
    return layer_impl(std::forward<Args>(args)...);
}

template <typename Layer>
template <typename DerivedLayer> requires std::derived_from<DerivedLayer, Layer>
polymorphic_layer<Layer>&
polymorphic_layer<Layer>::operator=(indirect_layer<DerivedLayer>&& derived)
{
    auto layer_ptr = _M_layer.lock();
    layer_ptr->register_layer(_M_name, derived);
    return *this;
}


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
        // TODO: Does it make sense to pass in an accelerator, like in basic_layer::register_layer?
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
