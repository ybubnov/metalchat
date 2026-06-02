// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <utility>

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


template <typename T> class unordered_covariant_map {
public:
    using key_type = std::string;
    using pointer = std::shared_ptr<T>;

    using container_type = std::unordered_map<key_type, pointer, _StringHash>;
    using iterator = container_type::iterator;
    using const_iterator = container_type::const_iterator;

    unordered_covariant_map()
    : _M_container()
    {}

    iterator
    begin()
    {
        return _M_container.begin();
    }

    const_iterator
    begin() const
    {
        return _M_container.begin();
    }

    iterator
    end()
    {
        return _M_container.end();
    }

    const_iterator
    end() const
    {
        return _M_container.end();
    }

    iterator
    find(const key_type& name)
    {
        return _M_container.find(name);
    }

    const_iterator
    find(const key_type& name) const
    {
        return _M_container.find(name);
    }

    pointer&
    at(const key_type& name)
    {
        return const_cast<pointer&>(std::as_const(*this).at(name));
    }

    const pointer&
    at(const key_type& name) const
    {
        if (auto it = _M_container.find(name); it != _M_container.end()) {
            return it->second;
        }
        throw std::out_of_range(
            std::format("unordered_covariant_map::find: '{}' key not found", name)
        );
    }

    void
    insert_or_assign(const key_type& name, const pointer& ptr)
    {
        if (auto it = _M_container.find(name); it != _M_container.end()) {
            pointer swap_ptr = ptr;
            it->second.swap(swap_ptr);
        } else {
            _M_container.insert_or_assign(name, ptr);
        }
    }

private:
    std::unordered_map<key_type, pointer, _StringHash> _M_container;
};


template <typename Layer>
concept mutable_layer = std::derived_from<Layer, basic_layer>;


/// A Wrapper around a shared pointer for arbitrary layer implementation provides invocable
/// functionality for `Layer` implementations.
template <mutable_layer Layer> class indirect_layer {
public:
    using layer_type = std::remove_cvref_t<Layer>;
    using layer_pointer = std::shared_ptr<layer_type>;

    using parameter_type = basic_tensor;
    using parameter_pointer = std::shared_ptr<basic_tensor>;

    /// Construct a shared layer with no managed layer, i.e. empty `shared_ptr`.
    indirect_layer()
    : _M_value(nullptr)
    {}

    indirect_layer(indirect_layer&& other) noexcept = default;
    indirect_layer(const indirect_layer& other) noexcept = default;

    /// Construct a shared layer which shares ownership of the layer managed by `r`.
    indirect_layer(const std::shared_ptr<layer_type>& r)
    : _M_value(r)
    {}

    template <typename... Args> requires std::constructible_from<Layer, Args...>
    indirect_layer(Args&&... args)
    : indirect_layer(std::make_shared<layer_type>(std::forward<Args>(args)...))
    {}

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

    indirect_layer&
    operator=(const layer_pointer& r)
    {
        _M_value = r;
        return *this;
    }

    const hardware_accelerator&
    accelerator() const;

    hardware_accelerator&
    accelerator();

    basic_layer&
    layer(const std::string& name);

    const basic_layer&
    layer(const std::string& name) const;

    basic_layer&
    layer_parent(const std::string& name);

    const basic_layer&
    layer_parent(const std::string& name) const;

    template <immutable_tensor Tensor>
    void
    set_parameter(const std::string& name, Tensor&& tensor);

    parameter_type&
    parameter(const std::string& name);

    std::vector<named_parameter>
    parameters(bool recurse = true) const;

    template <std::invocable<named_parameter> Function>
    void
    apply(Function fn, bool recurse = true);

    template <std::invocable<named_parameter> Function>
    void
    apply(Function fn, bool recurse = true) const;

    template <std::invocable<named_layer> Function>
    void
    apply(Function fn);

    virtual ~indirect_layer() = default;

private:
    std::shared_ptr<layer_type> _M_value;
};


template <mutable_layer Layer> indirect_layer(Layer&&) -> indirect_layer<Layer>;


template <mutable_layer Layer> class polymorphic_layer;


/// Layer is a basic building block of neural networks in MetalChat. A layer specifies a set of
/// (trainable) parameters it uses for computation and a set of upstream layers, used within a
/// layer computation logic.
class basic_layer {
public:
    using layer_type = basic_layer;
    using layer_pointer = std::shared_ptr<basic_layer>;

    using parameter_type = basic_tensor;
    using parameter_pointer = std::shared_ptr<basic_tensor>;

    basic_layer(const hardware_accelerator& accelerator, char delimiter);

    /// Construct a layer that is a associated with the specified hardware accelerator.
    basic_layer(const hardware_accelerator& accelerator);

    char
    delimiter() const;

    /// Get a constant reference to the hardware accelerator.
    const hardware_accelerator&
    accelerator() const;

    /// Get a reference to the hardware accelerator.
    hardware_accelerator&
    accelerator();

    layer_type&
    layer(const std::string& name);

    // const layer_type&
    // layer(const std::string& name) const;

    layer_pointer&
    layer_ptr(const std::string& name);

    const layer_pointer&
    layer_ptr(const std::string& name) const;

    layer_type&
    layer_parent(const std::string& name);

    const layer_type&
    layer_parent(const std::string& name) const;

    /// Return a reference to the registered parameter by the specified name.
    ///
    /// This method also supports recursive lookup of the parameter within children layers
    /// if the name contains a dot ('.') delimiter.
    parameter_type&
    parameter(const std::string& name);

    /// \copydoc parameter(const std::string&)
    const parameter_type&
    parameter(const std::string& name) const;

    /// Return a set of parameters with fully-qualified names. Parameters of different layers
    /// are separated using a configured delimiter symbol.
    ///
    /// If you want to return only parameters of the current layer and drop upstream parameters,
    /// you could call this method with `recurse = false`.
    std::vector<named_parameter>
    parameters(bool recurse = true) const;

    /// Return a pointer to the registered parameter by the specified name.
    ///
    /// This method also supports recursive lookup of the parameter within children layers
    /// if the name contains a dot ('.') delimiter.
    parameter_pointer&
    parameter_ptr(const std::string& name);

    /// \copydoc parameter_ptr(const std::string&)
    const parameter_pointer&
    parameter_ptr(const std::string& name) const;

    /// Set value to the registered layer parameter.
    ///
    /// When the specified parameter is not found, the method throws an exception. The method
    /// supports assignment of the nested parameters.
    ///
    /// Example:
    ///
    /// ```cpp
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
        auto& param_ptr = parameter_ptr(name);
        auto tensor_ptr = std::dynamic_pointer_cast<Tensor>(param_ptr);
        if (!tensor_ptr) {
            throw std::invalid_argument(
                "basic_layer::set_parameter: tensor types are not compatible"
            );
        }
        *tensor_ptr = std::move(tensor);
    }

    /// Add a parameter to the layer.
    ///
    /// The parameter can be accessed using  \ref basic_layer::parameter method and updated with
    /// \ref basic_layer::set_parameter method respectively.
    ///
    /// A common practice is registering parameters of the layers that could be updated
    /// externally (loaded from a file, or stored after inference):
    ///
    /// ```cpp
    /// using namespace metalchat;
    ///
    /// struct custom_layer : public layer {
    ///     // Declare parameters here.
    ///     shared_tensor<float, 3> weight;
    ///
    ///     custom_layer(hardware_accelerator accelerator)
    ///     : layer(accelerator)
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
    /// ```cpp
    /// using namespace metalchat;
    ///
    /// struct custom_layer : public layer {
    ///     // Declare parameters here.
    ///     shared_tensor<float, 3> weight;
    ///
    ///     custom_layer(hardware_accelerator accelerator)
    ///     : layer(accelerator),
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
        auto ptr = tensor_ptr.get();
        _M_params.insert_or_assign(name, ptr);
        return tensor_ptr;
    }

    /// Register an upstream layer for the current layer. The layer could be accessed using
    /// the given name using `basic_layer::layer` method.
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
    /// ```cpp
    /// using namespace metalchat;
    ///
    /// struct custom_layer : public layer {
    ///     // Declare upstream layers here.
    ///     nn::indirect_layer<nn::linear<float>> linear1;
    ///     nn::indirect_layer<nn::linear<float>> linear2;
    ///
    ///    custom_layer(hardware_accelerator accelerator)
    ///    : layer(accelerator)
    ///    {
    ///       // Register layers here.
    ///       linear1 = register_layer<nn::linear<float>>("linear1");
    ///       linear2 = register_layer<nn::linear<float>>("linear2");
    ///    }
    /// };
    /// ```
    template <mutable_layer Layer, typename... Args>
    requires std::constructible_from<Layer, Args..., hardware_accelerator&>
    indirect_layer<Layer>
    register_layer(const std::string& name, Args&&... args)
    {
        indirect_layer<Layer> layer(std::forward<Args>(args)..., _M_accelerator);
        return register_layer(name, layer);
    }

    template <mutable_layer Layer>
    indirect_layer<Layer>
    register_layer(const std::string& name, const indirect_layer<Layer>& layer)
    {
        // When there is an existing layer, that is registered as polymorphic
        // (in other words, subscribed to the updates of the layer implementation),
        // replace it with a new implementation.
        if (auto it = _M_polymorphic_pointers.find(name); it != _M_polymorphic_pointers.end()) {
            *(it->second) = layer.get();
        }
        _M_layers.insert_or_assign(name, layer.get());
        return layer;
    }

    template <mutable_layer Layer, typename... Args>
    requires std::constructible_from<Layer, Args..., hardware_accelerator&>
    polymorphic_layer<Layer>
    register_polymorphic_layer(const std::string& name, Args&&... args);

    template <mutable_layer Layer>
    polymorphic_layer<Layer>
    register_polymorphic_layer(const std::string& name, const polymorphic_layer<Layer>& layer);

    template <std::invocable<named_parameter> Function>
    void
    apply(Function fn, bool recurse = true) const
    {
        for (const auto& [param_name, param_ptr] : _M_params) {
            fn(named_parameter{param_name, param_name, param_ptr});
        }
        if (!recurse) {
            return;
        }

        using value_type = std::pair<std::string, layer_pointer>;
        std::deque<value_type> layers;

        for (const auto& [layer_name, layer_ptr] : _M_layers) {
            layers.emplace_back(layer_name, layer_ptr);
        }

        while (!layers.empty()) {
            auto [layer_path, layer_ptr] = layers.front();
            layers.pop_front();

            // Iterate over the downstream layers, and push them back to the queue.
            for (const auto& [child_name, child_layer_ptr] : layer_ptr->_M_layers) {
                auto child_path = layer_path + _M_delimiter + child_name;
                layers.emplace_back(child_path, child_layer_ptr);
            }

            for (const auto& [param_name, param_ptr] : layer_ptr->_M_params) {
                auto param_path = layer_path + _M_delimiter + param_name;
                fn(named_parameter{param_path, param_name, param_ptr});
            }
        }
    }

    /// Apply a function to every parameters of the layer.
    ///
    /// This method traverses all parameters in breadth-first way when `recurse` parameter
    /// is set to `true`. Otherwise, only parameters of the current layer are visited.
    template <std::invocable<named_parameter> Function>
    void
    apply(Function fn, bool recurse = true)
    {
        std::as_const(*this).apply(fn, recurse);
    }

    template <std::invocable<named_layer> Function>
    void
    apply(Function fn)
    {
        using value_type = std::pair<std::string, layer_pointer>;

        std::deque<value_type> layers;
        for (const auto& [layer_name, layer_ptr] : _M_layers) {
            layers.emplace_back(layer_name, layer_ptr);
        }

        while (!layers.empty()) {
            auto [layer_path, layer_ptr] = layers.front();
            layers.pop_front();

            // Iterate over the downstream layers, and push them back to the queue.
            for (const auto& [child_name, child_layer_ptr] : layer_ptr->_M_layers) {
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
    unordered_covariant_map<basic_layer> _M_layers;
    unordered_covariant_map<basic_tensor> _M_params;

    // Every registration of the polymorphic layer should put the pointer into this
    // mapping, so that replacement of a layer through the registration methods would
    // also update the implementations of polymorphic layers.
    unordered_covariant_map<indirect_layer<basic_layer>> _M_polymorphic_pointers;

    hardware_accelerator _M_accelerator;
    char _M_delimiter;
};


template <mutable_layer Layer>
const hardware_accelerator&
indirect_layer<Layer>::accelerator() const
{
    return _M_value->accelerator();
}


template <mutable_layer Layer>
hardware_accelerator&
indirect_layer<Layer>::accelerator()
{
    return _M_value->accelerator();
}


template <mutable_layer Layer>
basic_layer&
indirect_layer<Layer>::layer(const std::string& name)
{
    return _M_value->layer(name);
}


template <mutable_layer Layer>
const basic_layer&
indirect_layer<Layer>::layer(const std::string& name) const
{
    return _M_value->layer(name);
}


template <mutable_layer Layer>
basic_layer&
indirect_layer<Layer>::layer_parent(const std::string& name)
{
    return _M_value->layer_parent(name);
}


template <mutable_layer Layer>
const basic_layer&
indirect_layer<Layer>::layer_parent(const std::string& name) const
{
    return _M_value->layer_parent(name);
}


template <mutable_layer Layer>
template <immutable_tensor Tensor>
void
indirect_layer<Layer>::set_parameter(const std::string& name, Tensor&& tensor)
{
    _M_value->set_parameter(name, std::move(tensor));
}


template <mutable_layer Layer>
indirect_layer<Layer>::parameter_type&
indirect_layer<Layer>::parameter(const std::string& name)
{
    return _M_value->parameter(name);
}


template <mutable_layer Layer>
std::vector<named_parameter>
indirect_layer<Layer>::parameters(bool recurse) const
{
    return _M_value->parameters(recurse);
}


template <mutable_layer Layer>
template <std::invocable<named_parameter> Function>
void
indirect_layer<Layer>::apply(Function fn, bool recurse)
{
    _M_value->apply(fn, recurse);
}


template <mutable_layer Layer>
template <std::invocable<named_layer> Function>
void
indirect_layer<Layer>::apply(Function fn)
{
    _M_value->apply(fn);
}


template <mutable_layer Layer> class polymorphic_layer {
public:
    using layer_type = std::remove_cvref_t<Layer>;

    polymorphic_layer(const std::shared_ptr<indirect_layer<basic_layer>>& layer_ptr)
    : _M_value(layer_ptr)
    {}

    polymorphic_layer(const indirect_layer<Layer>& layer)
    : polymorphic_layer(std::make_shared<indirect_layer<basic_layer>>(layer.get()))
    {}

    polymorphic_layer()
    : _M_value(nullptr)
    {}

    std::shared_ptr<layer_type>
    get() const
    {
        return std::dynamic_pointer_cast<layer_type>(_M_value->get());
    }

    std::shared_ptr<indirect_layer<basic_layer>>&
    operator*() noexcept
    {
        return _M_value;
    }

    const std::shared_ptr<indirect_layer<basic_layer>>&
    operator*() const noexcept
    {
        return _M_value;
    }

    template <class... Args>
    auto
    operator()(Args&&... args)
    {
        auto& layer = *get();
        return layer(std::forward<Args>(args)...);
    }

    template <std::derived_from<Layer> DerivedLayer>
    polymorphic_layer&
    operator=(indirect_layer<DerivedLayer>&& derived)
    {
        _M_value = std::make_shared<indirect_layer<basic_layer>>(std::move(derived));
        return *this;
    }

    template <std::derived_from<Layer> DerivedLayer>
    polymorphic_layer&
    operator=(polymorphic_layer<DerivedLayer>&& derived)
    {
        _M_value = derived._M_value;
        return *this;
    }

private:
    template <mutable_layer FriendLayer> friend class polymorphic_layer;

    std::shared_ptr<indirect_layer<basic_layer>> _M_value;
};


template <mutable_layer Layer, typename... Args>
requires std::constructible_from<Layer, Args..., hardware_accelerator&>
polymorphic_layer<Layer>
basic_layer::register_polymorphic_layer(const std::string& name, Args&&... args)
{
    auto layer = register_layer<Layer>(name, std::forward<Args>(args)...);
    return register_polymorphic_layer(name, polymorphic_layer<Layer>(layer));
}


template <mutable_layer Layer>
polymorphic_layer<Layer>
basic_layer::register_polymorphic_layer(
    const std::string& name, const polymorphic_layer<Layer>& layer
)
{
    _M_polymorphic_pointers.insert_or_assign(name, *layer);
    return layer;
}


template <typename Predicate>
concept named_layer_predicate = requires(const Predicate predicate) {
    { predicate(std::declval<named_layer>()) } -> std::same_as<bool>;
};


template <named_layer_predicate... Predicates> struct layer_match_all {
    std::tuple<Predicates...> predicates;

    layer_match_all(Predicates... preds)
    : predicates(std::make_tuple(preds...))
    {}

    bool
    operator()(named_layer layer) const
    {
        return invoke_all(layer, std::index_sequence_for<Predicates...>{});
    }

private:
    template <std::size_t... PredicateIndices>
    bool
    invoke_all(named_layer& layer, std::index_sequence<PredicateIndices...>) const
    {
        return (true && ... && invoke<PredicateIndices>(layer));
    }

    template <std::size_t PredicateIndex>
    bool
    invoke(named_layer& layer) const
    {
        auto pred = std::get<PredicateIndex>(predicates);
        return pred(layer);
    }
};


/// A \ref nn::named_layer_predicate implementation that returns true in case, when the layer
/// type is possible to dynamically-cast to the specified layer type.
///
/// \tparam Layer a layer type that is used to assert possibility of dynamic cast.
template <mutable_layer Layer> struct layer_common_with {
    bool
    operator()(named_layer layer) const
    {
        return std::dynamic_pointer_cast<Layer>(layer.ptr) != nullptr;
    }
};


/// A \ref nn::named_layer_predicate implementation that returns true in case, when the layer
/// name matches the specified regular expression.
///
/// \param regex a regular expression to match the layer name.
struct layer_match_name {
    const std::regex re;

    layer_match_name(const std::string& regex)
    : re(regex)
    {}

    bool
    operator()(named_layer layer) const
    {
        return std::regex_match(layer.name, re);
    }
};


/// Replaces all matches of the layers with the specified predicate by replacing them.
///
/// Method implements replacement in two phases: (1) traversal through the layers in a search
/// for matching candidates, (2) replacement of layers through re-registration.
///
/// The method is not transactional and might leave the original layer partially processed,
/// if an exception is encountered during the replacement phase.
///
/// \param input a layer that will be searched for matching layers for replacement.
/// \param pred a predicate invoked for each layer in a search loop.
/// \param generator a generator of \ref nn::indirect_layer instances used for replacement.
///
/// ```cpp
/// using namespace metalchat;
///
/// using LLama3 = nn::llama3<bf16>;
/// auto gpu = hardware_accelerator()
/// auto llm = nn::indirect_layer<LLama3>(/* ... */, gpu);
///
/// using BasicLinear = nn::basic_linear<bf16>;
/// using TargetLayer = quantization::lora_linear<bf16>;
/// auto pred = layer_common_with<BasicLinear>();
///
/// replace_layer(llm, pred, [&]() {
///     return nn:indirect_layer<TargetLayer>(gpu);
/// });
/// ```
template <mutable_layer Layer, named_layer_predicate Pred, typename Generator>
void
replace_layer(nn::indirect_layer<Layer>& input, Pred pred, Generator generator)
{
    std::vector<nn::named_layer> candidates;

    auto find_candidates = [&](nn::named_layer layer) {
        if (pred(layer)) {
            candidates.push_back(layer);
        }
    };

    input->apply(find_candidates);

    for (auto& layer : candidates) {
        auto& layer_parent = input.layer_parent(layer.path);
        layer_parent.register_layer(layer.name, generator());
    }
}


/// Replaces all matches of the layers with the provided replacement.
///
/// \warning the method assigns a shallow copy of the layer, therefore all replacements will be
/// sharing a pointer to the same layer instance.
///
/// \param input a layer that will be searched for matching layers for replacement.
/// \param pred a predicate invoked for each layer in a search loop.
/// \param replacement a replacement layer that will be assigned in each replacement case.
template <mutable_layer Layer, named_layer_predicate Pred, mutable_layer Replacement>
void
replace_layer(
    nn::indirect_layer<Layer>& input, Pred pred, nn::indirect_layer<Replacement> replacement
)
{
    return replace_layer(input, pred, [&]() { return replacement; });
}


} // namespace nn
} // namespace metalchat
