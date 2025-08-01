#pragma once

#include <deque>
#include <functional>
#include <optional>

#include <metalchat/safetensor.h>
#include <metalchat/tensor/concept.h>


namespace metalchat {


template <class T, T... Indices, class Function>
void
constexpr_switch(T index, std::integer_sequence<T, Indices...>, Function function)
{
    std::initializer_list<int>(
        {(index == Indices ? function(std::integral_constant<T, Indices>{}), 0 : 0)...}
    );
}


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

    template <class... Args>
    auto
    operator()(Args&&... args)
    {
        return (*_M_value)(std::forward<Args>(args)...);
    }

    /// Return the raw shared pointer to the layer.
    std::shared_ptr<Layer>
    get()
    {
        return _M_value;
    }

    /// Dereference the stored pointer to the Layer.
    Layer*
    operator->() noexcept
    {
        return _M_value;
    }

    /// Dereference the stored pointer to the Layer.
    Layer&
    operator*() noexcept
    {
        return (*_M_value);
    }

private:
    std::shared_ptr<Layer> _M_value;
};


/// Layer is a basic building block of neural networks in MetalChat. A layer specifies a set of
/// (trainable) parameters it uses for computation and a set of upstream layers, used within a
/// layer computation logic.
class basic_layer {
public:
    using parameter_pointer = std::shared_ptr<basic_tensor>;
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

    /// Initialize a layer and all upstream layers with a given safetensor file.
    ///
    /// This method uses a parameter `N` to define the maximum number of dimensions of tensors
    /// to allocate. From the efficiency perspective it is limited by 8, but could be extended
    /// up to arbitrary number of dimensions.
    template <contiguous_container Container, std::size_t N = 8>
    void
    initialize(const std::unordered_map<std::string, safetensor<Container>>& weights)
    {
        auto visitor = [&](const std::string& param_name, parameter_pointer param) {
            if (auto it = weights.find(param_name); it != weights.end()) {
                auto& [_, weight] = *it;
                std::size_t dim = weight.dim();

                constexpr_switch(dim, std::make_index_sequence<N>{}, [&](auto i) {
                    auto sizes = weight.sizes();

                    using value_type = typename Container::value_type;
                    using tensor_type = tensor<value_type, i, Container>;

                    auto tensor = tensor_type(sizes.begin(), sizes.end(), weight.container());
                    move_tensor_to_pointer(param, std::move(tensor));
                });
            }
        };

        visit_parameters(visitor);
    }

    /// Register an upstream layer for the current layer. The layer could be accessed using
    /// the given name using `basic_layer::get_layer` method.
    ///
    /// The registry of layers owns the upstream layer, and the method returns a object pointing
    /// to that owned layer.
    ///
    /// A common practice is registering upstream layers within a downstream layer constructor
    /// like in the example below.
    ///
    /// ```c++
    /// using namespace metalchat;
    ///
    /// class custom_layer : public basic_layer {
    /// private:
    ///     // Declare upstream layers here.
    ///     nn::shared_linear<float> linear1;
    ///     nn::shared_linear<float> linear2;
    ///
    /// public:
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

    /// Get upstream layer by name. This method does not perform recursive lookup and only
    /// returns layers registered at the current layer. If layer is not registered, method
    /// throws exception.
    const basic_layer&
    get_layer(const std::string& name) const;

    /// Add a parameter to the layer.
    ///
    /// The parameter can be accessed using `get_parameter` method and updated with `set_parameter`
    /// method.
    ///
    /// A common practice is registering parameters of the layers that could be updated
    /// externally (loaded from a file, or stored after inference):
    ///
    /// ```c++
    /// using namespace metalchat;
    ///
    /// class custom_layer : public basic_layer {
    /// private:
    ///     // Declare parameters here.
    ///     shared_tensor<float, 3> weight;
    ///
    /// public:
    ///     custom_layer(hardware_accelerator accelerator)
    ///     : basic_layer(accelerator)
    ///     {
    ///         weight = register_parameter("weight", empty<float>({10, 4, 3}, accelerator));
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

    /// Add a parameter to the layer.
    template <immutable_tensor Tensor>
    shared_tensor_ptr<Tensor>
    register_parameter(const std::string& name, Tensor&& tensor)
    {
        auto tensor_ptr = shared_tensor_ptr(std::move(tensor));
        return register_parameter(name, tensor_ptr);
    }

    template <immutable_tensor Tensor>
    void
    set_parameter(const std::string& name, Tensor&& tensor)
    {
        if (auto it = _M_params.find(name); it != _M_params.end()) {
            move_tensor_to_pointer(it->second, std::move(tensor));
        } else {
            throw std::invalid_argument(std::format("parameter '{}' is not registered", name));
        }
    }

    parameter_pointer
    get_parameter(const std::string& name) const;

    /// Return a set of parameters with fully-qualified names. Parameters of different layers
    /// are separated using dot (".") delimiter symbol.
    ///
    /// If you want to return only parameters of the current layer and drop upstream parameters,
    /// you could call this method with `recurse = false`.
    const parameter_container
    get_parameters(bool recurse = true) const;

    template <std::invocable<const std::string&, parameter_pointer> Visitor>
    void
    visit_parameters(Visitor visitor, bool recurse = true) const
    {
        for (const auto& [full_name, param] : _M_params) {
            visitor(full_name, param);
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
                visitor(full_name, param);
            }
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


} // namespace metalchat
