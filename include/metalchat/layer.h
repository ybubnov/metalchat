#pragma once

#include <deque>
#include <functional>
#include <optional>

#include <metalchat/safetensor.h>
#include <metalchat/tensor/concept.h>
#include <metalchat/tensor/polymorphic.h>


namespace metalchat {


template <class T, T... Indices, class Function>
void
constexpr_switch(T index, std::integer_sequence<T, Indices...>, Function function)
{
    std::initializer_list<int>(
        {(index == Indices ? function(std::integral_constant<T, Indices>{}), 0 : 0)...}
    );
}


template <typename Layer> class shared_layer {
public:
    shared_layer()
    : _m_value(nullptr)
    {}

    shared_layer(Layer&& layer)
    : _m_value(std::move(layer))
    {}

    shared_layer(std::shared_ptr<Layer> shared_layer)
    : _m_value(shared_layer)
    {}

    template <class... Args>
    auto
    operator()(Args&&... args)
    {
        return (*_m_value)(std::forward<Args>(args)...);
    }

    std::shared_ptr<Layer>
    get()
    {
        return _m_value;
    }

    Layer*
    operator->() noexcept
    {
        return _m_value;
    }

    Layer&
    operator*() noexcept
    {
        return (*_m_value);
    }

private:
    std::shared_ptr<Layer> _m_value;
};


/// Layer is a basic building block of neural networks in MetalChat. A layer specifies a set of
/// (trainable) parameters it uses for computation and a set of upstream layers, used within a
/// layer computation logic.
class layer {
public:
    using pointer = std::shared_ptr<layer>;

    using parameter_container = std::unordered_map<std::string, polymorphic_tensor>;
    using layer_container = std::unordered_map<std::string, pointer>;

    layer(hardware_accelerator accelerator)
    : _m_layers(),
      _m_params(),
      _m_accelerator(accelerator)
    {}

    const hardware_accelerator&
    accelerator() const
    {
        return _m_accelerator;
    }

    hardware_accelerator&
    accelerator()
    {
        return _m_accelerator;
    }

    /// Initialize a layer and all upstream layers with a given safetensor file.
    ///
    /// This method uses a parameter `N` to define the maximum number of dimensions of tensors
    /// to allocate. From the efficiency perspective it is limited by 8, but could be extended
    /// up to arbitrary number of dimensions.
    template <allocator Allocator, std::size_t N = 8>
    void
    initialize(const safetensor_file& weights, Allocator alloc)
    {
        auto visitor = [&](const std::string& param_name, polymorphic_tensor param) {
            if (auto it = weights.find(param_name); it != weights.end()) {
                auto [_, weight] = *it;
                std::size_t dim = weight.dim();

                constexpr_switch(dim, std::make_index_sequence<N>{}, [&](auto i) {
                    param.emplace(std::move(weight.as<i, Allocator>(alloc)));
                });
            }
        };

        visit_parameters(visitor);
    }

    /// Register an upstream layer for the current layer. The layer could be accessed using
    /// the given name using `layer::get_layer` method.
    ///
    /// The registry of layers owns the upstream layer, and the method returns a object pointing
    /// to that owned layer.
    ///
    /// A common practice is registering upstream layers within a downstream layer constructor
    /// like in the example below.
    ///
    /// ```cpp
    /// using namespace metalchat;
    ///
    /// class custom_layer : public layer {
    /// private:
    ///     // Declare upstream layers here.
    ///     nn::shared_linear<float> linear1;
    ///     nn::shared_linear<float> linear2;
    ///
    /// public:
    ///    custom_layer(hardware_accelerator accelerator)
    ///    : layer(accelerator)
    ///    {
    ///       // Register layers here.
    ///       linear1 = register_layer("linear1", nn::linear<float>(accelerator));
    ///       linear2 = register_layer("linear2", nn::linear<float>(accelerator));
    ///    }
    /// };
    /// ```
    template <typename Layer>
    shared_layer<Layer>
    register_layer(const std::string& name, Layer&& l)
    {
        auto layer_ptr = std::make_shared<Layer>(std::move(l));
        _m_layers.emplace(name, layer_ptr);
        return shared_layer(layer_ptr);
    }

    /// Get upstream layer by name. This method does not perform recursive lookup and only
    /// returns layers registered at the current layer. If layer is not registered, method
    /// throws exception.
    const layer&
    get_layer(const std::string& name) const
    {
        return (*_m_layers.at(name).get());
    }

    void
    register_parameter(const std::string& name, polymorphic_tensor tensor)
    {
        _m_params.insert_or_assign(name, tensor);
    }

    template <immutable_tensor Tensor>
    void
    register_parameter(const std::string& name, Tensor&& tensor)
    {
        register_parameter(name, polymorphic_tensor(std::move(tensor)));
    }

    template <immutable_tensor Tensor>
    void
    register_parameter(const std::string& name, std::shared_ptr<Tensor> tensor_ptr)
    {
        register_parameter(name, polymorphic_tensor(tensor_ptr));
    }

    template <immutable_tensor Tensor>
    void
    set_parameter(const std::string& name, Tensor&& tensor)
    {
        if (auto it = _m_params.find(name); it != _m_params.end()) {
            it->second.emplace(std::move(tensor));
        } else {
            throw std::invalid_argument(std::format("parameter '{}' is not registered", name));
        }
    }

    polymorphic_tensor
    get_parameter(const std::string& name) const
    {
        if (auto it = _m_params.find(name); it != _m_params.end()) {
            return it->second;
        } else {
            throw std::invalid_argument(std::format("parameter '{}' is not registered", name));
        }
    }

    /// Return a set of parameters with fully-qualified names. Parameters of different layers
    /// are separated using dot (".") delimiter symbol.
    ///
    /// If you want to return only parameters of the current layer and drop upstream parameters,
    /// you could call this method with `recurse = false`.
    const parameter_container
    get_parameters(bool recurse = true) const
    {
        parameter_container params;

        auto visitor = [&](const std::string& name, polymorphic_tensor param) {
            params.insert_or_assign(name, param);
        };

        visit_parameters(visitor, recurse);
        return params;
    }

    template <std::invocable<const std::string&, polymorphic_tensor> Visitor>
    void
    visit_parameters(Visitor visitor, bool recurse = true) const
    {
        for (const auto& [full_name, param] : _m_params) {
            visitor(full_name, param);
        }

        if (!recurse) {
            return;
        }

        using layer_type = layer_container::value_type;
        std::deque<layer_type> layers(_m_layers.begin(), _m_layers.end());

        while (!layers.empty()) {
            auto [name, layer_ptr] = layers.front();
            layers.pop_front();

            // Iterate over the downstream layers, and push them back to the queue.
            for (auto [child_name, child_layer_ref] : layer_ptr->_m_layers) {
                auto full_name = name + "." + child_name;
                layers.emplace_back(full_name, child_layer_ref);
            }

            for (auto [param_name, param] : layer_ptr->_m_params) {
                auto full_name = name + "." + param_name;
                visitor(full_name, param);
            }
        }
    }

private:
    parameter_container _m_params;
    layer_container _m_layers;
    hardware_accelerator _m_accelerator;
};


} // namespace metalchat
