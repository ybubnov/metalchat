#pragma once

#include <deque>
#include <functional>
#include <optional>

#include <metalchat/safetensor.h>
#include <metalchat/tensor_concept.h>
#include <metalchat/tensor_polymorphic.h>


namespace metalchat {


class layer {
public:
    using reference = std::reference_wrapper<layer>;

    using parameter_container = std::unordered_map<std::string, polymorphic_tensor>;
    using layer_container = std::unordered_map<std::string, reference>;

    layer(layer&&) = default;
    layer(const layer&) = delete;

    layer()
    : _m_layers(),
      _m_params()
    {}

    template <allocator Allocator>
    void
    initialize(const safetensor_file& weights, Allocator alloc)
    {
        auto params = get_parameters();
        for (auto [param_name, param] : params) {
            if (auto it = weights.find(param_name); it != weights.end()) {
                auto [_, weight] = *it;

                switch (param.dimensions()) {
                case 1:
                    param.emplace(std::move(weight.as<1, Allocator>(alloc)));
                    break;
                case 2:
                    param.emplace(std::move(weight.as<2, Allocator>(alloc)));
                    break;
                case 3:
                    param.emplace(std::move(weight.as<3, Allocator>(alloc)));
                    break;
                case 4:
                    param.emplace(std::move(weight.as<4, Allocator>(alloc)));
                    break;
                default:
                    throw std::runtime_error(
                        std::format("unsupported tensor dimensionality = {}", param.dimensions())
                    );
                }
            }
        }
    }

    void
    register_layer(const std::string& name, layer& l)
    {
        _m_layers.insert_or_assign(name, std::ref(l));
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

    const parameter_container
    get_parameters(bool recurse = true) const
    {
        parameter_container params(_m_params.begin(), _m_params.end());

        if (!recurse) {
            return params;
        }

        using layer_type = layer_container::value_type;
        std::deque<layer_type> layers(_m_layers.begin(), _m_layers.end());

        while (!layers.empty()) {
            auto [name, layer] = layers.front();
            layers.pop_front();

            // Iterate over the downstream layers, and push them back to the queue.
            for (auto [child_name, child_layer] : layer.get()._m_layers) {
                auto full_name = name + "." + child_name;
                layers.emplace_back(full_name, child_layer);
            }

            for (auto [param_name, param] : layer.get()._m_params) {
                auto full_name = name + "." + param_name;
                params.insert_or_assign(full_name, param);
            }
        }

        return params;
    }

private:
    parameter_container _m_params;
    layer_container _m_layers;
};


} // namespace metalchat
