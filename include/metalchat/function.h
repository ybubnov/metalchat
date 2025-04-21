#pragma once

#include <functional>
#include <optional>

#include <metalchat/tensor_concept.h>
#include <metalchat/tensor_polymorphic.h>


namespace metalchat {


class function {
public:
    void
    register_function(const std::string name, function& fn)
    {
        _m_funcs.insert_or_assign(name, std::reference_wrapper(fn));
    }

    void
    register_parameter(const std::string name, polymorphic_tensor tensor)
    {
        _m_params.insert_or_assign(name, tensor);
    }

    template <immutable_tensor Tensor>
    void
    register_parameter(const std::string name, Tensor&& tensor)
    {
        register_parameter(name, polymorphic_tensor(std::move(tensor)));
    }

    template <immutable_tensor Tensor>
    void
    register_parameter(const std::string name, std::shared_ptr<Tensor> tensor_ptr)
    {
        register_parameter(name, polymorphic_tensor(tensor_ptr));
    }

    template <immutable_tensor Tensor>
    void
    set_parameter(const std::string name, Tensor&& tensor)
    {
        if (auto it = _m_params.find(name); it != _m_params.end()) {
            it->second.emplace(std::move(tensor));
        } else {
            throw std::invalid_argument(std::format("parameter '{}' is not registered", name));
        }
    }

    polymorphic_tensor
    get_parameter(const std::string name) const
    {
        if (auto it = _m_params.find(name); it != _m_params.end()) {
            return it->second;
        }
        throw std::invalid_argument(std::format("parameter '{}' is not registered", name));
    }

private:
    std::unordered_map<std::string, polymorphic_tensor> _m_params;
    std::unordered_map<std::string, std::reference_wrapper<function>> _m_funcs;
};


} // namespace metalchat
