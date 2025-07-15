#include <metalchat/accelerator.h>
#include <metalchat/kernel.h>
#include <metalchat/layer.h>


namespace metalchat {


layer::layer(const hardware_accelerator& accelerator)
: _m_layers(),
  _m_params(),
  _m_accelerator(accelerator)
{}


const hardware_accelerator&
layer::accelerator() const
{
    return _m_accelerator;
}


hardware_accelerator&
layer::accelerator()
{
    return _m_accelerator;
}


const layer&
layer::get_layer(const std::string& name) const
{
    try {
        return (*_m_layers.at(name).get());
    } catch (std::out_of_range) {
        throw std::runtime_error(std::format("layer '{}' is not registered", name));
    }
}


void
layer::register_parameter(const std::string& name, polymorphic_tensor tensor)
{
    _m_params.insert_or_assign(name, tensor);
}


polymorphic_tensor
layer::get_parameter(const std::string& name) const
{
    if (auto it = _m_params.find(name); it != _m_params.end()) {
        return it->second;
    }

    const layer* this_layer = this;

    std::size_t start_pos = 0, pos = 0;
    const char delimiter = '.';

    for (pos = name.find(delimiter); pos != name.npos; pos = name.find(delimiter, start_pos)) {
        const auto layer_name = name.substr(start_pos, pos - start_pos);
        start_pos = pos + 1;

        this_layer = &this_layer->get_layer(layer_name);
    }

    auto param_name = name.substr(start_pos);
    if (auto it = this_layer->_m_params.find(param_name); it != this_layer->_m_params.end()) {
        return it->second;
    }

    throw std::invalid_argument(std::format("parameter '{}' is not registered", name));
}


const layer::parameter_container
layer::get_parameters(bool recurse) const
{
    parameter_container params;

    auto visitor = [&](const std::string& name, polymorphic_tensor param) {
        params.insert_or_assign(name, param);
    };

    visit_parameters(visitor, recurse);
    return params;
}


} // namespace metalchat
