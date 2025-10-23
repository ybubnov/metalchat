#include <metalchat/accelerator.h>
#include <metalchat/kernel.h>
#include <metalchat/nn/layer.h>


namespace metalchat {
namespace nn {


basic_layer::basic_layer(const hardware_accelerator& accelerator)
: _M_layers(),
  _M_params(),
  _M_accelerator(accelerator)
{}


const hardware_accelerator&
basic_layer::accelerator() const
{
    return _M_accelerator;
}


hardware_accelerator&
basic_layer::accelerator()
{
    return _M_accelerator;
}


const basic_layer&
basic_layer::get_layer(const std::string& name) const
{
    try {
        return (*_M_layers.at(name).get());
    } catch (std::out_of_range) {
        throw std::runtime_error(std::format("layer '{}' is not registered", name));
    }
}


basic_layer::parameter_pointer
basic_layer::get_parameter(const std::string& name) const
{
    if (auto it = _M_params.find(name); it != _M_params.end()) {
        return it->second;
    }

    const basic_layer* this_layer = this;

    std::size_t start_pos = 0, pos = 0;
    const char delimiter = '.';

    for (pos = name.find(delimiter); pos != name.npos; pos = name.find(delimiter, start_pos)) {
        const auto layer_name = name.substr(start_pos, pos - start_pos);
        start_pos = pos + 1;

        this_layer = &this_layer->get_layer(layer_name);
    }

    auto param_name = name.substr(start_pos);
    if (auto it = this_layer->_M_params.find(param_name); it != this_layer->_M_params.end()) {
        return it->second;
    }

    throw std::invalid_argument(std::format("parameter '{}' is not registered", name));
}


const basic_layer::parameter_container
basic_layer::get_parameters(bool recurse) const
{
    parameter_container params;

    auto fn = [&](const std::string& name, parameter_pointer param) {
        params.insert_or_assign(name, param);
    };

    apply(fn, recurse);
    return params;
}


} // namespace nn
} // namespace metalchat
