// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metalchat/accelerator.h>
#include <metalchat/kernel.h>
#include <metalchat/nn/layer.h>


namespace metalchat {
namespace nn {


basic_layer::basic_layer(const hardware_accelerator& accelerator, char delimiter)
: _M_layers(),
  _M_params(),
  _M_polymorphic_pointers(),
  _M_accelerator(accelerator),
  _M_delimiter(delimiter)
{}


basic_layer::basic_layer(const hardware_accelerator& accelerator)
: basic_layer(accelerator, '.')
{}


char
basic_layer::delimiter() const
{
    return _M_delimiter;
}


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


basic_layer::layer_type&
basic_layer::layer(const std::string& name)
{
    return *layer_ptr(name);
}


const basic_layer::layer_pointer&
basic_layer::layer_ptr(const std::string& name) const
{
    auto layer = this;
    auto delim_pos = name.rfind(delimiter());
    auto layer_name = name;

    if (delim_pos != std::string::npos) {
        layer_name = name.substr(delim_pos + 1);
        layer = &layer_parent(name);
    }

    try {
        return layer->_M_layers.at(layer_name);
    } catch (std::out_of_range) {
        throw std::runtime_error(
            std::format("nn::basic_layer::layer_ptr: '{}' is not registered", name)
        );
    }
}


basic_layer::layer_pointer&
basic_layer::layer_ptr(const std::string& name)
{
    return const_cast<layer_pointer&>(std::as_const(*this).layer_ptr(name));
}


const basic_layer::layer_type&
basic_layer::layer_parent(const std::string& name) const
{
    const basic_layer* this_layer = this;
    std::size_t start_pos = 0, pos = 0;

    for (pos = name.find(delimiter()); pos != std::string::npos;
         pos = name.find(delimiter(), start_pos)) {
        const auto layer_name = name.substr(start_pos, pos - start_pos);
        start_pos = pos + 1;

        try {
            this_layer = this_layer->_M_layers.at(layer_name).get();
        } catch (std::out_of_range) {
            throw std::runtime_error(
                std::format("nn::basic_layer::layer_parent: '{}' is not registered", name)
            );
        }
    }

    return *this_layer;
}


basic_layer::layer_type&
basic_layer::layer_parent(const std::string& name)
{
    return const_cast<layer_type&>(std::as_const(*this).layer_parent(name));
}


basic_layer::parameter_type&
basic_layer::parameter(const std::string& name)
{
    return *parameter_ptr(name);
}


basic_layer::parameter_pointer&
basic_layer::parameter_ptr(const std::string& name)
{
    return const_cast<parameter_pointer&>(std::as_const(*this).parameter_ptr(name));
}


const basic_layer::parameter_pointer&
basic_layer::parameter_ptr(const std::string& name) const
{
    auto delim_pos = name.rfind(delimiter());
    auto param_name = name;

    // Create a shared pointer without a deleter to prevent freeing
    // the memory of the current class instance.
    std::shared_ptr<const basic_layer> layer(this, [](const basic_layer*) {});

    if (delim_pos != std::string::npos) {
        param_name = name.substr(delim_pos + 1);
        layer = layer_ptr(name.substr(0, delim_pos));
    }

    try {
        return layer->_M_params.at(param_name);
    } catch (std::out_of_range) {
        throw std::runtime_error(
            std::format("nn::basic_layer::parameter_ptr: '{}' is not registered", name)
        );
    }
}


std::vector<named_parameter>
basic_layer::parameters(bool recurse) const
{
    std::vector<named_parameter> params;

    auto fn = [&](named_parameter parameter) { params.push_back(parameter); };

    apply(fn, recurse);
    return params;
}


} // namespace nn
} // namespace metalchat
