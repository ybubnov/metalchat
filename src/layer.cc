// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metalchat/accelerator.h>
#include <metalchat/kernel.h>
#include <metalchat/nn/layer.h>


namespace metalchat {
namespace nn {


basic_layer::basic_layer(char delimiter, const hardware_accelerator& accelerator)
: _M_layers(),
  _M_params(),
  _M_accelerator(accelerator),
  _M_delimiter(delimiter)
{}


basic_layer::basic_layer(const hardware_accelerator& accelerator)
: basic_layer('.', accelerator)
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


void
basic_layer::initialize()
{}


basic_layer&
basic_layer::get_parent_layer(const std::string& name) const
{
    basic_layer* this_layer = const_cast<basic_layer*>(this);
    std::size_t start_pos = 0, pos = 0;

    for (pos = name.find(_M_delimiter); pos != std::string::npos;
         pos = name.find(_M_delimiter, start_pos)) {
        const auto layer_name = name.substr(start_pos, pos - start_pos);
        start_pos = pos + 1;

        try {
            this_layer = this_layer->_M_layers.at(layer_name).get();
        } catch (std::out_of_range) {
            throw std::runtime_error(std::format("layer '{}' is not registered", name));
        }
    }

    return *this_layer;
}


basic_layer&
basic_layer::get_layer(const std::string& name) const
{
    const basic_layer* this_layer = this;
    auto delim_pos = name.rfind(_M_delimiter);
    auto layer_name = name;

    if (delim_pos != std::string::npos) {
        layer_name = name.substr(delim_pos + 1);
        this_layer = &get_parent_layer(name);
    }

    if (auto it = this_layer->_M_layers.find(layer_name); it != this_layer->_M_layers.end()) {
        return *(it->second);
    }

    throw std::runtime_error(std::format("layer '{}' is not registered", name));
}


basic_layer::parameter_pointer
basic_layer::get_parameter(const std::string& name) const
{
    const basic_layer* this_layer = this;
    auto delim_pos = name.rfind(_M_delimiter);
    auto param_name = name;

    if (delim_pos != std::string::npos) {
        param_name = name.substr(delim_pos + 1);
        this_layer = &get_layer(name.substr(0, delim_pos));
    }

    if (auto it = this_layer->_M_params.find(param_name); it != this_layer->_M_params.end()) {
        return it->second;
    }

    throw std::invalid_argument(std::format("parameter '{}' is not registered", name));
}


const basic_layer::parameter_container
basic_layer::get_parameters(bool recurse) const
{
    parameter_container params;

    auto fn = [&](named_parameter parameter) {
        params.insert_or_assign(parameter.path, parameter.ptr);
    };

    apply(fn, recurse);
    return params;
}


} // namespace nn
} // namespace metalchat
