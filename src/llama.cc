// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <jsoncons/json.hpp>

#include <metalchat/nn/options.h>


namespace metalchat {
namespace nn {


llama3_options
default_llama3_1b_options()
{
    return llama3_options()
        .head_dim(64)
        .n_heads(32)
        .n_kv_heads(8)
        .n_layers(16)
        .max_seq_len(1024)
        .rope_theta(500000.0f)
        .norm_eps(1e-5);
}


llama3_options::llama3_options()
: _M_max_seq_len(1024)
{}


llama3_options
llama3_options::head_dim(std::size_t value) const noexcept
{
    llama3_options o = *this;
    o._M_head_dim = value;
    return o;
}


std::size_t
llama3_options::head_dim() const noexcept
{
    return _M_head_dim;
}


llama3_options
llama3_options::n_heads(std::size_t value) const noexcept
{
    llama3_options o = *this;
    o._M_n_heads = value;
    return o;
}


std::size_t
llama3_options::n_heads() const noexcept
{
    return _M_n_heads;
}


llama3_options
llama3_options::n_kv_heads(std::size_t value) const noexcept
{
    llama3_options o = *this;
    o._M_n_kv_heads = value;
    return o;
}


std::size_t
llama3_options::n_kv_heads() const noexcept
{
    return _M_n_kv_heads;
}


llama3_options
llama3_options::n_layers(std::size_t value) const noexcept
{
    llama3_options o = *this;
    o._M_n_layers = value;
    return o;
}


std::size_t
llama3_options::n_layers() const noexcept
{
    return _M_n_layers;
}


llama3_options
llama3_options::max_seq_len(std::size_t value) const noexcept
{
    llama3_options o = *this;
    o._M_max_seq_len = value;
    return o;
}


std::size_t
llama3_options::max_seq_len() const noexcept
{
    return _M_max_seq_len;
}


llama3_options
llama3_options::rope_theta(float value) const noexcept
{
    llama3_options o = *this;
    o._M_rope_theta = value;
    return o;
}


float
llama3_options::rope_theta() const noexcept
{
    return _M_rope_theta;
}


llama3_options
llama3_options::norm_eps(float value) const noexcept
{
    llama3_options o = *this;
    o._M_norm_eps = value;
    return o;
}


float
llama3_options::norm_eps() const noexcept
{
    return _M_norm_eps;
}


} // namespace nn
} // namespace metalchat
