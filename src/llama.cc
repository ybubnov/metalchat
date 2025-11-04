// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metalchat/nn/llama.h>


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
        .heap_size(std::size_t(512) * 1024 * 1024);
}


void
llama3_options::set_head_dim(std::size_t head_dim)
{
    _M_head_dim = head_dim;
}


llama3_options
llama3_options::head_dim(std::optional<std::size_t> head_dim) const noexcept
{
    llama3_options o = *this;
    if (head_dim.has_value()) {
        o.set_head_dim(head_dim.value());
    }
    return o;
}


std::size_t
llama3_options::head_dim() const noexcept
{
    return _M_head_dim;
}


void
llama3_options::set_n_heads(std::size_t n_heads)
{
    _M_n_heads = n_heads;
}


llama3_options
llama3_options::n_heads(std::optional<std::size_t> n_heads) const noexcept
{
    llama3_options o = *this;
    if (n_heads.has_value()) {
        o.set_n_heads(n_heads.value());
    }
    return o;
}


std::size_t
llama3_options::n_heads() const noexcept
{
    return _M_n_heads;
}


void
llama3_options::set_n_kv_heads(std::size_t n_kv_heads)
{
    _M_n_kv_heads = n_kv_heads;
}


llama3_options
llama3_options::n_kv_heads(std::optional<std::size_t> n_kv_heads) const noexcept
{
    llama3_options o = *this;
    if (n_kv_heads.has_value()) {
        o.set_n_kv_heads(n_kv_heads.value());
    }
    return o;
}


std::size_t
llama3_options::n_kv_heads() const noexcept
{
    return _M_n_kv_heads;
}


void
llama3_options::set_n_layers(std::size_t n_layers)
{
    _M_n_layers = n_layers;
}


llama3_options
llama3_options::n_layers(std::optional<std::size_t> n_layers) const noexcept
{
    llama3_options o = *this;
    if (n_layers.has_value()) {
        o.set_n_layers(n_layers.value());
    }
    return o;
}


std::size_t
llama3_options::n_layers() const noexcept
{
    return _M_n_layers;
}


void
llama3_options::set_max_seq_len(std::size_t max_seq_len)
{
    _M_max_seq_len = max_seq_len;
}


llama3_options
llama3_options::max_seq_len(std::optional<std::size_t> max_seq_len) const noexcept
{
    llama3_options o = *this;
    if (max_seq_len.has_value()) {
        o.set_max_seq_len(max_seq_len.value());
    }
    return o;
}


std::size_t
llama3_options::max_seq_len() const noexcept
{
    return _M_max_seq_len;
}


void
llama3_options::set_heap_size(std::size_t heap_size)
{
    _M_heap_size = heap_size;
}


llama3_options
llama3_options::heap_size(std::optional<std::size_t> heap_size) const noexcept
{
    llama3_options o = *this;
    if (heap_size.has_value()) {
        o.set_heap_size(heap_size.value());
    }
    return o;
}


std::size_t
llama3_options::heap_size() const noexcept
{
    return _M_heap_size;
}


void
llama3_options::set_rope_theta(float rope_theta)
{
    _M_rope_theta = rope_theta;
}


llama3_options
llama3_options::rope_theta(std::optional<float> rope_theta) const noexcept
{
    llama3_options o = *this;
    if (rope_theta.has_value()) {
        o.set_rope_theta(rope_theta.value());
    }
    return o;
}


float
llama3_options::rope_theta() const noexcept
{
    return _M_rope_theta;
}


} // namespace nn
} // namespace metalchat
