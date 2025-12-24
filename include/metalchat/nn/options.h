// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once


namespace metalchat {
namespace nn {


struct llama3_options {
public:
    llama3_options();
    llama3_options(const llama3_options&) = default;

    llama3_options
    head_dim(std::size_t value) const noexcept;

    llama3_options
    n_heads(std::size_t value) const noexcept;

    llama3_options
    n_kv_heads(std::size_t value) const noexcept;

    llama3_options
    n_layers(std::size_t value) const noexcept;

    llama3_options
    max_seq_len(std::size_t value) const noexcept;

    llama3_options
    rope_theta(float value) const noexcept;

    llama3_options
    norm_eps(float value) const noexcept;

    std::size_t
    head_dim() const noexcept;

    std::size_t
    n_heads() const noexcept;

    std::size_t
    n_kv_heads() const noexcept;

    std::size_t
    n_layers() const noexcept;

    std::size_t
    max_seq_len() const noexcept;

    float
    rope_theta() const noexcept;

    float
    norm_eps() const noexcept;

private:
    std::size_t _M_head_dim = 0;
    std::size_t _M_n_heads = 0;
    std::size_t _M_n_kv_heads = 0;
    std::size_t _M_n_layers = 0;
    std::size_t _M_max_seq_len = 0;
    float _M_rope_theta = 0.0f;
    float _M_norm_eps = 0.0f;
};


llama3_options
default_llama3_1b_options();


} // namespace nn
} // namespace metalchat
