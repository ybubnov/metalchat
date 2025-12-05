// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <metalchat/container.h>
#include <metalchat/tensor/basic.h>
#include <metalchat/tensor/future.h>
#include <metalchat/tensor/shared.h>


namespace metalchat {


template <typename T, std::size_t N, contiguous_container Container> struct tensor_traits {
    using value_type = T;
    using void_container = container_remove_type<Container>::type;
    using container_type = container_rebind<value_type, void_container>::type;
    using type = tensor<value_type, N, container_type>;
    using pointer = shared_tensor_ptr<type>;
    using future = future_tensor<value_type, N>;
};


} // namespace metalchat
