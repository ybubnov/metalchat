// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metalchat/kernel.h>

#include "metal_impl.h"


namespace metalchat {


std::tuple<dim3, dim3>
make_kernel_grid_2d(std::size_t num_rows, std::size_t dim_size, std::size_t max_threads)
{
    auto data_size = dim_size * num_rows;

    if (data_size <= max_threads) {
        auto thread = dim3(dim_size, num_rows);
        auto grid = dim3(dim_size, num_rows);
        return std::make_tuple(grid, thread);
    }

    if (dim_size <= max_threads) {
        auto thread = dim3(dim_size);
        auto grid = dim3(dim_size, num_rows);
        return std::make_tuple(grid, thread);
    }

    auto thread_size = max_threads;
    auto thread_groups = ceil_div(dim_size, thread_size);

    auto thread = dim3(thread_size);
    auto grid = dim3(thread_size * thread_groups, num_rows);

    return std::make_tuple(grid, thread);
}


basic_kernel::basic_kernel(metal::shared_kernel kernel, const hardware_accelerator& accelerator)
: _M_name(kernel->function->name()->utf8String(), kernel->function->name()->length()),
  _M_kernel(kernel),
  _M_accelerator(accelerator)
{}


std::string
basic_kernel::name() const
{
    return _M_name;
}


hardware_accelerator&
basic_kernel::get_accelerator()
{
    return _M_accelerator;
}


hardware_accelerator::allocator_type
basic_kernel::get_allocator() const
{
    return _M_accelerator.get_allocator();
}


const metal::shared_kernel
basic_kernel::get_metal_kernel() const
{
    return _M_kernel;
}


std::size_t
basic_kernel::max_threads_per_threadgroup()
{
    return _M_kernel->pipeline->maxTotalThreadsPerThreadgroup();
}


} // namespace metalchat
