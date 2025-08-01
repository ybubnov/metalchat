#include <metalchat/kernel.h>

#include "metal_impl.h"


namespace metalchat {


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
