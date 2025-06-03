#include <metalchat/kernel.h>

#include "metal_impl.h"


namespace metalchat {


basic_kernel::basic_kernel(metal::shared_kernel kernel, const hardware_accelerator& accelerator)
: _m_name(kernel->function->name()->utf8String()),
  _m_kernel(kernel),
  _m_accelerator(accelerator)
{}


std::string
basic_kernel::name() const
{
    return _m_name;
}


hardware_accelerator&
basic_kernel::get_accelerator()
{
    return _m_accelerator;
}


hardware_accelerator::allocator_type
basic_kernel::get_allocator() const
{
    return _m_accelerator.get_allocator();
}


const metal::shared_kernel
basic_kernel::get_metal_kernel() const
{
    return _m_kernel;
}


std::size_t
basic_kernel::max_threads_per_threadgroup()
{
    return _m_kernel->pipeline->maxTotalThreadsPerThreadgroup();
}


} // namespace metalchat
