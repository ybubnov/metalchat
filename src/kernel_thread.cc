#include <metalchat/kernel_thread.h>

#include "metal_impl.h"


namespace metalchat {


void
hardware_function_encoder::initialize(const std::string& name, const metal::shared_kernel kernel)
{
    _m_name = name;
    _m_encoder->setComputePipelineState(kernel->pipeline.get());
}


void
hardware_function_encoder::encode(const void* data, std::size_t size)
{
    _m_encoder->setBytes(data, size, _m_buffer++);
}


void
hardware_function_encoder::encode(metal::shared_buffer buffer)
{
    _m_encoder->setBuffer(buffer->ptr.get(), 0, _m_buffer++);
}


void
hardware_function_encoder::encode_memory_barrier(metal::shared_buffer buffer)
{
    const MTL::Resource* resources[1] = {buffer->ptr.get()};
    _m_encoder->memoryBarrier(resources, 1);
}


void
hardware_function_encoder::dispatch(dim3 grid, dim3 group)
{
    std::stringstream command_name_stream;
    command_name_stream << _m_name << "<" << grid << "," << group << ">" << std::endl;

    auto command_name = command_name_stream.str();
    auto command_name_ptr
        = NS::TransferPtr(NS::String::string(command_name.c_str(), NS::UTF8StringEncoding));
    _m_encoder->setLabel(command_name_ptr.get());

    MTL::Size threads_per_grid(grid.x, grid.y, grid.z);
    MTL::Size threads_per_group(group.x, group.y, group.z);
    _m_encoder->dispatchThreads(threads_per_grid, threads_per_group);
}


} // namespace metalchat
