// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <metalchat/kernel_thread.h>

#include "metal_impl.h"


namespace metalchat {


struct kernel_queue {
    std::size_t id;

    NS::SharedPtr<MTL::CommandQueue> queue;
    NS::SharedPtr<MTL::CommandBuffer> commands;
    NS::SharedPtr<MTL::ComputeCommandEncoder> encoder;
    NS::SharedPtr<MTL::Event> event;

    kernel_queue() {}

    kernel_queue(metal::shared_device device)
    : id(0),
      queue(NS::TransferPtr(device->ptr->newCommandQueue())),
      commands(NS::TransferPtr(queue->commandBuffer())),
      encoder(NS::TransferPtr(commands->computeCommandEncoder(MTL::DispatchTypeConcurrent))),
      event(NS::TransferPtr(device->ptr->newEvent()))
    {
        auto label = NS::TransferPtr(NS::String::string("metalchat", NS::UTF8StringEncoding));
        queue->setLabel(label.get());
    }

    kernel_queue
    partition()
    {
        kernel_queue kq = *this;

        kq.id++;
        kq.commands = NS::TransferPtr(kq.queue->commandBuffer());
        kq.commands->enqueue();
        kq.commands->encodeWait(kq.event.get(), id);

        auto encoder = kq.commands->computeCommandEncoder(MTL::DispatchTypeConcurrent);
        kq.encoder = NS::TransferPtr(encoder);
        return kq;
    }

    void
    on_completed(kernel_callback_type callback)
    {
        commands->addCompletedHandler([callback = callback](const MTL::CommandBuffer* buf) {
            callback();
        });
    }
};


hardware_function_encoder::hardware_function_encoder(
    std::shared_ptr<kernel_queue> queue_ptr, hardware_function_encoder::allocator_type alloc
)
: _M_allocator(alloc),
  _M_queue(queue_ptr),
  _M_buffer(0),
  _M_name()
{}


void
hardware_function_encoder::initialize(const std::string& name, const metal::shared_kernel kernel)
{
    _M_name = name;
    _M_queue->encoder->setComputePipelineState(kernel->pipeline.get());
}


void
hardware_function_encoder::encode(const void* data, std::size_t size)
{
    _M_queue->encoder->setBytes(data, size, _M_buffer++);
}


void
hardware_function_encoder::encode(metal::shared_buffer buffer, std::size_t offset)
{
    _M_queue->encoder->setBuffer(buffer->ptr, offset, _M_buffer++);
}


void
hardware_function_encoder::encode_memory_barrier(metal::shared_buffer buffer)
{
    const MTL::Resource* resources[1] = {buffer->ptr};
    _M_queue->encoder->memoryBarrier(resources, 1);
}


void
hardware_function_encoder::on_completed(kernel_callback_type callback)
{
    _M_queue->on_completed(callback);
}


void
hardware_function_encoder::dispatch(dim3 grid, dim3 group)
{
    std::stringstream command_name_stream;
    command_name_stream << _M_name << "<" << grid << "," << group << ">" << std::endl;

    auto command_name = command_name_stream.str();
    auto cmd_name = NS::TransferPtr(NS::String::string(command_name.c_str(), NS::UTF8StringEncoding));
    _M_queue->encoder->setLabel(cmd_name.get());

    MTL::Size threads_per_grid(grid.x, grid.y, grid.z);
    MTL::Size threads_per_group(group.x, group.y, group.z);
    _M_queue->encoder->dispatchThreads(threads_per_grid, threads_per_group);
}


kernel_thread::kernel_thread(const kernel_queue& queue, std::size_t capacity, allocator_type alloc)
: _M_allocator(alloc),
  _M_queue(std::make_shared<kernel_queue>(queue)),
  _M_promise(std::make_shared<promise_type>()),
  _M_future(_M_promise->get_future()),
  _M_size(0),
  _M_capacity(capacity),
  _M_committed(false)
{
    // After the completion of the kernel execution, release the promise and all blocks
    // waiting for the completion of this kernel.
    _M_queue->commands->addCompletedHandler([promise
                                             = _M_promise](const MTL::CommandBuffer* buffer) {
        if (buffer->error() != nullptr) {
            auto failure_reason = buffer->error()->localizedDescription();
            auto exception_ptr
                = std::make_exception_ptr(std::runtime_error(failure_reason->utf8String()));
            promise->set_exception(exception_ptr);
        } else {
            promise->set_value();
        }
    });
}


void
kernel_thread::on_completed(kernel_callback_type callback)
{
    _M_queue->on_completed(callback);
}


kernel_thread::~kernel_thread()
{
    //  If thread was completed, the code below does absolutely nothing, otherwise,
    //  on object deletion all commands are committed to the device.
    make_ready_at_thread_exit();
}


std::size_t
kernel_thread::size() const
{
    return _M_size;
}


std::size_t
kernel_thread::capacity() const
{
    return _M_capacity;
}


bool
kernel_thread::joinable() const
{
    return (!_M_committed) && (_M_size < _M_capacity);
}


void
kernel_thread::make_ready_at_thread_exit()
{
    if (!_M_committed) {
        auto label = std::format("metalchat commands (size={})", _M_size);
        auto cmd_label = NS::TransferPtr(NS::String::string(label.c_str(), NS::UTF8StringEncoding));

        _M_queue->encoder->endEncoding();

        _M_queue->commands->setLabel(cmd_label.get());
        _M_queue->commands->encodeSignalEvent(_M_queue->event.get(), _M_queue->id + 1);
        _M_queue->commands->commit();

        _M_committed = true;
    }
}


recursive_kernel_thread::recursive_kernel_thread(
    metal::shared_device device, std::size_t thread_capacity
)
: _M_allocator(hardware_memory_allocator(device)),
  _M_queue(std::make_shared<kernel_queue>(device)),
  _M_thread(std::make_shared<kernel_thread>(*_M_queue, thread_capacity, _M_allocator)),
  _M_thread_capacity(thread_capacity)
{}


std::shared_ptr<kernel_thread>
recursive_kernel_thread::get_this_thread()
{
    if (!_M_thread->joinable()) {
        auto queue = std::make_shared<kernel_queue>(_M_queue->partition());
        auto thread = std::make_shared<kernel_thread>(*queue, _M_thread_capacity, _M_allocator);

        _M_queue.swap(queue);
        _M_thread.swap(thread);
    }
    return _M_thread;
}


} // namespace metalchat
