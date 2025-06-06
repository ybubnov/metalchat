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
        kernel_queue kq;

        kq.id = id + 1;
        kq.queue = queue;
        kq.event = event;

        kq.commands = NS::TransferPtr(kq.queue->commandBuffer());
        kq.commands->enqueue();
        kq.commands->encodeWait(kq.event.get(), id);

        kq.encoder
            = NS::TransferPtr(kq.commands->computeCommandEncoder(MTL::DispatchTypeConcurrent));
        return kq;
    }
};


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


kernel_thread::kernel_thread(const kernel_queue& queue, std::size_t capacity, allocator_type alloc)
: _m_allocator(alloc),
  _m_queue(std::make_shared<kernel_queue>(queue)),
  _m_promise(std::make_shared<promise_type>()),
  _m_future(_m_promise->get_future()),
  _m_size(0),
  _m_capacity(capacity),
  _m_committed(false)
{
    // After the completion of the kernel execution, release the promise and all blocks
    // waiting for the completion of this kernel.
    _m_queue->commands->addCompletedHandler([promise
                                             = _m_promise](const MTL::CommandBuffer* buffer) {
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
kernel_thread::on_completed(kernel_thread::callback_type callback)
{
    _m_queue->commands->addCompletedHandler([callback = callback](const MTL::CommandBuffer* buf) {
        callback();
    });
}


hardware_function_encoder
kernel_thread::get_hardware_function_encoder()
{
    return hardware_function_encoder(_m_queue->encoder, _m_allocator);
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
    return _m_size;
}


std::size_t
kernel_thread::capacity() const
{
    return _m_capacity;
}


bool
kernel_thread::joinable() const
{
    return (!_m_committed) && (_m_size < _m_capacity);
}


void
kernel_thread::make_ready_at_thread_exit()
{
    if (!_m_committed) {
        auto label = std::format("metalchat commands (size={})", _m_size);
        auto label_ptr = NS::String::string(label.c_str(), NS::UTF8StringEncoding);
        auto commands_label = NS::TransferPtr(label_ptr);

        _m_queue->encoder->endEncoding();

        _m_queue->commands->setLabel(commands_label.get());
        _m_queue->commands->encodeSignalEvent(_m_queue->event.get(), _m_queue->id + 1);
        _m_queue->commands->commit();

        _m_committed = true;
    }
}


recursive_kernel_thread::recursive_kernel_thread(
    metal::shared_device device, std::size_t thread_capacity
)
: _m_allocator(std::make_shared<hardware_memory_allocator<void>>(device)),
  _m_queue(std::make_shared<kernel_queue>(device)),
  _m_thread(std::make_shared<kernel_thread>(*_m_queue, thread_capacity, _m_allocator)),
  _m_thread_capacity(thread_capacity)
{}


std::shared_ptr<kernel_thread>
recursive_kernel_thread::get_this_thread()
{
    if (!_m_thread->joinable()) {
        auto queue = std::make_shared<kernel_queue>(_m_queue->partition());
        auto thread = std::make_shared<kernel_thread>(*queue, _m_thread_capacity, _m_allocator);

        _m_queue.swap(queue);
        _m_thread.swap(thread);
    }
    return _m_thread;
}


} // namespace metalchat
