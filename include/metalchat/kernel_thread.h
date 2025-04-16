#pragma once

#include <concepts>
#include <cstdlib>
#include <format>
#include <future>
#include <memory>


namespace metalchat {


template <typename T>
concept hardware_encodable_function
    = requires(std::remove_reference_t<T> t, MTL::ComputeCommandEncoder* encoder) {
          { t.encode(encoder) } -> std::same_as<void>;
      };


class kernel_thread {
private:
    using promise_type = std::promise<void>;

    NS::SharedPtr<MTL::CommandBuffer> _m_commands;

    std::shared_ptr<promise_type> _m_promise;
    std::shared_future<void> _m_future;

    std::size_t _m_size;
    std::size_t _m_capacity;

    bool _m_committed;
    int _m_id = std::rand();

public:
    using callback_type = std::function<void()>;

    kernel_thread(const kernel_thread&) noexcept = default;

    kernel_thread(NS::SharedPtr<MTL::CommandQueue> queue, std::size_t capacity)
    : _m_commands(NS::TransferPtr(queue->commandBuffer())),
      _m_promise(std::make_shared<promise_type>()),
      _m_future(_m_promise->get_future()),
      _m_size(0),
      _m_capacity(capacity),
      _m_committed(false)
    {
        // After the completion of the kernel execution, release the promise and all blocks
        // waiting for the completion of this kernel.
        _m_commands->addCompletedHandler([promise = _m_promise](const MTL::CommandBuffer*) {
            // TODO: handle the error.
            promise->set_value();
        });
    }

    ~kernel_thread()
    {
        // std::cout << "kernel_thread::~kernel_thread()" << std::endl;
        //  If thread was completed, the code below does absolutely nothing, otherwise,
        //  on object deletion all commands are committed to the device.
        make_ready_at_thread_exit();
    }

    std::size_t
    size() const
    {
        return _m_size;
    }

    std::size_t
    capacity() const
    {
        return _m_capacity;
    }

    bool
    joinable() const
    {
        return (!_m_committed) && (_m_size < _m_capacity);
    }

    template <hardware_encodable_function F>
    std::shared_future<void>
    push(F& f, std::optional<callback_type> callback = std::nullopt)
    {
        // std::cout << "kernel_thread::push, size=" << _m_size << ", cap=" << _m_capacity <<
        // std::endl;

        if (!joinable()) {
            throw std::runtime_error(
                std::format("thread: thread is either committed or reached its capacity")
            );
        }
        if (callback.has_value()) {
            _m_commands->addCompletedHandler([callback
                                              = callback.value()](const MTL::CommandBuffer*) {
                // std::cout << "kernel_thread::callback" << std::endl;
                callback();
            });
        }

        auto encoder = _m_commands->computeCommandEncoder(MTL::DispatchTypeSerial);
        f.encode(encoder);
        encoder->endEncoding();

        _m_size++;

        if (_m_size == _m_capacity) {
            make_ready_at_thread_exit();
        }
        return _m_future;
    }

    void
    make_ready_at_thread_exit()
    {
        if (!_m_committed) {
            // std::cout << "kernel_thread::commit" << std::endl;
            _m_commands->commit();
            _m_committed = true;
        }
    }
};


class shared_kernel_thread {
private:
    NS::SharedPtr<MTL::CommandQueue> _m_queue;
    std::shared_ptr<kernel_thread> _m_this_thread;
    std::size_t _m_thread_capacity;

public:
    shared_kernel_thread(const shared_kernel_thread&) noexcept = default;

    shared_kernel_thread(NS::SharedPtr<MTL::CommandQueue> queue, std::size_t thread_capacity)
    : _m_queue(queue),
      _m_this_thread(std::make_shared<kernel_thread>(queue, thread_capacity)),
      _m_thread_capacity(thread_capacity)
    {}

    std::shared_ptr<kernel_thread>
    get_this_thread()
    {
        if (!_m_this_thread->joinable()) {
            auto thread = std::make_shared<kernel_thread>(_m_queue, _m_thread_capacity);
            _m_this_thread.swap(thread);
        }
        return _m_this_thread;
    }
};


} // namespace metalchat
