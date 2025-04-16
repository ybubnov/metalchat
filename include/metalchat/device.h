#pragma once


#include <filesystem>
#include <iostream>
#include <unordered_map>

#include <metalchat/kernel.h>
#include <metalchat/kernel_thread.h>


namespace metalchat {


class device {
private:
    NS::SharedPtr<MTL::Device> _m_device;
    NS::SharedPtr<MTL::Library> _m_library;

    std::unordered_map<std::string, kernel_base> _m_kernels;
    shared_kernel_thread _m_this_thread;

    NS::SharedPtr<MTL::Device>
    _m_make_device()
    {
        return NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    }

    shared_kernel_thread
    _m_make_kernel_thread(NS::SharedPtr<MTL::Device>, std::size_t thread_capacity)
    {
        auto queue = NS::TransferPtr(_m_device->newCommandQueue());
        return shared_kernel_thread(queue, thread_capacity);
    }

public:
    device(device&&) noexcept = default;
    device(const device&) = delete;

    device(const std::filesystem::path& path, std::size_t thread_capacity = 64)
    : _m_device(_m_make_device()),
      _m_library(),
      _m_kernels(),
      _m_this_thread(_m_make_kernel_thread(_m_device, thread_capacity))
    {
        auto path_str = path.string();
        auto path_cstr = path_str.c_str();

        auto filepath = NS::TransferPtr(NS::String::string(path_cstr, NS::UTF8StringEncoding));
        auto url = NS::TransferPtr(NS::URL::fileURLWithPath(filepath.get()));

        NS::SharedPtr<NS::Error> error = NS::TransferPtr(NS::Error::alloc());
        NS::Error* error_ptr = error.get();

        _m_library = NS::TransferPtr(_m_device->newLibrary(url.get(), &error_ptr));
        if (!_m_library) {
            auto failure_reason = error_ptr->localizedDescription();
            throw std::runtime_error(failure_reason->utf8String());
        }
    }

    std::string
    name() const
    {
        auto device_name = NS::TransferPtr(_m_device->name());
        return std::string(device_name->utf8String());
    }

    inline MTL::Device*
    operator->()
    {
        return _m_device.get();
    }

    inline MTL::Device*
    operator*()
    {
        return _m_device.get();
    }

    kernel_base
    load(const std::string& name)
    {
        if (auto it = _m_kernels.find(name); it != _m_kernels.end()) {
            return it->second;
        }

        auto kernel = kernel_base(name, _m_library, _m_this_thread);
        _m_kernels.insert_or_assign(name, kernel);
        return kernel;
    }

    kernel_base
    load(const std::string& name, const std::string& type)
    {
        return load(std::format("{}_{}", name, type));
    }
};


} // namespace metalchat
