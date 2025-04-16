#pragma once


#include <filesystem>
#include <iostream>
#include <unordered_map>

#include <metalchat/kernel.h>


namespace metalchat {


class device {
private:
    NS::SharedPtr<MTL::Device> _m_device;
    NS::SharedPtr<MTL::Library> _m_library;

    std::unordered_map<std::string, kernel_base> _m_kernels;

public:
    device(const std::filesystem::path& path)
    : _m_device(),
      _m_library(),
      _m_kernels()
    {
        auto path_str = path.string();
        auto path_cstr = path_str.c_str();

        auto filepath = NS::TransferPtr(NS::String::string(path_cstr, NS::UTF8StringEncoding));
        auto url = NS::TransferPtr(NS::URL::fileURLWithPath(filepath.get()));

        _m_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());

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

        auto kernel = kernel_base(name, _m_device, _m_library);
        _m_kernels.insert_or_assign(name, kernel);
        return kernel;
    }

    kernel_base
    load(const std::string& name, const std::string& type)
    {
        return load(std::format("{}_{}", name, type));
    }

    NS::SharedPtr<MTL::Function>
    make_fn(const std::string& fname)
    {
        auto name = NS::TransferPtr(NS::String::string(fname.c_str(), NS::UTF8StringEncoding));
        auto op_kernel = NS::TransferPtr(_m_library->newFunction(name.get()));
        return op_kernel;
    }
};


} // namespace metalchat
