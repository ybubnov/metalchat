#pragma once

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


#include <filesystem>


namespace metalama {


class device {
private:
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL::Library> m_library;

public:
    device(const std::filesystem::path& path)
    {
        auto path_str = path.string();
        auto path_cstr = path_str.c_str();

        auto filepath = NS::TransferPtr(NS::String::string(path_cstr, NS::UTF8StringEncoding));
        auto url = NS::TransferPtr(NS::URL::fileURLWithPath(filepath.get()));

        m_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());

        NS::Error* error = nullptr;
        m_library = NS::TransferPtr(m_device->newLibrary(url.get(), &error));

        if (!m_library) {
            throw std::runtime_error(error->localizedFailureReason()->utf8String());
        }
    }

    std::string
    name() const
    {
        auto device_name = NS::TransferPtr(m_device->name());
        return std::string(device_name->utf8String());
    }

    inline MTL::Device*
    operator->()
    {
        return m_device.get();
    }

    NS::SharedPtr<MTL::Function>
    make_fn(const std::string& fname)
    {
        auto name = NS::TransferPtr(NS::String::string(fname.c_str(), NS::UTF8StringEncoding));
        auto op_kernel = NS::TransferPtr(m_library->newFunction(name.get()));
        return op_kernel;
    }
};


} // namespace metalama
