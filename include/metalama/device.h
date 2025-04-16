#pragma once

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


#include <filesystem>

#include <metalama/tensor.h>


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
};


class op {
private:
    friend class device;

    std::string m_op;

public:
    op(const std::string& op)
    : m_op(op)
    {}

    // void
    // operator()(tensor_base<T, 1>... tensors, const device& device)
    //{
    // }
};


} // namespace metalama
