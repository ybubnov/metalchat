#pragma once
#include <iostream>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


#include <filesystem>


namespace metalchat {


struct dim3 {
    const std::size_t x, y, z;

    constexpr dim3(std::size_t x_, std::size_t y_ = 1, std::size_t z_ = 1)
    : x(x_),
      y(y_),
      z(z_)
    {}

    friend std::ostream&
    operator<<(std::ostream& os, const dim3& d)
    {
        os << "<" << d.x << "," << d.y << "," << d.z << ">";
        return os;
    }

    std::size_t
    numel() const
    {
        return x * y * z;
    }
};


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

        NS::SharedPtr<NS::Error> error = NS::TransferPtr(NS::Error::alloc());
        NS::Error* error_ptr = error.get();

        m_library = NS::TransferPtr(m_device->newLibrary(url.get(), &error_ptr));

        if (!m_library) {
            auto failure_reason = error_ptr->localizedDescription();
            throw std::runtime_error(failure_reason->utf8String());
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


} // namespace metalchat
