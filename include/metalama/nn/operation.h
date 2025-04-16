#pragma once


#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <metalama/tensor.h>


namespace metalama {
namespace nn {


class operation {
private:
    std::string m_op;

protected:
    device& m_device;

public:
    operation(const std::string& op, device& device)
    : m_op(op),
      m_device(device)
    {}

    std::string
    name() const
    {
        return m_op;
    }

    template <typename T, std::size_t N, template <typename U> class Reference>
    NS::SharedPtr<MTL::Buffer>
    make_buf(const tensor<T, N, Reference>& t)
    {
        auto size = t.numel() * sizeof(T);
        std::cout << "buffer(ptr)=" << t.data_ptr() << "; [0]=" << t.data_ptr()[0] << std::endl;
        return NS::TransferPtr(
            m_device->newBuffer(t.data_ptr(), size, MTL::ResourceStorageModeShared)
        );
    }

    template <typename T, std::size_t N>
    NS::SharedPtr<MTL::Buffer>
    make_buf(const tensor<T, N, device_ref>& t)
    {
        return t.storage()->m_buf;
    }
};


} // namespace nn
} // namespace metalama
