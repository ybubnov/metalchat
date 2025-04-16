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

    template <typename T, std::size_t N>
    NS::SharedPtr<MTL::Buffer>
    make_buf(const tensor<T, N>& t)
    {
        auto size = t.numel() * sizeof(T);
        std::cout << "buffer(ptr)=" << t.data_ptr() << "; [0]=" << t.data_ptr()[0] << std::endl;
        return NS::TransferPtr(
            m_device.get()->newBuffer(t.data_ptr(), size, MTL::ResourceStorageModeShared)
        );
    }
};


class op {
private:
    friend class device;

    std::string m_op;

protected:
    device& m_device;

public:
    op(const std::string& op, device& device)
    : m_op(op),
      m_device(device)
    {}

    std::string
    name() const
    {
        return m_op;
    }
};


template <typename T>
struct device_ref : public array_ref<T> {
    using ptr_type = NS::SharedPtr<MTL::Buffer>;

    ptr_type m_buf;

    device_ref(NS::SharedPtr<MTL::Buffer> buf)
    : m_buf(buf)
    {}

    T*
    data() override
    {
        return static_cast<T*>(m_buf->contents());
    }

    ~device_ref() { std::cout << "device_ref::~device_ref()" << std::endl; }
};


class embedding : public op {

public:
    embedding(const std::string& opname, device& device)
    : op(opname, device)
    {}

    template <typename T>
    tensor<T, 2>
    operator()(const tensor<int32_t, 1>& input, const tensor<T, 2>& weight)
    {
        auto op_kernel = this->m_device.make_fn(this->name());

        NS::Error* error = nullptr;
        auto pipeline
            = NS::TransferPtr(this->m_device->newComputePipelineState(op_kernel.get(), &error));
        if (error != nullptr) {
            throw std::runtime_error("failed to create compute pipeline");
        }

        auto command_queue = NS::TransferPtr(this->m_device->newCommandQueue());

        auto input_buf = this->m_device.make_buf(input);
        auto weight_buf = this->m_device.make_buf(weight);
        auto weight_stride = weight.stride(0);

        auto result_buf = NS::TransferPtr(
            this->m_device->newBuffer(input.numel() * sizeof(T), MTL::ResourceStorageModeShared)
        );

        auto command_buf = NS::TransferPtr(command_queue->commandBuffer());
        auto command_encoder = NS::TransferPtr(command_buf->computeCommandEncoder());

        command_encoder->setComputePipelineState(pipeline.get());
        command_encoder->setBuffer(input_buf.get(), 0, 0);
        command_encoder->setBuffer(weight_buf.get(), 0, 1);
        command_encoder->setBytes(&weight_stride, sizeof(weight_stride), 0, 2);
        command_encoder->setBuffer(result_buf.get(), 0, 3);

        MTL::Size grid_size(input.size(0), weight.size(1), 1);
        MTL::Size thread_group_size(1, 1, 1);
        command_encoder->dispatchThreadgroups(grid_size, thread_group_size);

        command_encoder->endEncoding();
        command_buf->commit();
        command_buf->waitUntilCompleted();

        auto shape = new std::size_t[2]{input.size(0), weight.size(1)};
        auto strides = new std::size_t[2]{shape[1], 1};

        using tensor_type = tensor<T, 2>;

        return tensor_type(
            std::move(std::make_unique<device_ref<T>>(result_buf)),
            tensor_type::traits::borrow(shape), tensor_type::traits::borrow(strides)
        );
    }
};


} // namespace metalama
