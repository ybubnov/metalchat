#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>


int
main()
{
    auto filepath
        = NS::TransferPtr(NS::String::string("metalchat.metallib", NS::UTF8StringEncoding));

    auto url = NS::TransferPtr(NS::URL::fileURLWithPath(filepath.get()));
    std::cout << "kernel=" << url->fileSystemRepresentation() << std::endl;

    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    auto device_name = NS::TransferPtr(device->name());
    std::cout << "name=" << device_name->utf8String() << std::endl;

    NS::Error* error = nullptr;
    auto library = NS::TransferPtr(device->newLibrary(url.get(), &error));
    if (!library) {
        std::cout << "failed to create shader library" << std::endl;
        return 1;
    }

    NS::Array* function_names = library->functionNames();
    for (NS::UInteger i = 0; i < function_names->count(); i++) {
        NS::String* function_name = (NS::String*)function_names->object(i);
        std::cout << "registered (" << function_name->utf8String() << ")" << std::endl;
    }

    auto mul_name = NS::TransferPtr(NS::String::string("mul", NS::UTF8StringEncoding));
    auto mul_kernel = NS::TransferPtr(library->newFunction(mul_name.get()));

    auto pipeline = NS::TransferPtr(device->newComputePipelineState(mul_kernel.get(), &error));
    if (error != nullptr) {
        std::cout << "failed to create compute pipeline" << std::endl;
        return 1;
    }
    std::cout << "created pipeline" << std::endl;

    auto command_queue = NS::TransferPtr(device->newCommandQueue());
    std::cout << "created queue" << std::endl;

    std::size_t length = 1000;
    NS::UInteger length_bytes = sizeof(float) * length;

    float* input = new float[length];
    float* other = new float[length];
    float* output = new float[length];
    for (std::size_t i = 0; i < length; i++) {
        input[i] = i;
        other[i] = 10 * i;
        output[i] = 0;
    }

    auto input_buf
        = NS::TransferPtr(device->newBuffer(input, length_bytes, MTL::ResourceStorageModeShared));
    auto other_buf
        = NS::TransferPtr(device->newBuffer(other, length_bytes, MTL::ResourceStorageModeShared));
    auto output_buf
        = NS::TransferPtr(device->newBuffer(output, length_bytes, MTL::ResourceStorageModeShared));
    std::cout << "created buffers" << std::endl;

    auto command_buf = NS::TransferPtr(command_queue->commandBuffer());
    auto command_encoder = NS::TransferPtr(command_buf->computeCommandEncoder());

    std::cout << "computing pipeline" << std::endl;
    command_encoder->setComputePipelineState(pipeline.get());
    command_encoder->setBuffer(input_buf.get(), 0, 0);
    command_encoder->setBuffer(other_buf.get(), 0, 1);
    command_encoder->setBuffer(output_buf.get(), 0, 2);

    MTL::Size grid_size(length, 1, 1);
    MTL::Size thread_group_size(std::min(length, pipeline->maxTotalThreadsPerThreadgroup()), 1, 1);
    command_encoder->dispatchThreadgroups(grid_size, thread_group_size);

    command_encoder->endEncoding();
    command_buf->commit();
    command_buf->waitUntilCompleted();

    // float* output_content = (float*)output_buf->contents();
    float* output_content = output;
    for (std::size_t i = 0; i < length; i++) {
        std::cout << output_content[i];
        if (i < length - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;

    free(input);
    free(other);
    free(output);
    return 0;
}
