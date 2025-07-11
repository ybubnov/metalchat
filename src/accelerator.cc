#include <format>

#include <CoreFoundation/CFBundle.h>
#include <rapidhash.h>

#include <metalchat/accelerator.h>
#include <metalchat/kernel.h>

#include "metal_impl.h"


namespace metalchat {


std::size_t
_StringHash::operator()(const std::string& s) const noexcept
{
    return rapidhash(s.c_str(), s.size());
}


hardware_accelerator::hardware_accelerator(
    const std::filesystem::path& path, std::size_t thread_capacity
)
: _m_device(metal::make_device()),
  _m_library(metal::make_library(path, _m_device)),
  _m_kernels(),
  _m_thread(std::make_shared<recursive_kernel_thread>(_m_device, thread_capacity))
{}


hardware_accelerator::hardware_accelerator(std::size_t thread_capacity)
: _m_device(metal::make_device()),
  _m_library(),
  _m_kernels(),
  _m_thread(std::make_shared<recursive_kernel_thread>(_m_device, thread_capacity))
{
    auto bundle_id = CFStringCreateWithCString(
        kCFAllocatorDefault, framework_identifier.c_str(), kCFStringEncodingUTF8
    );

    auto bundle = CFBundleGetBundleWithIdentifier(bundle_id);
    if (!bundle) {
        throw std::invalid_argument(std::format(
            "accelerator: the binary is not linked with framework '{}'", framework_identifier
        ));
    }
    auto resources_url = CFBundleCopyResourcesDirectoryURL(bundle);
    if (!resources_url) {
        throw std::runtime_error(
            "accelerator: cannot extract resource URL from the framework bundle"
        );
    }

    auto library_name = CFSTR("metalchat.metallib");
    auto library_url = CFURLCreateCopyAppendingPathComponent(
        kCFAllocatorDefault, resources_url, library_name, /*isDirectory=*/false
    );

    _m_library = metal::make_library(reinterpret_cast<const NS::URL*>(library_url), _m_device);

    CFRelease(library_url);
    CFRelease(library_name);
    CFRelease(resources_url);
    CFRelease(bundle_id);
}


std::size_t
hardware_accelerator::max_buffer_size() const
{
    return _m_device->ptr->maxBufferLength();
}


std::shared_ptr<kernel_thread>
hardware_accelerator::get_this_thread()
{
    return _m_thread->get_this_thread();
}


metal::shared_device
hardware_accelerator::get_metal_device()
{
    return _m_device;
}


hardware_accelerator::allocator_type
hardware_accelerator::get_allocator() const
{
    return _m_thread->get_allocator();
}


void
hardware_accelerator::set_allocator(hardware_accelerator::allocator_type alloc)
{
    _m_thread->set_allocator(alloc);
}


std::string
hardware_accelerator::name() const
{
    auto device_name = _m_device->ptr->name();
    return std::string(device_name->utf8String(), device_name->length());
}


const basic_kernel&
hardware_accelerator::load(const std::string& name)
{
    if (auto it = _m_kernels.find(name); it != _m_kernels.end()) {
        return it->second;
    }

    auto fn_name = NS::RetainPtr(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
    auto fn_ptr = NS::RetainPtr(_m_library->ptr->newFunction(fn_name.get()));
    if (!fn_ptr) {
        throw std::invalid_argument(
            std::format("hardware_accelerator: function {} not found in a shader library", name)
        );
    }

    fn_ptr->setLabel(fn_name.get());

    NS::SharedPtr<NS::Error> error = NS::RetainPtr(NS::Error::alloc());
    NS::Error* error_ptr = error.get();

    auto descriptor = NS::RetainPtr(MTL::ComputePipelineDescriptor::alloc());
    descriptor->init();
    descriptor->setComputeFunction(fn_ptr.get());
    descriptor->setLabel(fn_name.get());

    auto pipeline_ptr = NS::RetainPtr(_m_device->ptr->newComputePipelineState(
        descriptor.get(), MTL::PipelineOptionNone, nullptr, &error_ptr
    ));

    if (!pipeline_ptr) {
        throw std::runtime_error(std::format(
            "hardware_accelerator: failed to create compute pipeline, {}",
            error_ptr->localizedDescription()->utf8String()
        ));
    }

    auto kernel_ptr = std::make_shared<metal::kernel>(fn_ptr, pipeline_ptr);
    auto kernel = basic_kernel(kernel_ptr, *this);

    _m_kernels.insert_or_assign(name, kernel);

    return _m_kernels.at(name);
}


const basic_kernel&
hardware_accelerator::load(const std::string& name, const std::string& type)
{
    return load(std::format("{}_{}", name, type));
}


} // namespace metalchat
