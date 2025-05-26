#include <format>

#include <CoreFoundation/CFBundle.h>

#include <metalchat/accelerator.h>


namespace metalchat {


hardware_accelerator::hardware_accelerator(
    const std::filesystem::path& path, std::size_t thread_capacity
)
: _m_device(_m_make_device()),
  _m_library(),
  _m_kernels(),
  _m_this_thread_group(std::make_shared<kernel_thread_group>(_m_device, thread_capacity))
{
    auto path_str = path.string();
    auto path_cstr = path_str.c_str();

    auto filepath = NS::TransferPtr(NS::String::string(path_cstr, NS::UTF8StringEncoding));
    auto url = NS::TransferPtr(NS::URL::fileURLWithPath(filepath.get()));

    _m_library = _m_make_library(url.get());
}


hardware_accelerator::hardware_accelerator(std::size_t thread_capacity)
: _m_device(_m_make_device()),
  _m_library(),
  _m_kernels(),
  _m_this_thread_group(std::make_shared<kernel_thread_group>(_m_device, thread_capacity))
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

    _m_library = _m_make_library(reinterpret_cast<const NS::URL*>(library_url));

    CFRelease(library_url);
    CFRelease(library_name);
    CFRelease(resources_url);
    CFRelease(bundle_id);
}


hardware_accelerator::allocator_type
hardware_accelerator::get_allocator() const
{
    return _m_this_thread_group->get_allocator();
}


void
hardware_accelerator::set_allocator(hardware_accelerator::allocator_type alloc)
{
    _m_this_thread_group->set_allocator(alloc);
}


std::string
hardware_accelerator::name() const
{
    auto device_name = NS::TransferPtr(_m_device->name());
    return std::string(device_name->utf8String());
}


basic_kernel
hardware_accelerator::load(const std::string& name)
{
    if (auto it = _m_kernels.find(name); it != _m_kernels.end()) {
        return it->second;
    }

    auto kernel = basic_kernel(name, _m_library, _m_this_thread_group);
    _m_kernels.insert_or_assign(name, kernel);
    return kernel;
}


basic_kernel
hardware_accelerator::load(const std::string& name, const std::string& type)
{
    return load(std::format("{}_{}", name, type));
}


} // namespace metalchat
