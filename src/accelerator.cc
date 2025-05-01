#include <metalchat/accelerator.h>


namespace metalchat {


hardware_accelerator::hardware_accelerator(
    const std::filesystem::path& path, std::size_t thread_capacity
)
: _m_device(_m_make_device()),
  _m_library(),
  _m_kernels(),
  _m_this_thread(_m_make_kernel_thread(thread_capacity))
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


hardware_accelerator::allocator_type
hardware_accelerator::get_allocator() const
{
    return _m_this_thread.get_allocator();
}


void
hardware_accelerator::set_allocator(hardware_accelerator::allocator_type alloc)
{
    _m_this_thread.set_allocator(alloc);
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

    auto kernel = basic_kernel(name, _m_library, _m_this_thread);
    _m_kernels.insert_or_assign(name, kernel);
    return kernel;
}


basic_kernel
hardware_accelerator::load(const std::string& name, const std::string& type)
{
    return load(std::format("{}_{}", name, type));
}


} // namespace metalchat
