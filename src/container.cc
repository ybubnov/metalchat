#include <format>

#include <sys/mman.h>

#include <metalchat/container.h>


namespace metalchat {


basic_memfile::basic_memfile(const std::filesystem::path& p)
{
    _m_file = std::fopen(p.c_str(), "r");
    if (_m_file == nullptr) {
        throw std::invalid_argument(
            std::format("basic_memfile: unable to open file '{}'", p.string())
        );
    }

    _m_file_size = static_cast<std::size_t>(std::filesystem::file_size(p));
    _m_file_p = _m_file_size;
}


basic_memfile::basic_memfile()
{
    _m_file = std::tmpfile();
    if (_m_file == nullptr) {
        throw std::invalid_argument("basic_memfile: unable to create temporary file");
    }
}


bool
basic_memfile::is_mapped() const noexcept
{
    return _m_map != MAP_FAILED && _m_map != nullptr;
}


basic_memfile&
basic_memfile::declare_mapped()
{
    if (is_mapped()) {
        return *this;
    }

    int fd = fileno(_m_file);
    if (fd == -1) {
        throw std::invalid_argument("basic_memfile: unable to get file descriptor for a file");
    }

    _m_map = static_cast<char_type*>(mmap(nullptr, _m_file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (_m_map == MAP_FAILED) {
        throw std::invalid_argument("basic_memfile: unable to memory-map safetensors a file");
    }

    return *this;
}


basic_memfile&
basic_memfile::undeclare_mapped()
{
    if (is_mapped()) {
        munmap(_m_map, _m_file_size);
        _m_map = nullptr;
    }

    return *this;
}


std::size_t
basic_memfile::size() const noexcept
{
    return _m_file_size;
}


const basic_memfile::char_type*
basic_memfile::data() const noexcept
{
    return _m_map;
}


basic_memfile::char_type*
basic_memfile::data() noexcept
{
    return _m_map;
}


basic_memfile::pos_type
basic_memfile::tellp() const noexcept
{
    return _m_file_p;
}


basic_memfile::pos_type
basic_memfile::tellg() const noexcept
{
    return _m_file_g;
}


basic_memfile&
basic_memfile::read(char_type* d, std::size_t size)
{
    if (!is_mapped()) {
        auto read = std::fread(d, sizeof(char_type), size, _m_file);
        _m_file_g += read;
        _m_file_size += read;
        return *this;
    }

    if (_m_file_size < _m_file_g + size) {
        throw std::out_of_range(
            std::format("file is too short ({}) to read ({}) more bytes", _m_file_size, size)
        );
    }

    std::memcpy(d, _m_map + _m_file_g, size);
    _m_file_g += size;

    return *this;
}


basic_memfile&
basic_memfile::read(void* d, std::size_t size)
{
    return read(static_cast<char_type*>(d), size);
}


basic_memfile&
basic_memfile::write(const char_type* s, std::size_t size)
{
    if (is_mapped()) {
        throw std::runtime_error("basic_memfile: writing to a memory-mapped file is prohibited");
    }

    auto written = std::fwrite(s, sizeof(char_type), size, _m_file);
    _m_file_p += written;
    _m_file_size += written;

    if (written != size) {
        throw std::runtime_error(std::format(
            "basic_memfile: failed to write {} elements, only {} succeeded", size, written
        ));
    }

    return *this;
}


basic_memfile&
basic_memfile::write(const void* s, std::size_t size)
{
    return write(static_cast<const char_type*>(s), size);
}


basic_memfile::~basic_memfile()
{
    undeclare_mapped();

    if (_m_file != nullptr) {
        std::fclose(_m_file);
        _m_file = nullptr;
    }
}


} // namespace metalchat
