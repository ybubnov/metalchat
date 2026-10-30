// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <format>

#include <sys/mman.h>

#include <metalchat/container.h>


namespace metalchat {


std::string
filemode(std::ios::openmode mode)
{
    std::string fmode("r");
    auto writeonly = std::ios::out;
    if ((mode & writeonly) == writeonly) {
        fmode = "w";
    }

    auto readwrite = std::ios::in | std::ios::out;
    if ((mode & readwrite) == readwrite) {
        fmode = "w+";
    }
    return fmode;
}


basic_memfile::basic_memfile(const std::filesystem::path& p, std::ios::openmode mode)
{
    auto fmode = filemode(mode);
    _M_file = std::fopen(p.c_str(), fmode.c_str());
    if (_M_file == nullptr) {
        throw std::invalid_argument(
            std::format("basic_memfile: unable to open file '{}'", p.string())
        );
    }

    _M_file_size = static_cast<std::size_t>(std::filesystem::file_size(p));
    _M_file_p = _M_file_size;
    _M_mode = mode;
}


basic_memfile::basic_memfile(const std::filesystem::path& p)
: basic_memfile(p, std::ios::in)
{}


basic_memfile::basic_memfile(std::ios::openmode mode)
{
    _M_mode = mode;
    _M_file = std::tmpfile();
    if (_M_file == nullptr) {
        throw std::invalid_argument("basic_memfile: unable to create temporary file");
    }
}


basic_memfile::basic_memfile()
: basic_memfile(std::ios::in)
{}


bool
basic_memfile::is_mapped() const noexcept
{
    return _M_map != MAP_FAILED && _M_map != nullptr;
}


basic_memfile&
basic_memfile::declare_mapped()
{
    if (is_mapped()) {
        return *this;
    }

    int fd = fileno(_M_file);
    if (fd == -1) {
        throw std::invalid_argument("basic_memfile: unable to get file descriptor for a file");
    }

    int prot = PROT_READ;
    int flags = MAP_PRIVATE;

    if (_M_mode & std::ios::out) {
        prot |= PROT_WRITE;
        flags = MAP_SHARED;
    }

    _M_map = static_cast<char_type*>(mmap(nullptr, _M_file_size, prot, flags, fd, 0));
    if (_M_map == MAP_FAILED) {
        throw std::invalid_argument("basic_memfile: unable to memory-map safetensors a file");
    }

    return *this;
}


basic_memfile&
basic_memfile::undeclare_mapped()
{
    if (is_mapped()) {
        if (writable()) {
            msync(_M_map, _M_file_size, MS_SYNC);
        }

        munmap(_M_map, _M_file_size);
        _M_map = nullptr;
    }

    return *this;
}


std::size_t
basic_memfile::size() const noexcept
{
    return _M_file_size;
}


const basic_memfile::char_type*
basic_memfile::data() const noexcept
{
    return _M_map;
}


basic_memfile::char_type*
basic_memfile::data() noexcept
{
    return _M_map;
}


basic_memfile::pos_type
basic_memfile::tellp() const noexcept
{
    return _M_file_p;
}


basic_memfile::pos_type
basic_memfile::tellg() const noexcept
{
    return _M_file_g;
}


basic_memfile&
basic_memfile::read(char_type* d, std::size_t size)
{
    if (!is_mapped()) {
        auto read = std::fread(d, sizeof(char_type), size, _M_file);
        _M_file_g += read;
        _M_file_size += read;
        return *this;
    }

    if (_M_file_size < _M_file_g + size) {
        throw std::out_of_range(
            std::format("file is too short ({}) to read ({}) more bytes", _M_file_size, size)
        );
    }

    std::memcpy(d, _M_map + _M_file_g, size);
    _M_file_g += size;

    return *this;
}


basic_memfile&
basic_memfile::read(void* d, std::size_t size)
{
    return read(static_cast<char_type*>(d), size);
}


bool
basic_memfile::writable() const
{
    return is_mapped() && static_cast<bool>(_M_mode & std::ios::out);
}


basic_memfile&
basic_memfile::write(const char_type* s, std::size_t size)
{
    if (writable()) {
        throw std::runtime_error("basic_memfile: file is not opened in a write mode");
    }

    auto written = std::fwrite(s, sizeof(char_type), size, _M_file);
    _M_file_p += written;
    _M_file_size += written;

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


void
basic_memfile::close()
{
    undeclare_mapped();

    if (_M_file != nullptr) {
        std::fclose(_M_file);
        _M_file = nullptr;
    }
}


basic_memfile::~basic_memfile() { close(); }


} // namespace metalchat
