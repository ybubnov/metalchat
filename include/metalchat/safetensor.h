#pragma once

#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <metalchat/accelerator.h>
#include <metalchat/container.h>
#include <metalchat/format.h>
#include <metalchat/tensor/basic.h>


namespace metalchat {


class basic_memfile {
public:
    basic_memfile(const std::filesystem::path& p);

    std::size_t
    size() const noexcept;

    std::uint8_t*
    tellp() const;

    basic_memfile&
    read(void* dest, std::size_t size);

    ~basic_memfile();

private:
    std::FILE* _m_file = nullptr;
    std::size_t _m_file_size = 0;
    std::size_t _m_file_off = 0;
    std::uint8_t* _m_map = nullptr;
};


class safetensor {
public:
    using shape_type = std::vector<std::size_t>;
    using data_pointer = std::shared_ptr<void>;

    safetensor(const shape_type& shape, data_pointer data_ptr);

    /// Return the number of dimensions in the tensor.
    std::size_t
    dim() const;

    /// Return the total number of elements in the tensor.
    std::size_t
    numel() const;

    /// Cast this safe-tensor to the specified the specified type.
    ///
    /// Method infers the type of the final tensor from the `value_type` of the provided
    /// allocator type. Allocation of tensors storing incomplete types (including void)
    /// is prohibited.
    template <std::size_t N, allocator Allocator>
    auto
    as(Allocator alloc) const
    {
        using container_type = Allocator::container_type;
        using value_type = Allocator::value_type;
        using tensor_type = tensor<value_type, N, container_type>;

        /// Some allocators (like `hardware_nocopy_allocator`) could simply store a
        /// pointer to the original data pointer, so once safetensor is destroyed,
        /// container won't be valid anymore.
        ///
        /// To avoid this, share the ownership of safetensor with a new container.
        auto data = static_cast<value_type*>(_m_data_ptr.get());
        auto container_ptr = alloc.allocate(data, numel());

        /// Create an artificial pair type, so that is stores both pointers, and then
        /// leave an access only to the newly created container pointer.
        using value_pointer = std::shared_ptr<void>;
        using container_pointer = Allocator::container_pointer;
        using shared_pointer = std::pair<value_pointer, container_pointer>;

        /// Once container is destroyed, memory file is attempted to be destroyed as well.
        auto shared_ptr = std::make_shared<shared_pointer>(_m_data_ptr, container_ptr);
        container_ptr = container_pointer(shared_ptr, container_ptr.get());

        return tensor_type(_m_shape.cbegin(), _m_shape.cend(), container_ptr);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const safetensor& st);

private:
    shape_type _m_shape;
    data_pointer _m_data_ptr;
};


class safetensor_file {
public:
    using container_type = std::unordered_map<std::string, safetensor>;

    using iterator = container_type::iterator;

    using const_iterator = container_type::const_iterator;

    safetensor_file(const std::filesystem::path& p);

    std::size_t
    size() const noexcept;

    iterator
    begin();

    const_iterator
    begin() const;

    iterator
    end();

    const_iterator
    end() const;

    const_iterator
    find(const std::string& tensor_name) const;

    const safetensor&
    operator[](const std::string& tensor_name) const;

private:
    std::shared_ptr<basic_memfile> _m_memfile;
    std::unordered_map<std::string, safetensor> _m_tensors;

    void
    parse();
};


} // namespace metalchat
