#pragma once

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <metalchat/accelerator.h>
#include <metalchat/allocator.h>
#include <metalchat/container.h>
#include <metalchat/tensor/basic.h>


namespace metalchat {


struct safetensor_metadata {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> data_offsets;
};


template <contiguous_container Container> class safetensor {
public:
    using container_type = Container;
    using container_pointer = std::shared_ptr<container_type>;
    using shape_type = std::vector<std::size_t>;

    safetensor(
        const std::string& name, const shape_type& shape, const container_pointer& container_ptr
    )
    : _M_name(name),
      _M_shape(shape),
      _M_container(container_ptr)
    {}

    const std::string&
    name() const
    {
        return _M_name;
    }

    /// Return the number of dimensions in the tensor.
    std::size_t
    dim() const
    {
        return _M_shape.size();
    }

    /// Return the total number of elements in the tensor.
    std::size_t
    numel() const
    {
        return std::accumulate(_M_shape.begin(), _M_shape.end(), 1, std::multiplies<std::size_t>());
    }

    const std::span<std::size_t>
    sizes() const
    {
        auto data_ptr = const_cast<std::size_t*>(_M_shape.data());
        return std::span<std::size_t>(data_ptr, _M_shape.size());
    }

    container_pointer
    container() const
    {
        return _M_container;
    }

    friend std::ostream&
    operator<<(std::ostream& os, const safetensor& st)
    {
        os << "safetensor(" << st._M_name << ", shape=[" << st._M_shape << "])";
        return os;
    }

private:
    std::string _M_name;
    shape_type _M_shape;
    container_pointer _M_container;
};


template <typename UnaryOp, typename Allocator>
concept invocable_with_safetensor = requires {
    requires allocator<Allocator>;
    requires std::invocable<UnaryOp, safetensor<typename Allocator::container_type>>;
};


class safetensor_document {
private:
    std::shared_ptr<basic_memfile> _M_file;
    std::vector<safetensor_metadata> _M_metadata;

public:
    safetensor_document(std::shared_ptr<basic_memfile> file);

    void*
    data() noexcept;

    std::vector<std::size_t>
    sizes() const;

    static std::vector<safetensor_metadata>
    load_header(basic_memfile& file);

    static std::vector<safetensor_metadata>
    load_header(std::shared_ptr<basic_memfile> file_ptr);

    template <allocator Allocator>
    static auto
    load(const std::filesystem::path& p, Allocator alloc)
    {
        auto file = std::make_shared<basic_memfile>(p);
        file->declare_mapped();

        auto alloc0 = aliasing_allocator(alloc, file);
        return load(alloc0);
    }

    template <hardware_allocator Allocator>
    static auto
    load(const std::filesystem::path& p, Allocator alloc)
    {
        auto file = std::make_shared<basic_memfile>(p);
        file->declare_mapped();
        auto document = safetensor_document(file);

        auto alloc0 = hardware_aliasing_allocator(alloc, file);
        return document.load(alloc0);
    }

    template <allocator Allocator>
    auto
    load(Allocator alloc)
    {
        using container_type = typename Allocator::container_type;
        using safetensor_type = safetensor<container_type>;

        std::unordered_map<std::string, safetensor_type> tensors;
        load(alloc, [&](safetensor_type s) { tensors.insert_or_assign(s.name(), s); });
        return tensors;
    }

    template <allocator Allocator, invocable_with_safetensor<Allocator> UnaryOp>
    void
    load(Allocator alloc, UnaryOp unary_op)
    {
        using value_type = typename Allocator::value_type;
        using const_pointer = typename Allocator::const_pointer;
        using container_type = typename Allocator::container_type;
        using container_pointer = typename Allocator::container_pointer;
        using tensor_type = safetensor<container_type>;

        for (const auto& tensor_metadata : _M_metadata) {
            auto tensor_pos = tensor_metadata.data_offsets[0];

            if (tensor_pos >= _M_file->size()) {
                throw std::runtime_error(std::format(
                    "safetensor: start data position {} for a tensor {} is out of bounds",
                    tensor_pos, tensor_metadata.name
                ));
            }

            auto tensor_shape = tensor_metadata.shape;
            auto tensor_numel = std::accumulate(
                tensor_shape.begin(), tensor_shape.end(), 1, std::multiplies<std::size_t>()
            );

            container_pointer container_ptr(nullptr);

            if (_M_file->is_mapped()) {
                const basic_memfile::char_type* container = _M_file->data() + tensor_pos;
                container_ptr = alloc.allocate((const_pointer)(container), tensor_numel);
            } else {
                auto container = std::make_shared<value_type[]>(tensor_numel);
                _M_file->read(container.get(), sizeof(value_type) * tensor_numel);

                container_ptr = alloc.allocate((const_pointer)(container.get()), tensor_numel);
            }

            auto tensor = tensor_type(tensor_metadata.name, tensor_metadata.shape, container_ptr);
            unary_op(tensor);
        }
    }
};


} // namespace metalchat
