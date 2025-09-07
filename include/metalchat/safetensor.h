#pragma once

#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>
#include <vector>

#include <metalchat/accelerator.h>
#include <metalchat/allocator.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/layer.h>
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


class safetensor_document;


class safetensor_typeinfo {
public:
    using key_type = std::reference_wrapper<const std::type_info>;
    using value_type = std::pair<std::string, std::size_t>;

    safetensor_typeinfo()
    : _M_type_info()
    {
        _M_type_info[typeid(std::int32_t)] = {"I32", 32};
        _M_type_info[typeid(float)] = {"F32", 32};
        _M_type_info[typeid(double)] = {"F64", 64};
        _M_type_info[typeid(dtype::bf16)] = {"BF16", 16};
    }

    safetensor_typeinfo(const safetensor_typeinfo&) = default;

    const value_type&
    operator[](const std::type_info& info)
    {
        return _M_type_info[key_type(info)];
    }

private:
    struct type_hash {
        std::size_t
        operator()(key_type t) const
        {
            return t.get().hash_code();
        }
    };

    struct type_eq {
        bool
        operator()(key_type lhs, key_type rhs) const
        {
            return lhs.get() == rhs.get();
        }
    };

    std::unordered_map<key_type, value_type, type_hash, type_eq> _M_type_info;
};


class safetensor_iterator {
private:
    using metadata_type = std::vector<safetensor_metadata>;
    using metadata_iterator = std::vector<safetensor_metadata>::const_iterator;

    using safetensor_container = std::shared_ptr<memory_container<void>>;
    using container_iterator = std::vector<safetensor_container>::const_iterator;

    safetensor_iterator(metadata_iterator input, container_iterator container)
    : _M_input(input),
      _M_container(container)
    {}

    metadata_iterator _M_input;
    container_iterator _M_container;

    friend class safetensor_document;

public:
    using container_type = memory_container<void>;

    using value_type = safetensor<container_type>;

    using iterator_category = std::forward_iterator_tag;

    using iterator = safetensor_iterator;

    using reference = value_type&;

    using pointer = value_type*;

    using difference_type = std::ptrdiff_t;

    safetensor_iterator()
    : _M_input(),
      _M_container()
    {}

    safetensor_iterator(const iterator& it) = default;

    iterator&
    operator++()
    {
        _M_input++;
        _M_container++;
        return *this;
    }

    iterator
    operator++(int)
    {
        auto result = *this;
        ++*this;
        return result;
    }

    value_type
    operator*() const
    {
        auto metadata = *_M_input;
        auto container_ptr = *_M_container;
        return value_type(metadata.name, metadata.shape, container_ptr);
    }

    bool
    operator==(const iterator& rhs) const
    {
        return _M_input == rhs._M_input;
    }

    bool
    operator!=(const iterator& rhs) const
    {
        return !operator==(rhs);
    }
};


class safetensor_document {
private:
    using safetensor_container = std::shared_ptr<memory_container<void>>;

    std::vector<safetensor_metadata> _M_metadata;
    std::vector<safetensor_container> _M_containers;

    safetensor_typeinfo _M_typeinfo;

    /// Parse safetensor metadata from the given file.
    ///
    /// Read header length and JSON-serialized tensor definitions into the metadata
    /// structure. Elements of the resulting vector are sorted by data offset in
    /// increasing order.
    static std::vector<safetensor_metadata>
    parse_metadata(basic_memfile& file);

    static std::vector<safetensor_metadata>
    parse_metadata(std::shared_ptr<basic_memfile> file_ptr);

    template <contiguous_container Container>
    void
    push_back(const safetensor_metadata& metadata, const std::shared_ptr<Container>& container)
    {
        _M_metadata.push_back(metadata);
        _M_containers.push_back(container);
    }

public:
    safetensor_document();
    safetensor_document(const safetensor_document&) = default;

    void
    push_back(const std::string& name, basic_tensor& tensor);

    void*
    data() noexcept;

    // std::vector<std::size_t>
    // offsets() const;

    std::vector<std::size_t>
    sizes() const;

    safetensor_iterator
    begin() const
    {
        return safetensor_iterator(_M_metadata.begin(), _M_containers.begin());
    }

    safetensor_iterator
    end() const
    {
        return safetensor_iterator(_M_metadata.end(), _M_containers.end());
    }

    template <allocator_t<void> Allocator>
    static safetensor_document
    load(const std::filesystem::path& p, Allocator alloc)
    {
        safetensor_document document;

        auto file = std::make_shared<basic_memfile>(p);
        auto metadata = parse_metadata(file->declare_mapped());

        auto a_alloc = aliasing_allocator(alloc, file);

        for (const auto& m : metadata) {
            auto data = file->data() + m.data_offsets[0];
            auto size = m.data_offsets[1] - m.data_offsets[0];

            auto container_ptr = a_alloc.allocate(data, size);
            document.push_back(m, container_ptr);
        }

        return document;
    }

    /*
    template <template Readable, allocator_t<void> Allocator>
    static void
    load(const Readable& r, Allocator alloc)
    {}
    */

    /*
    template <allocator_t<void> Allocator>
    void
    load(basic_layer& layer, Allocator alloc)
    {
        using container_type = Allocator::container_type;
        layer.initialize(begin(), end(), alloc);
    }

    template <allocator_t<void> Allocator>
    static void
    load(const std::filesystem::path& p, basic_layer& layer, Allocator alloc)
    {
        auto file = std::make_shared<basic_memfile>(p);
        file->declare_mapped();
        safetensor_document document(file, safetensor_openmode::in);

        auto alloc0 = aliasing_allocator(alloc, file);
        return document.load(layer, alloc0);
    }

    template <typename T>
    static void
    load(const std::filesystem::path& p, basic_layer& layer)
    {
        auto alloc = layer.accelerator().get_allocator();
        return load(p, layer, make_rebind_allocator<T>(alloc));
    }
    */

    static void
    save(const std::filesystem::path& p, basic_layer& layer);

    void
    save(const std::filesystem::path& p);
};


} // namespace metalchat
