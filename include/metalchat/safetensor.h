#pragma once

#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>
#include <unordered_map>
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


class safetensor {
public:
    using container_type = basic_container;
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
    dimensions() const
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
    container_ptr() const
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


template <allocator_t<void> Allocator> class safetensor_allocator {
public:
    using container_ptr = std::shared_ptr<basic_container>;
    using container_allocator = std::function<container_ptr(void*, std::size_t, Allocator&)>;

    safetensor_allocator()
    : _M_type_alloc()
    {
        register_type<std::int32_t>("I32");
        register_type<float>("F32");
        register_type<double>("F64");
        register_type<dtype::bf16>("BF16");
    }

    safetensor_allocator(const safetensor_allocator&) = default;

    container_ptr
    allocate(const std::string& type_name, void* data, std::size_t size, Allocator& alloc)
    {
        auto allocator = _M_type_alloc[type_name];
        return allocator(data, size, alloc);
    }

private:
    template <typename T>
    void
    register_type(const std::string& type_name)
    {
        _M_type_alloc[type_name] = allocator_rebinder<T, Allocator>::allocate;
    }

    std::unordered_map<std::string, container_allocator, _StringHash> _M_type_alloc;
};


class safetensor_typeinfo {
public:
    using key_type = std::reference_wrapper<const std::type_info>;
    using value_type = std::pair<std::string, std::size_t>;

    safetensor_typeinfo()
    : _M_type_info()
    {
        register_type<std::int32_t>("I32", 32);
        register_type<float>("F32", 32);
        register_type<double>("F64", 64);
        register_type<dtype::bf16>("BF16", 16);
    }

    safetensor_typeinfo(const safetensor_typeinfo&) = default;

    const value_type&
    operator[](const std::type_info& info)
    {
        return _M_type_info[key_type(info)];
    }

private:
    template <typename T>
    void
    register_type(const std::string& type_name, std::size_t type_size)
    {
        _M_type_info[typeid(T)] = {type_name, type_size};
    }

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

    using safetensor_container = std::shared_ptr<basic_container>;
    using container_iterator = std::vector<safetensor_container>::const_iterator;

    safetensor_iterator(metadata_iterator input, container_iterator container)
    : _M_input(input),
      _M_container(container)
    {}

    metadata_iterator _M_input;
    container_iterator _M_container;

    friend class safetensor_document;

public:
    using container_type = basic_container;

    using value_type = safetensor;

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
    using safetensor_container = std::shared_ptr<basic_container>;

    std::vector<safetensor_metadata> _M_metadata;
    std::vector<safetensor_container> _M_containers;
    std::unordered_map<std::string, std::size_t, _StringHash> _M_names;

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

    void
    insert(const safetensor_metadata& metadata, const safetensor_container& container);

public:
    using iterator = safetensor_iterator;
    using const_iterator = const iterator;

    safetensor_document();
    safetensor_document(const safetensor_document&) = default;

    void*
    data() noexcept;

    // std::vector<std::size_t>
    // offsets() const;

    std::vector<std::size_t>
    sizes() const;

    iterator
    begin()
    {
        return iterator(_M_metadata.begin(), _M_containers.begin());
    }

    iterator
    end()
    {
        return iterator(_M_metadata.end(), _M_containers.end());
    }

    const_iterator
    begin() const
    {
        return iterator(_M_metadata.begin(), _M_containers.begin());
    }

    const_iterator
    end() const
    {
        return iterator(_M_metadata.end(), _M_containers.end());
    }

    template <allocator_t<void> Allocator>
    static safetensor_document
    open(const std::filesystem::path& p, Allocator alloc)
    {
        using allocator_type = aliasing_allocator<Allocator>;

        safetensor_document document;
        safetensor_allocator<allocator_type> allocator;

        auto file = std::make_shared<basic_memfile>(p);
        auto metadata = parse_metadata(file->declare_mapped());
        auto container_alloc = allocator_type(alloc, file);

        for (const auto& m : metadata) {
            auto data = file->data() + m.data_offsets[0];
            auto size = m.data_offsets[1] - m.data_offsets[0];

            auto container_ptr = allocator.allocate(m.dtype, data, size, container_alloc);
            document.insert(m, container_ptr);
        }

        return document;
    }

    static safetensor_document
    open(const std::filesystem::path& p);

    void
    insert(const std::string& name, basic_tensor& tensor);

    // void
    // insert(basic_layer& layer);

    void
    load(basic_layer& layer);

    /*
    template <allocator_t<void> Allocator>
    void
    load(basic_layer& layer, Allocator alloc)
    {
        using container_type = Allocator::container_type;
        layer.initialize(begin(), end(), alloc);
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

    // friend std::ostream&
    // operator<<(std::ostream& os, const safetensor_document& document)
    // {
    //     return os;
    // }
    //
    // friend std::istream&
    // operator>>(std::istream& is, safetensor_document& document)
    // {
    //     return is;
    // }
};


} // namespace metalchat
