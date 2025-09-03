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


template <allocator Allocator> class safetensor_iterator {
private:
    using metadata_type = std::vector<safetensor_metadata>;
    using metadata_iterator = std::vector<safetensor_metadata>::const_iterator;

    safetensor_iterator(
        std::shared_ptr<basic_memfile> file, metadata_iterator input, Allocator alloc
    )
    : _M_file(file),
      _M_input(input),
      _M_alloc(alloc)
    {}

    std::shared_ptr<basic_memfile> _M_file;
    metadata_iterator _M_input;
    Allocator _M_alloc;

    friend class safetensor_document;

public:
    using allocator_type = Allocator;
    using container_type = Allocator::container_type;

    using value_type = safetensor<container_type>;

    using iterator_category = std::forward_iterator_tag;

    using iterator = safetensor_iterator<Allocator>;

    using reference = value_type&;

    using pointer = value_type*;

    using difference_type = std::ptrdiff_t;

    safetensor_iterator()
    : _M_file(nullptr),
      _M_input(),
      _M_alloc()
    {}

    safetensor_iterator(const iterator& it) = default;

    iterator&
    operator++()
    {
        _M_input++;
        return *this;
    }

    iterator
    operator++(int)
    {
        auto result = *this;
        ++*_M_input;
        return result;
    }

    value_type
    operator*() const
    {
        using const_pointer = Allocator::const_pointer;
        using container_value_type = Allocator::value_type;
        using container_pointer = Allocator::container_pointer;

        auto& metadata = *_M_input;
        auto data_offset = metadata.data_offsets[0];

        if (data_offset >= _M_file->size()) {
            throw std::runtime_error(std::format(
                "safetensor_iterator: start data position {} for a tensor {} is out of bounds",
                data_offset, metadata.name
            ));
        }

        auto numel = std::accumulate(
            metadata.shape.begin(), metadata.shape.end(), 1, std::multiplies<std::size_t>()
        );

        container_pointer container_ptr(nullptr);
        auto alloc = const_cast<std::remove_const_t<Allocator>&>(_M_alloc);

        if (_M_file->is_mapped()) {
            const basic_memfile::char_type* container = _M_file->data() + data_offset;
            container_ptr = alloc.allocate((const_pointer)(container), numel);
        } else {
            auto container = std::make_shared<container_value_type[]>(numel);
            _M_file->read(container.get(), sizeof(container_value_type) * numel);

            container_ptr = alloc.allocate((const_pointer)(container.get()), numel);
        }

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


/// Specifies available file open flags for a safetensor document.
enum safetensor_openmode { in, out };


class safetensor_document {
private:
    std::shared_ptr<basic_memfile> _M_file;
    std::vector<safetensor_metadata> _M_metadata;
    safetensor_openmode _M_mode;
    safetensor_typeinfo _M_typeinfo;

    static std::vector<safetensor_metadata>
    parse_metadata(basic_memfile& file);

    static std::vector<safetensor_metadata>
    parse_metadata(std::shared_ptr<basic_memfile> file_ptr);

public:
    safetensor_document(std::shared_ptr<basic_memfile> file, safetensor_openmode mode);

    void*
    data() noexcept;

    std::vector<std::size_t>
    sizes() const;

    template <allocator Allocator>
    safetensor_iterator<Allocator>
    begin(Allocator alloc) const
    {
        return safetensor_iterator<Allocator>(_M_file, _M_metadata.begin(), alloc);
    }

    template <allocator Allocator>
    safetensor_iterator<Allocator>
    end(Allocator alloc) const
    {
        return safetensor_iterator<Allocator>(_M_file, _M_metadata.end(), alloc);
    }

    template <allocator Allocator>
    void
    load(basic_layer& layer, Allocator alloc)
    {
        using container_type = Allocator::container_type;
        layer.initialize<container_type>(begin<Allocator>(alloc), end<Allocator>(alloc));
    }

    template <allocator Allocator>
    static void
    load(const std::filesystem::path& p, basic_layer& layer, Allocator alloc)
    {
        auto file = std::make_shared<basic_memfile>(p);
        file->declare_mapped();
        safetensor_document document(file, safetensor_openmode::in);

        auto alloc0 = aliasing_allocator(alloc, file);
        return document.load(layer, alloc0);
    }

    template <hardware_allocator Allocator>
    static void
    load(const std::filesystem::path& p, basic_layer& layer, Allocator alloc)
    {
        auto file = std::make_shared<basic_memfile>(p);
        file->declare_mapped();
        safetensor_document document(file, safetensor_openmode::in);

        auto alloc0 = hardware_aliasing_allocator(alloc, file);
        return document.load(layer, alloc0);
    }

    template <typename T>
    static void
    load(const std::filesystem::path& p, basic_layer& layer)
    {
        auto alloc = layer.accelerator().get_allocator();
        return load(p, layer, make_rebind_allocator<T>(alloc));
    }

    void
    save(basic_layer& layer);

    static void
    save(const std::filesystem::path& p, basic_layer& layer)
    {
        auto file = std::make_shared<basic_memfile>(p, "w");
        safetensor_document document(file, safetensor_openmode::out);
        document.save(layer);
    }
};


} // namespace metalchat
