#pragma once

#include <algorithm>
#include <filesystem>
#include <format>
#include <istream>
#include <numeric>
#include <streambuf>
#include <unordered_map>
#include <vector>

#include <metalchat/accelerator.h>
#include <metalchat/allocator.h>
#include <metalchat/container.h>
#include <metalchat/dtype.h>
#include <metalchat/layer.h>
#include <metalchat/tensor/basic.h>


namespace metalchat {


class spanbuf : public std::streambuf {
    using streambuf_type = std::streambuf;

    std::span<char_type> _M_buf;

public:
    using char_type = streambuf_type::char_type;
    using int_type = streambuf_type::int_type;
    using pos_type = streambuf_type::pos_type;
    using off_type = streambuf_type::off_type;
    using traits_type = streambuf_type::traits_type;

    spanbuf(char_type* data, pos_type size) noexcept
    : streambuf_type()
    {
        span(std::span(data, size));
    }

    spanbuf(const spanbuf&) = delete;

    void
    span(std::span<char_type> s) noexcept
    {
        _M_buf = s;
        setg(_M_buf.data(), _M_buf.data(), _M_buf.data() + _M_buf.size());
    }

protected:
    std::streambuf*
    setbuf(char_type* s, std::streamsize n) override
    {
        span(std::span(s, n));
        return this;
    }

    pos_type
    seekoff(
        off_type off, std::ios_base::seekdir way, std::ios_base::openmode which = std::ios_base::in
    ) override
    {
        pos_type pos = pos_type(off_type(-1));
        which &= std::ios_base::in;

        if (!which) {
            return pos;
        }

        switch (way) {
        case std::ios_base::beg:
            pos = 0;
            break;
        case std::ios_base::cur:
            pos = gptr() - eback();
            break;
        case std::ios_base::end:
            pos = egptr() - eback();
            break;
        default:
            return pos;
        }

        if (pos < 0 || pos > _M_buf.size()) {
            return pos_type(off_type(-1));
        }

        setg(eback(), eback() + pos, egptr());
        return pos;
    }

    pos_type
    seekpos(pos_type sp, std::ios_base::openmode which = std::ios_base::in) override
    {
        return seekoff(off_type(sp), std::ios_base::beg, which);
    }
};


struct safetensor_metadata {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> data_offsets;

    std::size_t
    size() const
    {
        return data_offsets[1] - data_offsets[0];
    }
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


class safetensor_typeinfo {
public:
    template <typename T> struct safetensor_type {
        using value_type = T;

        std::string_view name;
        std::size_t bytes;
    };

    static constexpr auto default_types = std::make_tuple(
        safetensor_type<bool>{"BOOL", 8},
        safetensor_type<std::int8_t>{"I8", 8},
        safetensor_type<std::uint8_t>{"U8", 8},
        safetensor_type<std::int16_t>{"I16", 16},
        safetensor_type<std::uint16_t>{"U16", 16},
        safetensor_type<dtype::bf16>{"BF16", 16},
        safetensor_type<std::int32_t>{"I32", 32},
        safetensor_type<std::uint32_t>{"U32", 32},
        safetensor_type<float>{"F32", 32},
        safetensor_type<double>{"F64", 64},
        safetensor_type<std::int64_t>{"I64", 64},
        safetensor_type<std::uint64_t>{"U64", 64}
    );

    using key_type = std::reference_wrapper<const std::type_info>;
    using value_type = std::pair<std::string, std::size_t>;

    safetensor_typeinfo()
    : _M_type_info()
    {
        constexpr auto default_types_size = std::tuple_size_v<decltype(default_types)>;
        register_default_types(std::make_index_sequence<default_types_size>{});
    }

    safetensor_typeinfo(const safetensor_typeinfo&) = default;

    const value_type&
    operator[](const std::type_info& info)
    {
        return _M_type_info[key_type(info)];
    }

    template <typename T>
    void
    register_type(const std::string& type_name, std::size_t type_size)
    {
        _M_type_info.insert_or_assign(typeid(T), value_type(type_name, type_size));
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

    template <std::size_t... TypeIndices>
    void
    register_default_types(std::index_sequence<TypeIndices...>)
    {
        (register_default_type<TypeIndices>(), ...);
    }

    template <std::size_t TypeIndex>
    void
    register_default_type()
    {
        using safetensor_type = std::tuple_element_t<TypeIndex, decltype(default_types)>;
        using value_type = safetensor_type::value_type;

        auto default_type = std::get<TypeIndex>(default_types);
        register_type<value_type>(std::string(default_type.name), default_type.bytes);
    }
};


/// A safetensor allocator is used to dynamically (in run-time) dispatch allocator type
/// binding according to the type of a tensor specified in the safetensor document.
///
/// This type is used internally within a \ref safetensor_document and does not expose
/// public API for registering new, unsupported types.
template <allocator_t<void> Allocator> class safetensor_allocator {
public:
    /// Type of the allocated container type. All containers are inherited from the
    /// basic containers, therefore allocator returns a polymorphic reference to the
    /// actual container implementation.
    using container_ptr = std::shared_ptr<basic_container>;

    /// A \ref safetensor_allocator default constructor.
    safetensor_allocator()
    : _M_type_alloc()
    {
        using safetensor_types = decltype(safetensor_typeinfo::default_types);
        constexpr auto default_types_size = std::tuple_size_v<safetensor_types>;

        register_default_types(std::make_index_sequence<default_types_size>{});
    }

    safetensor_allocator(const safetensor_allocator&) = default;

    /// Allocate an a block of contiguous memory of the specified type and initialize it with
    /// the data specified by the argument `data`.
    ///
    /// \param type_name a type name of container elements (e.g. 'I32', 'F32', 'F64', etc.).
    /// \param data a contiguous block of data to initialize new memory with.
    /// \param size a size of a new container in bytes.
    /// \param alloc a basic void allocator to use for typed allocation.
    container_ptr
    allocate(const std::string& type_name, void* data, std::size_t size, Allocator& alloc)
    {
        auto& [_, allocator] = _M_type_alloc[type_name];
        return allocator(data, size, alloc);
    }

    /// Allocate an uninitialized a block of contiguous memory of the specified type.
    ///
    /// \param type_name a type name of container elements (e.g. 'I32', 'F32', 'F64', etc.).
    /// \param size a size of a new container in bytes.
    /// \param alloc a basic void allocator to use for typed allocation.
    container_ptr
    allocate(const std::string& type_name, std::size_t size, Allocator& alloc)
    {
        auto& [allocator, _] = _M_type_alloc[type_name];
        return allocator(size, alloc);
    }

private:
    /// A function pointer type to the allocator that creates uninitialized contiguous
    /// block of memory.
    using make_alloc = container_ptr (*)(std::size_t, const Allocator&);

    /// A function pointer type to the allocator that creates contiguous block of memory
    /// and initialized it with the data specified by the `const void*` data pointer.
    using copy_alloc = container_ptr (*)(const void*, std::size_t, const Allocator&);

    /// Store both function pointers within the same container for the ease of access.
    using container_alloc = std::pair<make_alloc, copy_alloc>;

    std::unordered_map<std::string, container_alloc, _StringHash> _M_type_alloc;

    template <typename T>
    void
    register_type(const std::string& type_name)
    {
        using value_type = std::remove_cvref_t<T>;
        using allocator_type = rebind_allocator<value_type, Allocator>;

        auto malloc = static_cast<make_alloc>(&allocator_type::static_allocate);
        auto calloc = static_cast<copy_alloc>(&allocator_type::static_allocate);

        _M_type_alloc[type_name] = container_alloc(malloc, calloc);
    }

    template <std::size_t... TypeIndices>
    void
    register_default_types(std::index_sequence<TypeIndices...>)
    {
        (register_default_type<TypeIndices>(), ...);
    }

    template <std::size_t TypeIndex>
    void
    register_default_type()
    {
        using safetensor_types = decltype(safetensor_typeinfo::default_types);
        using safetensor_type = std::tuple_element_t<TypeIndex, safetensor_types>;
        using value_type = safetensor_type::value_type;

        auto default_type = std::get<TypeIndex>(safetensor_typeinfo::default_types);
        register_type<value_type>(std::string(default_type.name));
    }
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


/// A document for writing and reading tensors in a `safetensor` format.
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
    parse_metadata(std::istream& is);

    void
    insert(const safetensor_metadata& metadata, const safetensor_container& container);

    void
    load(const safetensor& st, basic_tensor& tensor) const;

public:
    using iterator = safetensor_iterator;
    using const_iterator = const iterator;

    /// A default \ref safetensor_document constructor.
    safetensor_document();

    /// A \ref safetensor_document copt constructor.
    safetensor_document(const safetensor_document&) = default;

    /// Returns tensor offsets (relative to a safetensor metadata header) in bytes.
    std::vector<std::size_t>
    offsets() const;

    /// Returns a list of tensor sizes in bytes.
    std::vector<std::size_t>
    sizes() const;

    /// Returns an iterator to the first safe tensor in a document.
    ///
    /// \note It's guaranteed that tensors are returned in an order defined by their offset
    /// in a document.
    ///
    /// ```c++
    /// auto document = safetensor_document::open("model.safetensors");
    /// for (auto it = document.begin(); it != document.end(); ++it) {
    ///     std::cout << (*it) << std::endl;
    /// }
    /// ```
    iterator
    begin();

    /// Returns an iterator past the last safe tensor in a document.
    iterator
    end();

    /// Returns a constant iterator to the first safe tensor in a document.
    const_iterator
    begin() const;

    /// Returns an iterator past the last safe tensor in a document.
    const_iterator
    end() const;

    /// Open a safetensor document.
    ///
    /// This implementation uses a memory-mapped file and allocates all tensors into
    /// \ref random_memory_container without copying actual memory. It is safe to destroy
    /// this instance after accessing tensors, since tensor pointers will carry over a
    /// pointer to the backing file. This means, until a pointer to the container exists,
    /// a memory-mapped file won't be closed.
    ///
    /// \param p A path in the filesystem to a file in a safetensor format.
    static safetensor_document
    open(const std::filesystem::path& p);

    /// Open a safetensor document.
    ///
    /// This implementation, like \ref safetensor_document::open(const std::filesystem::path&) uses
    /// a memory-mapped files. But all tensors are allocated using \ref hardware_memory_container.
    ///
    /// Similarly, pointer to a memory-mapped file is carried over by the tensors.
    ///
    /// This is the most efficient implementation, since it tries to allocate buffers of
    /// maximally allowed size by a hardware accelerator, and then uses
    /// \ref pooling_allocator_adapter and \ref nocopy_allocator to avoid copying memory from
    /// the memory-mapped file.
    ///
    /// \param p A path in the filesystem to a file in a safetensor format.
    /// \param accelerator A hardware accelerator.
    static safetensor_document
    open(const std::filesystem::path& p, hardware_accelerator& accelerator);

    /// Open a safetensor document.
    ///
    /// This implementation reads safetensor data from the specified basic stream. So all reads
    /// from the stream will result in copying data from stream to tensor containers. The
    /// containers do not hold a reference to the specified stream.
    ///
    /// \tparam Allocator A type of the allocator used to allocate tensor containers.
    /// \param is An input string stream, that will be used to retrieve tensors from.
    /// \param alloc An instance of a void container allocator to allocate tensor containers.
    template <allocator_t<void> Allocator>
    static safetensor_document
    open(std::istream& is, Allocator alloc)
    {
        auto metadata = parse_metadata(is);

        safetensor_document document;
        safetensor_allocator<Allocator> allocator;

        for (const auto& m : metadata) {
            auto container_ptr = allocator.allocate(m.size(), alloc);

            is.read(container_ptr->data_ptr(), m.size());
            if (is.gcount() != m.size()) {
                throw std::runtime_error(std::format(
                    "safetensor_document::open: unable to read tensor of size {}", m.size()
                ));
            }

            document.insert(m, container_ptr);
        }

        return document;
    }

    /// Open a safetensor document
    ///
    /// This implementation reads safetensor data from the specified memory-mapped file and then
    /// uses a paginated allocator to create large metal buffers to allocate tensors from. All
    /// containers hold a pointer to the opened file.
    ///
    /// \tparam Allocator A type of the allocator used to allocate tensor containers.
    /// \param p A path in the filesystem to a file in a safetensor format.
    /// \param alloc An instance of the Allocator type.
    /// \param max_size A maximum size of the buffer to allocate.
    template <allocator_t<void> Allocator>
    static safetensor_document
    open(const std::filesystem::path& p, Allocator& alloc, std::size_t max_size = -1)
    {
        auto file = std::make_shared<basic_memfile>(p);
        file->declare_mapped();

        spanbuf streambuf(file->data(), file->size());
        std::istream safetensor_stream(&streambuf);

        auto metadata = parse_metadata(safetensor_stream);

        std::vector<std::size_t> sizes;
        for (const auto& m : metadata) {
            sizes.push_back(m.size());
        }

        auto data_ptr = file->data() + safetensor_stream.tellg();

        // Use an aliasing allocator to bind file pointer to container pointer,
        // so that file is closed (and evicted from mapped memory), only when
        // all sub-allocated containers are also destroyed.
        auto aliasing_alloc = aliasing_allocator(std::forward<Allocator>(alloc), file);

        // Some Apple devices limit the memory that is possible to allocated within
        // a single buffer, here we define a paginated allocator to split memory-mapped
        // file into the non-overlapping contiguous containers.
        auto page_alloc = paginated_allocator_adapter(std::move(aliasing_alloc), max_size);
        auto containers = page_alloc.allocate(data_ptr, sizes);

        /// Independently of the specified base allocator, construct a final document,
        /// for this purpose, build containers relative to the paginated containers
        /// (instead of relative to the mapped file).
        char* container_data_ptr = nullptr;
        if (!containers.empty()) {
            container_data_ptr = static_cast<char*>(containers.front()->data());
        }

        // All containers allocated by paginated allocator contain an alias to the
        // memory-mapped file, so there only thing that is left is to allocate memory
        // from those containers.
        using allocator_type = pooling_allocator_adapter<null_allocator<Allocator>>;
        auto container_alloc = allocator_type(null_allocator<Allocator>{}, containers);

        safetensor_document document;
        safetensor_allocator<allocator_type> allocator;

        for (const auto& m : metadata) {
            auto data = container_data_ptr + m.data_offsets[0];

            auto container_ptr = allocator.allocate(m.dtype, data, m.size(), container_alloc);
            document.insert(m, std::move(container_ptr));
        }

        return document;
    }

    /// Open a safetensor document.
    ///
    /// This implementation is similar to
    /// \ref safetensor_document::open(const std::filesystem::path&, Allocator&, std::size_t),
    /// except that allocator must be an r-value.
    template <allocator_t<void> Allocator>
    static safetensor_document
    open(const std::filesystem::path& p, Allocator&& alloc, std::size_t max_size = -1)
    {
        return open(p, alloc, max_size);
    }

    /// Insert a tensor into the safetensor document.
    ///
    /// The implementation saves a pointer to the underlying container, so the tensor referring
    /// to that container could be destroyed by a caller.
    ///
    /// Example:
    /// ```c++
    /// auto weight = zeros<float>({3, 4});
    ///
    /// safetensor_document doc;
    /// doc.insert("weight", weight);
    /// doc.save("weights.safetensors");
    /// ```
    ///
    /// \param name A name of the tensor to insert.
    /// \param tensor A tensor data to insert into the safetensor document.
    void
    insert(const std::string& name, const basic_tensor& tensor);

    /// Insert all registered parameters of the specified layer.
    ///
    /// This method recursively traverses layer and inserts parameters into the safetensor
    /// document.
    ///
    /// Example:
    /// ```c++
    /// auto linear = nn::linear<float>({10, 64});
    ///
    /// safetensor_document doc;
    /// doc.insert(linear);
    /// doc.save("linear.safetensors");
    /// ```
    ///
    /// \param layer A layer to use.
    void
    insert(const basic_layer& layer);

    /// Load memory containers from a safetensor document into a layer.
    ///
    /// The implementation is identical to the \ref safetensor_document::load(basic_layer&), the
    /// difference is that safetensor file is not returned to the caller.
    ///
    /// \warning Layer parameters should be using the same container type as the safetensor
    /// document.
    ///
    /// \param p A path to load tensors from.
    /// \param layer A layer instance to load tensors into.
    static void
    load(const std::filesystem::path& p, basic_layer& layer);

    /// Load memory containers from a safetensor document into a layer.
    ///
    /// The traverses through all tensors in the \ref safetensor_document and assigns them to
    /// the registered parameters of the specified layer. Method raises an exception, when
    /// the parameter is not registered in the layer, but is presented in the document.
    ///
    /// \note Consider using \ref safetensor_document::begin() and \ref safetensor_document::end()
    /// iterators to implement a custom logic of weights assignment.
    ///
    /// \warning Layer parameters should be using the same container type as the safetensor
    /// document.
    ///
    /// Example:
    /// ```c++
    /// hardware_accelerator accelerator;
    /// nn::linear<float> linear(accelerator);
    ///
    /// auto doc = safetensor_document::open("linear.safetensors", accelerator);
    /// doc.load(linear);
    /// ```
    void
    load(basic_layer& layer) const;

    /// Load memory container from a safetensor document into a tensor.
    ///
    /// The implementation assigns a new container to the specified tensor (which means that
    /// target tensor might be empty or any arbitrary size), and resets the size of the tensor
    /// to correctly address elements of the new container. The method resets offsets if they
    /// were set in the target tensor, see \ref tensor_accessor::resize for more details.
    ///
    /// Depending on the allocator type and the way safetensor document was opened, new container
    /// might alias a pointer to the resources that were used to create a container (like memory-
    /// mapped files).
    ///
    /// \warning Tensor should be using the same container type as the safetensor document.
    ///
    /// Example:
    /// ```c++
    /// tensor<float> target;
    /// auto doc = safetensor_document::open("linear.safetensors");
    /// doc.load("weight", target);
    /// ```
    ///
    /// \param name A name of the tensor to load.
    /// \param tensor A target tensor that will be updated.
    void
    load(const std::string& name, basic_tensor& tensor) const;

    /// Save all registered parameters of the layer into the file at the specified location.
    ///
    /// \warning Layer parameters should be using the same container type as the safetensor
    /// document.
    ///
    /// \param p A path to the file to save tensors.
    /// \param layer A layer containing parameters to save into the safetensors document.
    static void
    save(const std::filesystem::path& p, basic_layer& layer);

    /// Save all registered tensors into the file at the specified location.
    ///
    /// \param p A path to the file to save tensors.
    void
    save(const std::filesystem::path& p);
};


} // namespace metalchat
