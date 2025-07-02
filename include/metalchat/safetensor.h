#pragma once

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <numeric>
#include <simdjson.h>

#include <metalchat/accelerator.h>
#include <metalchat/allocator.h>
#include <metalchat/container.h>
#include <metalchat/format.h>
#include <metalchat/tensor/basic.h>


using namespace simdjson;


namespace metalchat {


struct safetensor_metadata {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> data_offsets;
};


} // namespace metalchat


template <>
simdjson_inline simdjson_result<std::vector<std::size_t>>
simdjson::ondemand::value::get()
{
    ondemand::array array;
    auto error = get_array().get(array);

    if (error) {
        return error;
    }

    std::vector<std::size_t> vec;

    for (auto v : array) {
        int64_t val;
        if ((error = v.get_int64().get(val))) {
            return error;
        }
        // TODO: rework this implementation, so that vector is created with a
        // fixed number of elements and does not grow too much.
        vec.push_back(static_cast<std::size_t>(val));
    }
    return vec;
}


template <>
simdjson_inline simdjson_result<metalchat::safetensor_metadata>
simdjson::ondemand::value::get() noexcept
{
    ondemand::object object;
    auto error = get_object().get(object);
    if (error) {
        return error;
    }

    metalchat::safetensor_metadata meta;
    if ((error = object["dtype"].get_string(meta.dtype))) {
        return error;
    }
    if ((error = object["shape"].get<std::vector<std::size_t>>().get(meta.shape))) {
        return error;
    }
    if ((error = object["data_offsets"].get<std::vector<std::size_t>>().get(meta.data_offsets))) {
        return error;
    }

    return meta;
}


namespace metalchat {


template <contiguous_container Container> class safetensor {
public:
    using container_type = Container;
    using container_pointer = std::shared_ptr<container_type>;
    using shape_type = std::vector<std::size_t>;

    safetensor(
        const std::string& name, const shape_type& shape, const container_pointer& container_ptr
    )
    : _m_name(name),
      _m_shape(shape),
      _m_container(container_ptr)
    {}

    const std::string&
    name() const
    {
        return _m_name;
    }

    /// Return the number of dimensions in the tensor.
    std::size_t
    dim() const
    {
        return _m_shape.size();
    }

    /// Return the total number of elements in the tensor.
    std::size_t
    numel() const
    {
        return std::accumulate(_m_shape.begin(), _m_shape.end(), 1, std::multiplies<std::size_t>());
    }

    const std::span<std::size_t>
    sizes() const
    {
        auto data_ptr = const_cast<std::size_t*>(_m_shape.data());
        return std::span<std::size_t>(data_ptr, _m_shape.size());
    }

    container_pointer
    container() const
    {
        return _m_container;
    }

    friend std::ostream&
    operator<<(std::ostream& os, const safetensor& st)
    {
        os << "safetensor(" << st._m_name << ", shape=[" << st._m_shape << "])";
        return os;
    }

private:
    std::string _m_name;
    shape_type _m_shape;
    container_pointer _m_container;
};


template <typename Constructor, typename Allocator>
concept invocable_with_safetensor = requires(Constructor constructor) {
    requires allocator<Allocator>;
    requires std::invocable<Constructor, safetensor<typename Allocator::container_type>>;
};


struct safetensors {

    template <allocator Allocator>
    static auto
    load(basic_memfile& file, Allocator alloc)
    {
        using container_type = typename Allocator::container_type;
        using safetensor_type = safetensor<container_type>;

        std::unordered_map<std::string, safetensor_type> tensors;
        load(file, alloc, [&](safetensor_type s) { tensors.insert_or_assign(s.name(), s); });
        return tensors;
    }

    template <allocator Allocator, invocable_with_safetensor<Allocator> Constructor>
    static void
    load(basic_memfile& file, Allocator alloc, Constructor constructor)
    {
        // Read the length of the header and then the header itself, ensure that the
        // the file contains enough data to avoid reading from inaccessible regions.
        uint64_t header_size = 0;
        file.read(&header_size, sizeof(header_size));

        char header[header_size];
        file.read(&header, sizeof(header));
        auto start_pos = file.tellg();

        simdjson::ondemand::parser json_parser;
        simdjson::padded_string header_padded(header, header_size);

        auto json_document = json_parser.iterate(header_padded);
        auto json_object = json_document.get_object();

        using value_type = typename Allocator::value_type;
        using const_pointer = typename Allocator::const_pointer;
        using container_type = typename Allocator::container_type;
        using container_pointer = typename Allocator::container_pointer;
        using tensor_type = safetensor<container_type>;

        std::vector<safetensor_metadata> metadata;

        for (auto json_field : json_object) {
            std::string_view field_name = json_field.unescaped_key();
            if (field_name == "__metadata__") {
                continue;
            }

            safetensor_metadata tensor_metadata;
            auto error = json_field.value().get<safetensor_metadata>().get(tensor_metadata);
            if (error) {
                throw std::runtime_error(simdjson::error_message(error));
            }

            tensor_metadata.name = field_name;
            metadata.push_back(tensor_metadata);
        }

        // Order metadata entries to ensure that we access file sequentially.
        auto metadata_comp = [](const safetensor_metadata& a, const safetensor_metadata& b) {
            return a.data_offsets[0] < b.data_offsets[0];
        };
        std::sort(metadata.begin(), metadata.end(), metadata_comp);

        for (const auto& tensor_metadata : metadata) {
            auto tensor_pos = start_pos + tensor_metadata.data_offsets[0];

            if (tensor_pos >= file.size()) {
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

            if (file.is_mapped()) {
                const basic_memfile::char_type* container = file.data() + tensor_pos;
                container_ptr = alloc.allocate((const_pointer)(container), tensor_numel);
            } else {
                auto container = std::make_shared<value_type[]>(tensor_numel);
                file.read(container.get(), sizeof(value_type) * tensor_numel);

                container_ptr = alloc.allocate((const_pointer)(container.get()), tensor_numel);
            }

            auto tensor = tensor_type(tensor_metadata.name, tensor_metadata.shape, container_ptr);
            constructor(tensor);
        }
    }
};


} // namespace metalchat
