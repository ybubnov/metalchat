// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <jsoncons/json.hpp>

#include <metalchat/safetensor.h>


JSONCONS_ALL_MEMBER_TRAITS(metalchat::safetensor_metadata, dtype, shape, data_offsets);


namespace metalchat {


safetensor_document::safetensor_document()
: _M_metadata(),
  _M_containers(),
  _M_typeinfo()
{}


std::vector<std::size_t>
safetensor_document::offsets() const
{
    std::vector<std::size_t> offsets;
    for (const auto& m : _M_metadata) {
        offsets.push_back(m.data_offsets[0]);
    }
    return offsets;
}


std::vector<std::size_t>
safetensor_document::sizes() const
{
    std::vector<std::size_t> result;
    for (const auto& metadata : _M_metadata) {
        result.push_back(metadata.data_offsets[1] - metadata.data_offsets[0]);
    }
    return result;
}


safetensor_document::iterator
safetensor_document::begin()
{
    return iterator(_M_metadata.begin(), _M_containers.begin());
}


safetensor_document::iterator
safetensor_document::end()
{
    return iterator(_M_metadata.end(), _M_containers.end());
}


safetensor_document::const_iterator
safetensor_document::begin() const
{
    return iterator(_M_metadata.begin(), _M_containers.begin());
}


safetensor_document::const_iterator
safetensor_document::end() const
{
    return iterator(_M_metadata.end(), _M_containers.end());
}


std::vector<safetensor_metadata>
safetensor_document::parse_metadata(std::istream& is)
{
    // Read the length of the header and then the header itself, ensure that the
    // the file contains enough data to avoid reading from inaccessible regions.
    uint64_t header_size = 0;
    is.read((char*)&header_size, sizeof(header_size));
    if (is.gcount() != sizeof(header_size)) {
        throw std::runtime_error(std::format(
            "safetensor_document: header size is corrupted, read {} != {}", is.gcount(),
            sizeof(header_size)
        ));
    }

    char header_bytes[header_size];
    is.read((char*)&header_bytes, sizeof(header_bytes));
    if (is.gcount() != sizeof(header_bytes)) {
        throw std::runtime_error(std::format(
            "safetensor_document: header is corrupted, read {} != {}", is.gcount(),
            sizeof(header_bytes)
        ));
    }

    std::string_view header(header_bytes, sizeof(header_bytes));
    std::vector<safetensor_metadata> metadata;

    using metadata_type = std::map<std::string, std::string>;
    using value_type = std::variant<safetensor_metadata, metadata_type>;
    using header_type = std::map<std::string, value_type>;

    auto tensor_metadata = jsoncons::decode_json<header_type>(header);

    for (auto [name, value] : tensor_metadata) {
        // Shame on designers of safetensor specification, garbage like this should
        // not be presented on the same level as tensor structures.
        if (name == "__metadata__") {
            continue;
        }

        auto meta = std::get<safetensor_metadata>(value);
        meta.name = name;
        metadata.push_back(meta);
    }

    // Order metadata entries to ensure that we access file sequentially.
    auto metadata_comp = [](const safetensor_metadata& a, const safetensor_metadata& b) {
        return a.data_offsets[0] < b.data_offsets[0];
    };
    std::sort(metadata.begin(), metadata.end(), metadata_comp);

    return metadata;
}


safetensor_document
safetensor_document::open(const std::filesystem::path& p)
{
    auto alloc = random_memory_allocator<void>();
    auto nocopy_alloc = nocopy_allocator(alloc);
    return open(p, nocopy_alloc);
}


safetensor_document
safetensor_document::open(const std::filesystem::path& p, hardware_accelerator& accelerator)
{
    auto alloc = accelerator.get_allocator();
    auto nocopy_alloc = nocopy_allocator(alloc, accelerator.get_metal_device());
    auto resident_alloc = hardware_resident_allocator(nocopy_alloc, accelerator.get_metal_device());

    using allocator_type = decltype(resident_alloc);

    return open(p, std::forward<allocator_type>(resident_alloc), accelerator.max_buffer_size());
}


void
safetensor_document::insert(
    const safetensor_metadata& metadata, const safetensor_container& container
)
{
    _M_names.insert_or_assign(metadata.name, _M_metadata.size());
    _M_metadata.push_back(metadata);
    _M_containers.push_back(container);
}


std::size_t
safetensor_document::container_offset() const
{
    std::size_t offset = 0;
    if (_M_metadata.size() > 0) {
        const auto& last = _M_metadata.back();
        offset = last.data_offsets[1];
    }
    return offset;
}


void
safetensor_document::insert(const safetensor& st)
{
    auto sizes = st.sizes();
    auto offset = container_offset();
    auto container_ptr = st.container_ptr();

    safetensor_metadata metadata{
        .name = st.name(),
        .dtype = st.dtype(),
        .shape = {sizes.begin(), sizes.end()},
        .data_offsets = {offset, offset + container_ptr->size()}
    };

    insert(metadata, container_ptr);
}


void
safetensor_document::insert(const std::string& name, const basic_tensor& tensor)
{
    auto sizes = tensor.sizes();
    auto offset = container_offset();
    auto container_ptr = tensor.container_ptr();

    const auto& [dtype_name, _] = _M_typeinfo[tensor.dtype()];

    safetensor_metadata metadata{
        .name = name,
        .dtype = dtype_name,
        .shape = {sizes.begin(), sizes.end()},
        .data_offsets = {offset, offset + container_ptr->size()}
    };

    insert(metadata, container_ptr);
}


void
safetensor_document::insert(const std::string& name, const std::string& source)
{
    auto pos = _M_names.at(source);
    auto metadata = _M_metadata[pos];
    auto container_ptr = _M_containers[pos];

    metadata.name = name;
    insert(metadata, container_ptr);
}


void
safetensor_document::insert(const nn::basic_layer& layer)
{
    auto insert_fn = [&](nn::named_parameter parameter) { insert(parameter.path, *parameter.ptr); };
    layer.apply(insert_fn, /*recurse=*/true);
}


void
safetensor_document::load(const safetensor& st, basic_tensor& tensor) const
{
    if (st.dimensions() != tensor.dimensions()) {
        throw std::runtime_error(std::format(
            "safetensor_document::load: target tensor '{}' dimensions are different {}!={}",
            st.name(), st.dimensions(), tensor.dimensions()
        ));
    }

    // Apart from setting a size for the tensor, adjust strides to correctly
    // access multi-dimensional data.
    tensor_accessor::resize(st.sizes(), tensor);

    auto ptr = st.container_ptr();
    tensor.set_container(ptr);
}


void
safetensor_document::load(const std::filesystem::path& p, nn::basic_layer& layer)
{
    auto document = safetensor_document::open(p, layer.accelerator());
    document.load(layer);
}


void
safetensor_document::load(const std::string& name, basic_tensor& tensor) const
{
    auto pos = _M_names.at(name);
    auto first = begin();
    std::advance(first, pos);

    auto st = *first;
    load(st, tensor);
}


void
safetensor_document::load(nn::basic_layer& layer) const
{
    for (auto it = begin(); it != end(); ++it) {
        auto safetensor = *it;
        auto parameter = layer.get_parameter(safetensor.name());
        load(safetensor, *parameter);
    }
}


void
safetensor_document::save(const std::filesystem::path& p, nn::basic_layer& layer)
{
    safetensor_document document;
    document.insert(layer);
    document.save(p);
}


void
safetensor_document::save(const std::filesystem::path& p)
{
    std::unordered_map<std::string, safetensor_metadata> metadata;
    for (auto m : _M_metadata) {
        metadata.insert_or_assign(m.name, m);
    }

    std::string header;
    jsoncons::encode_json(metadata, header);

    auto header_size = header.size();

    basic_memfile file(p, std::ios::out);
    file.write(&header_size, sizeof(header_size));
    file.write(header.c_str(), header_size);

    for (std::size_t i = 0; i < _M_metadata.size(); i++) {
        file.write(_M_containers[i]->data_ptr(), _M_metadata[i].size());
    }
}


} // namespace metalchat
