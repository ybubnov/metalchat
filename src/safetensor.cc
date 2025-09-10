#include <glaze/json.hpp>

#include <metalchat/safetensor.h>


template <> struct glz::meta<metalchat::safetensor_metadata> {
    using T = metalchat::safetensor_metadata;

    static constexpr auto value
        = object("dtype", &T::dtype, "shape", &T::shape, "data_offsets", &T::data_offsets);
};


namespace metalchat {


// A compilation check to ensure correct reflection of the structure fields.
static_assert(glz::reflect<safetensor_metadata>::size == 3);


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


std::vector<safetensor_metadata>
safetensor_document::parse_metadata(basic_memfile& file)
{
    // Read the length of the header and then the header itself, ensure that the
    // the file contains enough data to avoid reading from inaccessible regions.
    uint64_t header_size = 0;
    file.read(&header_size, sizeof(header_size));

    char header_bytes[header_size];
    file.read(&header_bytes, sizeof(header_bytes));
    std::string_view header(header_bytes, sizeof(header_bytes));

    std::map<std::string, safetensor_metadata> tensor_metadata;
    std::vector<safetensor_metadata> metadata;

    auto json_context = glz::context{};
    constexpr auto json_opts = glz::opts{.error_on_unknown_keys = false};
    auto err = glz::read<json_opts>(tensor_metadata, header, json_context);
    if (err) {
        throw std::runtime_error(glz::format_error(err, header));
    }

    auto start_pos = file.tellg();
    for (auto [name, meta] : tensor_metadata) {
        // Shame on designers of safetensor specification, garbage like this should
        // not be presented on the same level as tensor structures.
        if (name == "__metadata__") {
            continue;
        }

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


std::vector<safetensor_metadata>
safetensor_document::parse_metadata(std::shared_ptr<basic_memfile> file_ptr)
{
    return parse_metadata(*file_ptr);
}


safetensor_document
safetensor_document::open(const std::filesystem::path& p)
{
    random_memory_allocator<void> alloc;
    return open(p, alloc);
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


void
safetensor_document::insert(const std::string& name, const basic_tensor& tensor)
{
    auto sizes = tensor.sizes();
    auto numel = tensor.numel();
    auto& [dtype_name, dtype_size] = _M_typeinfo[tensor.dtype()];

    std::size_t begin = 0;
    if (_M_metadata.size() > 0) {
        const auto& last = _M_metadata.back();
        begin = last.data_offsets[1];
    }

    // TODO: Does it make sense to assert that `end` is byte-aligned?
    std::size_t size = numel * dtype_size / 8;

    safetensor_metadata metadata{
        .name = name,
        .dtype = dtype_name,
        .shape = {sizes.begin(), sizes.end()},
        .data_offsets = {begin, begin + size}
    };

    insert(metadata, tensor.container_ptr());
}


void
safetensor_document::insert(const basic_layer& layer)
{
    auto insert_fn = [&](const std::string& name, std::shared_ptr<basic_tensor> tensor) {
        insert(name, *tensor);
    };
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

    auto sizes = st.sizes();

    for (std::size_t i = 0; i < tensor.dimensions(); i++) {
        tensor.set_size(i, sizes[i]);
    }

    auto ptr = st.container_ptr();
    tensor.set_container(ptr);
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
safetensor_document::load(basic_layer& layer) const
{
    for (auto it = begin(); it != end(); ++it) {
        auto safetensor = *it;
        auto parameter = layer.get_parameter(safetensor.name());
        load(safetensor, *parameter);
    }
}


void
safetensor_document::save(const std::filesystem::path& p, basic_layer& layer)
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
        for (auto& offset : m.data_offsets) {
        }
        metadata.insert_or_assign(m.name, m);
    }

    std::string header;
    auto err = glz::write_json(metadata, header);
    if (err) {
        throw std::runtime_error(glz::format_error(err, header));
    }

    auto header_size = header.size();

    basic_memfile file(p, "w");
    file.write(&header_size, sizeof(header_size));
    file.write(header.c_str(), header_size);

    for (std::size_t i = 0; i < _M_metadata.size(); i++) {
        // TODO: validate data sizes.
        auto size = _M_metadata[i].data_offsets[1] - _M_metadata[i].data_offsets[0];
        file.write(_M_containers[i]->data_ptr(), size);
    }
}


} // namespace metalchat
