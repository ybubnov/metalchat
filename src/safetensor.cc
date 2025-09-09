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


void*
safetensor_document::data() noexcept
{
    // std::size_t offset = _M_metadata.empty() ? 0 : _M_metadata.front().data_offsets[0];
    // return _M_file->data() + offset;
    return nullptr;
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

        for (auto& offset : meta.data_offsets) {
            offset += start_pos;
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
safetensor_document::insert(const std::string& name, basic_tensor& tensor)
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
safetensor_document::load(basic_layer& layer)
{
    for (auto it = begin(); it != end(); ++it) {
        auto safetensor = *it;
        auto parameter = layer.get_parameter(safetensor.name());

        if (safetensor.dimensions() != parameter->dimensions()) {
            throw std::runtime_error(std::format(
                "safetensor_document::load: target tensor '{}' dimensions are different {}!={}",
                safetensor.name(), safetensor.dimensions(), parameter->dimensions()
            ));
        }

        auto sizes = safetensor.sizes();

        for (std::size_t i = 0; i < parameter->dimensions(); i++) {
            parameter->set_size(i, sizes[i]);
        }

        auto ptr = safetensor.container_ptr();
        parameter->set_container(ptr);
    }
}


void
safetensor_document::save(const std::filesystem::path& p, basic_layer& layer)
{
    safetensor_document document;

    auto insert = [&](const std::string& name, std::shared_ptr<basic_tensor> tensor) {
        document.insert(name, *tensor);
    };

    layer.apply(insert, /*recurse=*/true);
    document.save(p);
}

void
safetensor_document::save(const std::filesystem::path& p)
{
    std::unordered_map<std::string, safetensor_metadata> metadata;
    for (const auto& m : _M_metadata) {
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
