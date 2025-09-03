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


safetensor_document::safetensor_document(
    std::shared_ptr<basic_memfile> file, safetensor_openmode mode
)
: _M_file(file),
  _M_metadata(),
  _M_mode(mode),
  _M_typeinfo()
{
    if (_M_mode == safetensor_openmode::in) {
        _M_metadata = parse_metadata(file);
    }
}


void*
safetensor_document::data() noexcept
{
    std::size_t offset = _M_metadata.empty() ? 0 : _M_metadata.front().data_offsets[0];
    return _M_file->data() + offset;
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


void
safetensor_document::save(basic_layer& layer)
{
    std::size_t begin = 0;
    std::unordered_map<std::string, safetensor_metadata> tensor_metadata;
    std::vector<const void*> tensor_data;

    auto emplace_metadata = [&](const std::string& name, std::shared_ptr<basic_tensor> tensor) {
        auto sizes = tensor->sizes();
        auto numel = tensor->numel();

        auto& [dtype_name, dtype_size] = _M_typeinfo[tensor->dtype()];
        auto end = begin + numel * dtype_size;

        // TODO: assert that end is byte-aligned.

        _M_metadata.push_back(safetensor_metadata{
            .name = name,
            .dtype = dtype_name,
            .shape = {sizes.begin(), sizes.end()},
            .data_offsets = {begin / 8, end / 8}
        });

        begin = end;

        tensor_data.push_back(tensor->data());
        tensor_metadata.insert_or_assign(name, _M_metadata.back());
    };

    layer.apply(emplace_metadata, /*recurse=*/true);

    std::string header;
    auto err = glz::write_json(tensor_metadata, header);
    if (err) {
        throw std::runtime_error(glz::format_error(err, header));
    }

    auto header_size = header.size();

    _M_file->write(&header_size, sizeof(header_size));
    _M_file->write(header.c_str(), header_size);

    for (std::size_t i = 0; i < _M_metadata.size(); i++) {
        auto tensor_size = _M_metadata[i].data_offsets[1] - _M_metadata[i].data_offsets[0];
        _M_file->write(tensor_data[i], tensor_size);
    }

    _M_file->close();
}


} // namespace metalchat
