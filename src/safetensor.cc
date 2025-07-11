#include <simdjson.h>

#include <metalchat/safetensor.h>


using namespace simdjson;


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


safetensor_document::safetensor_document(std::shared_ptr<basic_memfile> file)
: _m_file(file),
  _m_metadata(load_header(file))
{}


void*
safetensor_document::data() noexcept
{
    std::size_t offset = _m_metadata.empty() ? 0 : _m_metadata.front().data_offsets[0];
    return _m_file->data() + offset;
}


std::vector<std::size_t>
safetensor_document::sizes() const
{
    std::vector<std::size_t> result;
    for (const auto& metadata : _m_metadata) {
        result.push_back(metadata.data_offsets[1] - metadata.data_offsets[0]);
    }
    return result;
}


std::vector<safetensor_metadata>
safetensor_document::load_header(basic_memfile& file)
{
    // Read the length of the header and then the header itself, ensure that the
    // the file contains enough data to avoid reading from inaccessible regions.
    uint64_t header_size = 0;
    file.read(&header_size, sizeof(header_size));

    char header[header_size];
    file.read(&header, sizeof(header));

    simdjson::ondemand::parser json_parser;
    simdjson::padded_string header_padded(header, header_size);

    auto json_document = json_parser.iterate(header_padded);
    auto json_object = json_document.get_object();

    std::vector<safetensor_metadata> metadata;
    auto start_pos = file.tellg();

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

        for (auto& offset : tensor_metadata.data_offsets) {
            offset += start_pos;
        }

        tensor_metadata.name = field_name;
        metadata.push_back(tensor_metadata);
    }

    // Order metadata entries to ensure that we access file sequentially.
    auto metadata_comp = [](const safetensor_metadata& a, const safetensor_metadata& b) {
        return a.data_offsets[0] < b.data_offsets[0];
    };
    std::sort(metadata.begin(), metadata.end(), metadata_comp);

    return metadata;
}


std::vector<safetensor_metadata>
safetensor_document::load_header(std::shared_ptr<basic_memfile> file_ptr)
{
    return load_header(*file_ptr);
}


} // namespace metalchat
