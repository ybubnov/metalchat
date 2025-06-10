#include <simdjson.h>

#include <metalchat/safetensor.h>


using namespace simdjson;


struct safetensor_ptr {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> data_offsets;
};


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
simdjson_inline simdjson_result<safetensor_ptr>
simdjson::ondemand::value::get() noexcept
{
    ondemand::object object;
    auto error = get_object().get(object);
    if (error) {
        return error;
    }

    safetensor_ptr ptr;
    if ((error = object["dtype"].get_string(ptr.dtype))) {
        return error;
    }
    if ((error = object["shape"].get<std::vector<std::size_t>>().get(ptr.shape))) {
        return error;
    }
    if ((error = object["data_offsets"].get<std::vector<std::size_t>>().get(ptr.data_offsets))) {
        return error;
    }

    return ptr;
}


namespace metalchat {


void
safetensor_file::parse()
{
    // Read the length of the header and then the header itself, ensure that the
    // the file contains enough data to avoid reading from inaccessible regions.
    uint64_t header_size = 0;
    _m_memfile->read(&header_size, sizeof(header_size));

    char header[header_size];
    _m_memfile->read(&header, sizeof(header));

    simdjson::ondemand::parser json_parser;
    simdjson::padded_string header_padded(header, header_size);

    auto json_document = json_parser.iterate(header_padded);
    auto json_object = json_document.get_object();

    for (auto json_field : json_object) {
        std::string_view field_name = json_field.unescaped_key();
        if (field_name == "__metadata__") {
            continue;
        }

        ::safetensor_ptr tensor_ptr;
        auto error = json_field.value().get<::safetensor_ptr>().get(tensor_ptr);
        if (error) {
            throw std::runtime_error(simdjson::error_message(error));
        }

        auto tensor_name = std::string(field_name);

        // Data pointer is owned by the memory file, therefore a data pointer should
        // also carry a pointer. If we won't do this, once `safetensor_file` destroyed
        // (for example, if it's created on stack), then all safetensors will be invalidated.
        auto data = _m_memfile->tellp() + tensor_ptr.data_offsets[0];
        auto data_ptr = std::shared_ptr<void>(_m_memfile, data);

        auto tensor = safetensor(tensor_ptr.shape, data_ptr);

        _m_tensors.insert_or_assign(tensor_name, tensor);
    }
}


} // namespace metalchat
