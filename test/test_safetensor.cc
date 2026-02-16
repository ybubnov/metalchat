// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <jsoncons/json.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/functional.h>
#include <metalchat/nn.h>
#include <metalchat/reference.h>
#include <metalchat/safetensor.h>
#include <metalchat/tensor.h>

#include "metalchat/testing.h"

using namespace metalchat;

JSONCONS_ALL_MEMBER_TRAITS(safetensor_index, metadata, weight_map);


struct scoped_temp_directory {
private:
    std::filesystem::path _M_name;

    static std::string
    random_string(std::size_t n, std::size_t alphabet_size = 16)
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_int_distribution<std::size_t> distribution(0, alphabet_size);

        std::string result(n, '0');
        std::generate_n(result.begin(), n, [&]() { return distribution(generator) + 'a'; });

        return result;
    }

public:
    scoped_temp_directory(std::string prefix)
    : _M_name(std::filesystem::temp_directory_path() / prefix / random_string(16))
    {
        std::filesystem::create_directories(_M_name);
    }

    std::filesystem::path
    path() const
    {
        return _M_name;
    }

    ~scoped_temp_directory()
    {
        if (_M_name != "" && _M_name != "/") {
            REQUIRE(std::filesystem::remove_all(_M_name) > 0);
        }
    }
};


TEST_CASE("Test model load", "[safetensor][integration]")
{
    using layer_type = nn::llama3<bf16>;
    using serializer_type = reference::llama3_safetensor_serializer<layer_type>;
    auto options = nn::default_llama3_1b_options();

    hardware_accelerator gpu0(16);

    auto repo_path = test_fixture_path() / "meta-llama/Llama-3.2-1B-Instruct/original";
    auto doc_path = repo_path / "model.safetensors";
    auto doc = safetensor_document::open(doc_path, gpu0);

    auto serializer = serializer_type(options, gpu0);
    auto m = serializer.load(doc);
    auto params = m.get_parameters();

    REQUIRE(params.size() == 179);
    for (auto [name, param] : params) {
        REQUIRE(param->numel() > 0);
        REQUIRE(param->container_ptr() != nullptr);
    }
}


namespace std {
template <> struct __libcpp_random_is_valid_realtype<bf16> : true_type {};
}; // namespace std


template <typename T> using linear_layer = nn::linear<T, random_memory_container<T>>;


TEST_CASE("Test write and read small model", "[safetensor]")
{
    struct model : public nn::basic_layer {
        model(hardware_accelerator& accelerator)
        : nn::basic_layer(accelerator)
        {
            auto w1 = rand<float>({10, 20});
            auto w2 = rand<bf16>({3, 4});

            register_layer<linear_layer<float>>("linear1", std::move(w1));
            register_layer<linear_layer<bf16>>("linear2", std::move(w2));
        }
    };

    scoped_temp_directory tmpdir("safetensor");
    auto model_path = tmpdir.path() / "model.st";

    hardware_accelerator accelerator;

    nn::indirect_layer<model> model_out(accelerator);
    safetensor_document::save(model_path, model_out);

    REQUIRE(std::filesystem::exists(model_path));

    nn::indirect_layer<model> model_in(accelerator);

    // Use a method that allocates safetensors using a random memory allocator
    // since the hardware-supported version requires resident allocator, which
    // is not available in the GitHub CI.
    auto doc = safetensor_document::open(model_path);
    doc.load(model_in);

    // Ensure that model parameter's data is the same.
    auto l1_out = model_out.get_parameter("linear1.weight");
    auto l1_in = model_in.get_parameter("linear1.weight");
    auto l1_out_begin = static_cast<float*>(l1_out->data());
    auto l1_in_begin = static_cast<float*>(l1_in->data());

    auto l1_out_vec = std::vector(l1_out_begin, l1_out_begin + l1_out->numel());
    auto l1_in_vec = std::vector(l1_in_begin, l1_in_begin + l1_in->numel());

    using Catch::Matchers::Approx;
    REQUIRE_THAT(l1_out_vec, Approx(l1_in_vec));

    auto l2_out = model_out.get_parameter("linear2.weight");
    auto l2_in = model_in.get_parameter("linear2.weight");
    auto l2_out_begin = static_cast<bf16*>(l2_out->data());
    auto l2_in_begin = static_cast<bf16*>(l2_in->data());

    auto l2_out_vec = std::vector(l2_out_begin, l2_out_begin + l2_out->numel());
    auto l2_in_vec = std::vector(l2_in_begin, l2_in_begin + l2_in->numel());

    REQUIRE_THAT(l2_out_vec, Approx(l2_in_vec));
}


TEST_CASE("Test tensor link", "[safetensor]")
{
    auto input = rand<float>({3, 4});

    safetensor_document doc;
    doc.insert("input.weight", input);
    doc.insert("output.weight", "input.weight");

    auto output = tensor<float, 2>();
    doc.load("output.weight", output);

    REQUIRE(output.size(0) == input.size(0));
    REQUIRE(output.size(1) == input.size(1));
    REQUIRE(output.container_ptr() == input.container_ptr());
}


TEST_CASE("Test sharded document", "[safetensor]")
{
    scoped_temp_directory tmpdir("sharded_safetensor");
    auto index_path = tmpdir.path() / "tensors.safetensors.index.json";
    auto path1 = tmpdir.path() / "tensors-0001-of-0002.safetensors";
    auto path2 = tmpdir.path() / "tensors-0002-of-0002.safetensors";

    auto tensor1 = rand<float>({4, 3});
    auto tensor2 = rand<float>({10, 6});

    safetensor_document doc1;
    safetensor_document doc2;

    doc1.insert("tensor1", tensor1);
    doc1.save(path1);
    doc2.insert("tensor2", tensor2);
    doc2.save(path2);

    safetensor_index index{
        .metadata = {},
        .weight_map = {{"tensor1", path1.string()}, {"tensor2", path2.string()}}
    };

    std::ofstream index_file(index_path);
    jsoncons::encode_json<safetensor_index>(index, index_file);
    index_file.close();

    auto doc = sharded_safetensor_document::open(index_path);
    auto size = std::distance(doc.begin(), doc.end());

    REQUIRE(size == 2);
}
