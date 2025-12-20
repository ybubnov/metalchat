// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <metalchat/accelerator.h>
#include <metalchat/autoloader.h>
#include <metalchat/functional.h>
#include <metalchat/nn.h>
#include <metalchat/tensor.h>

#include "metalchat/testing.h"

using namespace metalchat;


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


TEST_CASE("Test model load", "[safetensor]")
{
    using LLama3 = nn::llama3<bf16>;

    hardware_accelerator gpu0(16);
    nn::indirect_layer<LLama3> m(nn::default_llama3_1b_options(), gpu0);

    auto doc_path = test_fixture_path() / "llama3.2:1b-instruct" / "model.safetensors";
    auto doc_adapter = llama3_reference_traits<bf16>::document_adaptor();
    auto doc = safetensor_document::open(doc_path, gpu0);

    doc = doc_adapter.adapt(doc);
    doc.load(m);
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


TEST_CASE("Test write and read small model", "[safetensor]")
{
    struct model : public nn::basic_layer {
        model(hardware_accelerator& accelerator)
        : nn::basic_layer(accelerator)
        {
            auto w1 = rand<float>({10, 20}, accelerator);
            auto w2 = rand<bf16>({3, 4}, accelerator);

            register_layer<nn::linear<float>>("linear1", std::move(w1));
            register_layer<nn::linear<bf16>>("linear2", std::move(w2));
        }
    };

    scoped_temp_directory tmpdir("safetensor");
    auto model_path = tmpdir.path() / "model.st";

    hardware_accelerator accelerator;

    nn::indirect_layer<model> model_out(accelerator);
    safetensor_document::save(model_path, model_out);

    REQUIRE(std::filesystem::exists(model_path));

    nn::indirect_layer<model> model_in(accelerator);
    safetensor_document::load(model_path, model_in);

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
