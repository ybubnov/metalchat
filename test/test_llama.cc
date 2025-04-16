#include <catch2/catch_test_macros.hpp>


#include <metalchat/bpe.h>
#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/llama.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Test make model", "[llama]")
{
    safetensor_file tensors("../Llama-3.2-1B/model.safetensors");
    metalchat::bpe bpe("../Llama-3.2-1B/original/tokenizer.model");
    metalchat::device gpu0("metalchat.metallib");

    auto m = make_llama<bf16>(tensors, gpu0);

    std::vector<int32_t> ids;
    bpe.encode(special_token::begin_text, ids);
    bpe.encode("The capital of Germany is ", ids);

    auto input = to_tensor<int32_t>({1, ids.size()}, ids.begin(), ids.end());

    auto output = m(input);
    std::cout << output << std::endl;

    REQUIRE(output.dim() == 3);

    std::size_t id = 0;
    auto last = input.size(1) - 1;
    bf16 max = output[0, last, 0];

    for (auto i = 0; i < output.size(2); i++) {
        if (output[0, last, i] > max) {
            id = i;
            max = output[0, last, i];
        }
    }

    std::cout << bpe.decode(id) << std::endl;
}
