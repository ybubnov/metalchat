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
    auto input = bpe.encode("Capital of Germany is ").reshape({1, -1});

    auto output = m(input);
    std::cout << output << std::endl;

    REQUIRE(output.dim() == 3);

    std::size_t id = 0;
    bf16 max = output[0, input.size(1) - 1, 0];

    for (auto i = 0; i < output.size(2); i++) {
        if (output[0, input.size(1) - 1, i] > max) {
            id = i;
            max = output[0, input.size(1) - 1, i];
        }
    }

    std::cout << bpe.decode(id) << std::endl;
}
