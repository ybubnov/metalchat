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

    auto input = bpe.encode("Sunday is the last day of ").reshape({1, -1});

    auto output = m(input);
    std::cout << output << std::endl;

    REQUIRE(output.dim() == 2);

    std::size_t id = 0;
    bf16 max;
    for (auto i = 0; i < output.size(1); i++) {
        if (output[input.size(0) - 1, i] > max) {
            id = i;
            max = output[input.size(0) - 1, i];
        }
    }

    std::cout << bpe.decode(id) << std::endl;
}
