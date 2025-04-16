#include <catch2/catch_test_macros.hpp>


#include <metalchat/bpe.h>
#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/kernel/sort.h>
#include <metalchat/llama.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;

template <typename T, contiguous_container Container>
std::size_t
first_(shared_tensor<T, 3, Container> t)
{
    auto last = t.size(1) - 1;
    return t[0, last, 0];
}


TEST_CASE("Test make model", "[llama]")
{
    metalchat::bpe bpe("../Llama-3.2-1B/original/tokenizer.model");
    metalchat::device gpu0("metalchat.metallib", 256);
    metalchat::sort<bf16, 1024> sort(gpu0);

    // Load tensors in lambda, so that all resources are cleaned up after the load.
    auto m = [&] -> auto {
        safetensor_file tensors("../llama32.safetensors");
        return make_llama<bf16>(tensors, gpu0);
    }();

    auto input_text = std::string("I have a dog called");

    std::vector<int32_t> ids;
    bpe.encode(special_token::begin_text, ids);
    bpe.encode(input_text, ids);

    auto input0 = shared_tensor(to_tensor<int32_t>({1, ids.size()}, ids.begin(), ids.end()));
    auto [_, indices0] = sort(m(input0, 0));
    auto id = first_(indices0.get());
    std::cout << input_text;
    std::cout << bpe.decode(id);

    for (std::size_t i = input0.size(1); i < 52; i++) {
        auto input = shared_tensor(full<int32_t>({1, 1}, id));
        auto [_, output] = sort(m(input, i));
        id = first_(output.get());
        std::cout << bpe.decode(id) << std::flush;
    }
    std::cout << std::endl;
}
