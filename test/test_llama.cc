#include <catch2/catch_test_macros.hpp>


#include <metalchat/bpe.h>
#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/llama.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;

template <typename T, ContiguousContainer Container>
std::size_t
argmax_(shared_tensor<T, 3, Container> t)
{
    std::size_t id = 0;
    auto last = t.size(1) - 1;
    T max = t[0, last, 0];

    for (auto i = 0; i < t.size(2); i++) {
        if (t[0, last, i] > max) {
            id = i;
            max = t[0, last, i];
        }
    }
    return id;
}


TEST_CASE("Test make model", "[llama]")
{
    metalchat::bpe bpe("../Llama-3.2-1B/original/tokenizer.model");
    metalchat::device gpu0("metalchat.metallib", 256);

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
    auto output0 = m(input0, 0);
    auto id = argmax_(output0.get());
    std::cout << input_text;
    std::cout << bpe.decode(id);

    for (auto i = input0.size(1); i < 52; i++) {
        auto input = shared_tensor(full<int32_t>({1, 1}, id));
        auto output = m(input, i);
        id = argmax_(output.get());
        std::cout << bpe.decode(id) << std::flush;
    }
    std::cout << std::endl;
}
