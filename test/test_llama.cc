#include <catch2/catch_test_macros.hpp>


#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/llama.h>
#include <metalchat/tensor.h>


using namespace metalchat;
using namespace metalchat::dtype;


TEST_CASE("Test make model", "[llama]")
{
    safetensor_file tensors("../Llama-3.2-1B/model.safetensors");
    metalchat::device gpu0("metalchat.metallib");

    auto m = make_llama<bf16>(tensors, gpu0);

    std::vector<int32_t> tokens({2028, 374, 264, 1296});
    auto input = to_tensor<int32_t>({1, 4}, tokens.cbegin(), tokens.cend());

    auto output = m(input);
    std::cout << output << std::endl;
}
