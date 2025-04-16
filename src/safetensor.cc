#include <utility>

#include <metalama/device.h>
#include <metalama/safetensor.h>


int
main()
{
    safetensor_file model_file("../Llama-3.2-1B/model.safetensors");
    //for (auto [name, tensor] : model_file) {
    //    std::cout << tensor << std::endl;
    //    std::cout << name << ": ";

    //    if (tensor.dim() == 1) {
    //        bfloat_tensor1d t = tensor.as<__fp16, 1>();
    //        std::cout << t << std::endl;
    //    }
    //    if (tensor.dim() == 2) {
    //        bfloat_tensor2d t = tensor.as<__fp16, 2>();
    //        std::cout << t << std::endl;
    //    }
    //}

    metalama::device gpu0("metalama.metallib");
    std::cout << "device = " << gpu0.name() << std::endl;
    std::cout << model_file["model.embed_tokens.weight"].as<__fp16, 2>() << std::endl;

    auto input = rand<int32_t, 3>({12, 4, 15});
    std::cout << input << std::endl;

    //metalama::op embedding<int32, __fp16>("embedding");
    //auto result = embedding(input, weight, /*device=*/gpu0);

    return 0;
}
