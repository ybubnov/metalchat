#include <utility>

#include <metalama/device.h>
#include <metalama/safetensor.h>


int
main()
{
    {
        metalama::safetensor_file model_file("../Llama-3.2-1B/model.safetensors");
        /*
        for (auto [name, tensor] : model_file) {
            std::cout << tensor << std::endl;
            std::cout << name << ": ";

            if (tensor.dim() == 1) {
                auto t = tensor.as<__fp16, 1>();
                std::cout << t << std::endl;
            }
            if (tensor.dim() == 2) {
                auto t = tensor.as<__fp16, 2>();
                std::cout << t << std::endl;
            }
        }
        */

        metalama::device gpu0("metalama.metallib");
        std::cout << "device = " << gpu0.name() << std::endl;

        auto weight = model_file["model.embed_tokens.weight"].as<__fp16, 2>();
        auto input = metalama::full<int32_t>(/*fill_value=*/1, 12);
        std::cout << input << std::endl;
        std::cout << weight << std::endl;

        metalama::embedding embedding("embedding_bf16", /*device=*/gpu0);
        auto result = embedding(input, weight);
        std::cout << result << std::endl;
    }

    std::cout << "free" << std::endl;
    std::cout << metalama::empty<int32_t>(1, 2, 3) << std::endl;

    return 0;
}
