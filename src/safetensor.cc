#include <utility>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/nn.h>
#include <metalchat/safetensor.h>


using namespace metalchat::dtype;


int
main()
{
    metalchat::safetensor_file model_file("../Llama-3.2-1B/model.safetensors");
    metalchat::device gpu0("metalchat.metallib");
    std::cout << "device = " << gpu0.name() << std::endl;

    {
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
        /*
        auto weight = model_file["model.embed_tokens.weight"].as<bf16, 2>();
        std::cout << weight << std::endl;

        auto input = metalchat::full<int32_t>({12}, 1);
        std::cout << input << std::endl;

        metalchat::nn::embedding<bf16> embedding(gpu0);

        auto result = embedding(input, weight);
        std::cout << result << std::endl;
        */
    }

    {
        auto weight = metalchat::full<bf16>({1024}, /*fill_value=*/2.0);
        auto input = metalchat::full<bf16>({1024}, /*fill_value=*/5.0);

        metalchat::nn::rmsnorm<bf16> rmsnorm(gpu0);
        auto result = rmsnorm(input, weight);
        std::cout << result << std::endl;
    }

    std::cout << "----" << std::endl;
    auto t = metalchat::empty<int32_t>({4, 3});
    std::cout << t.t() << std::endl;
    std::cout << "free" << std::endl;

    return 0;
}
