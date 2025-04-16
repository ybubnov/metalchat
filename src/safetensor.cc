#include <utility>

#include <metalchat/device.h>
#include <metalchat/nn.h>
#include <metalchat/safetensor.h>


int
main()
{
    {
        metalchat::safetensor_file model_file("../Llama-3.2-1B/model.safetensors");
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
        metalchat::device gpu0("metalchat.metallib");
        std::cout << "device = " << gpu0.name() << std::endl;

        auto weight = model_file["model.embed_tokens.weight"].as<__fp16, 2>();
        std::cout << weight << std::endl;

        auto input = metalchat::full<int32_t>({12}, /*fill_value=*/1);
        std::cout << input << std::endl;

        metalchat::nn::embedding embedding("embedding_f16", gpu0);

        auto result = embedding(input, weight);
        std::cout << result << std::endl;

        metalchat::sum sum("sum_f16", gpu0);
        auto s = metalchat::full<__fp16>({10}, /*fill_value=*/3);
        std::cout << "sum = " << sum(s) << std::endl;
    }

    auto t = metalchat::empty<int32_t>({1, 2, 3});
    std::cout << t << std::endl;
    std::cout << "free" << std::endl;

    return 0;
}
