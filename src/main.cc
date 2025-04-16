#include <utility>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/llama.h>
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
                auto t = tensor.as<bf16, 1>();
                std::cout << t << std::endl;
            }
            if (tensor.dim() == 2) {
                auto t = tensor.as<bf16, 2>();
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

    //{
    //    auto weight = metalchat::full<bf16>({1024}, /*fill_value=*/2.0);
    //    auto input = metalchat::full<bf16>({1024}, /*fill_value=*/5.0);

    //    metalchat::nn::rmsnorm<bf16> rmsnorm(gpu0);
    //    auto result = rmsnorm(input, weight);
    //    std::cout << result << std::endl;
    //}

    {
        auto input = metalchat::full<bf16>({128, 2048}, 2.0);
        auto up = model_file["model.layers.0.mlp.up_proj.weight"].as<bf16, 2>();
        auto down = model_file["model.layers.0.mlp.down_proj.weight"].as<bf16, 2>();
        auto gate = model_file["model.layers.0.mlp.gate_proj.weight"].as<bf16, 2>();

        std::cout << input << std::endl;
        std::cout << gate << std::endl;
        std::cout << up << std::endl;
        std::cout << down << std::endl;

        metalchat::llama::mlp mlp(gate, up, down, gpu0);
        auto result = mlp(input);

        std::cout << "result=" << result << std::endl;
    }


    return 0;
}
