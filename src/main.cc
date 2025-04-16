#include <utility>

#include <metalchat/device.h>
#include <metalchat/dtype.h>
#include <metalchat/functional.h>
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

    // {
    //     auto input = metalchat::full<bf16>({128, 2048}, bf16(0.5));
    //     auto up = model_file["model.layers.0.mlp.up_proj.weight"].as<bf16, 2>();
    //     auto down = model_file["model.layers.0.mlp.down_proj.weight"].as<bf16, 2>();
    //     auto gate = model_file["model.layers.0.mlp.gate_proj.weight"].as<bf16, 2>();

    //     metalchat::llama::mlp mlp(gate, up, down, gpu0);
    //     auto result = mlp(input);

    //     std::cout << "result=" << result << std::endl;
    // }

    {
        auto input = metalchat::empty<bf16>({5});
        for (std::size_t i = 0; i < 5; i++) {
            input[i] = bf16(i);
        }

        metalchat::softmax<bf16> softmax(gpu0);

        std::cout << input << std::endl;
        std::cout << softmax(input) << std::endl;
    }

    return 0;
}
