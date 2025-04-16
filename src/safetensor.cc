#include <utility>

#include <metalama/safetensor.h>


int
main()
{
    safetensor_file model_file("../Llama-3.2-1B/model.safetensors");
    for (auto [name, tensor] : model_file) {
        std::cout << tensor << std::endl;
        // std::cout << "  strides=[" << tensor.strides << "]" << std::endl;

        if (tensor.dim() == 1) {
            bfloat_tensor1d t = tensor.as<__fp16, 1>();
            std::cout << t << std::endl;
        }
        if (tensor.dim() == 2) {
            bfloat_tensor2d t = tensor.as<__fp16, 2>();
            std::cout << t << std::endl;
        }
    }
    return 0;
}
