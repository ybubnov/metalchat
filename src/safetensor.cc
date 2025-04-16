#include <utility>

#include "safetensor.h"


int
main()
{
    safetensor_file model_file("../Llama-3.2-1B/model.safetensors");
    for (auto tensor : model_file) {
        std::cout << tensor << std::endl;
    }
    return 0;
}
