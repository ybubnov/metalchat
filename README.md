# MetalChat - Llama inference for Apple Silicon

MetalChat is a [Metal](https://developer.apple.com/metal/)-accelerated C++ framework and command
line interpreter for inference of [Meta Llama](https://www.llama.com/) models.

> [!IMPORTANT]
> The library API and CLI are under active development, therefore they may change without any
> deprecation notice. See issues tab for the list of known issues or missing features.

## Installation

The framework and binary could be installed using Homebrew package manager in a following way
```sh
brew tap ybubnov/metalchat https://github.com/ybubnov/metalchat
brew install --HEAD metalchat
```

Alternatively you could build a [Conan](https://conan.io/) package locally using dependencies
download from the Conan registry. After that, you could use the MetalChat framework just like
any other Conan dependency.
```sh
git clone https://github.com/ybubnov/metalchat
cd metalchat
conan build \
    --build=missing \
    --settings build_type=Release \
    --conf tools.build.skip_test=True \
    --options '&:build_executable'=False \
    --options '&:use_system_libs'=False
conan export-pkg
```

If you are using CMake to as a build system, you could link the framework using an automatically
exported target:
```cmake
find_package(metalchat CONFIG REQUIRED)
target_link_libraries(build_target PRIVATE metalchat::metalchat)
```

## Usage

The library provides a API for low-level tensor manipulation, as well as high-level API
for running a language model inference.

Unlike the general-purpose ML frameworks, MetalChat kernel support is limited and provides
batched operations that are reasonable for LLM inference.
```c++
#include <metalchat/metalchat.h>

int main()
{
    metalchat::hardware_accelerator gpu0;
    metalchat::kernel::cumsum<float> cumsum(gpu0);

    auto input = metalchat::rand<float>({4, 10});
    auto output = cumsum(input).get();
    std::cout << output << std::endl;
}
```

For this expample, you would need to download weights from the HuggingFace
[Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
```c++
#include <metalchat/metalchat.h>

int main()
{
    using Transformer = metalchat::huggingface::llama3;
    using Repository = metalchat::filesystem_repository<Transformer>;

    Repository repo("Llama-3.2-1B-Instruct");
    auto tokenizer = repo.retrieve_tokenizer();
    auto transformer = repo.retreive_transformer();

    metalchat::interpreter chat(transformer, tokenizer);

    chat.send(metalchat::basic_message("system", "You are a helpful assistant"));
    chat.send(metalchat::basic_message("user", "What is the capital of France?"));

    std::cout << chat.receive_text() << std::endl;
    // Prints: The capital of France is Paris.
    return 0;
}
```

## License

The MetalChat is distributed under GPLv3 license. See the [LICENSE](LICENSE) file for full license
text.
