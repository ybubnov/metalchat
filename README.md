# MetalChat - Llama inference for Apple Silicon

MetalChat is a [Metal](https://developer.apple.com/metal/)-accelerated C++ framework and command
line interpreter for inference of [Meta Llama](https://www.llama.com/) models.

> [!IMPORTANT]
> The library API and CLI are under active development, therefore they may change without any
> deprecation notice. See issues tab for the list of known issues or missing features.


## Getting started

See the [getting started](https://metalchat.readthedocs.org/en/latest/guides/getting_started.html)
guide for using MetalChat as a library and
[command line](https://metalchat.readthedocs.org/en/latest/guides/command_line.html) guide for
using MetalChat to interact with LLM model from the command line.

## Installation

The framework and command line utility could be installed using Homebrew package manager in a
following way:
```sh
brew tap ybubnov/metalchat https://github.com/ybubnov/metalchat
brew install --HEAD metalchat
```

Alternatively you could build a [Conan](https://conan.io/) package locally using dependencies
downloaded from the Conan registry. After that, you could use the MetalChat framework just like
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

If you are using CMake as a build system, you could link the framework using an automatically
exported target:
```cmake
find_package(metalchat CONFIG REQUIRED)
target_link_libraries(build_target PRIVATE MetalChat::MetalChat)
```

## License

The MetalChat is distributed under GPLv3 license. See the [LICENSE](LICENSE) file for full license
text.
