Getting started
===============

This guide walks through the most basic usage of the library, precisely - how to build a
simple chat application with MetalChat framework.

Building a conan package
^^^^^^^^^^^^^^^^^^^^^^^^

The library is being distributed as a Conan package (unfortunately, not yet available in the
Conan center). So the easiest way is to compile the Conan package locally:

.. code:: console

   $ git clone https://github.com/ybubnov/metalchat
   $ cd metalchat
   $ conan build \
        --build=missing \
        --settings build_type=Release \
        --conf tools.build.skip_testTrue \
        --options '&:build_executable'=False \
        --options '&:use_system_libs'=False

After the compilation of the framework, you could export a Conan package, so it's available
for the usage.

.. code:: console

   $ conan export-pkg


Downloading a model
^^^^^^^^^^^^^^^^^^^

In our application we will show how programmatically to send questions to a Llama 3.2 chat. For
this purpose you will need a copy of the model weights and model vocabulary. Both of those could
by downloaded from the
`HuggingFace repository <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`_. For the
purpose of this guide we would need *Instruct* type of the model, which is trained to answer
questions in chat format.

Writing an application
^^^^^^^^^^^^^^^^^^^^^^

The next step is to write a C++ application. In the example below we use a performance-optimized
implementation of the Llama 3.2 inference with default sampling.

.. code-block:: c++
   :caption: main.cc

   #include <metalchat/metalchat.h>

   int main(int argc, char** argv)
   {
       using Transformer = metalchat::huggingface::llama3;
       using Repository = metalchat::filesystem_repository<Transformer>;

       Repository repository(argv[1]);
       auto tokenizer = repository.retrieve_tokenizer();
       auto transformer = repository.retrieve_transformer();

       metalchat::interpreter chat(transformer, tokenizer);

       chat.write(metalchat::basic_message("system", "You are a helpful assistant"));
       chat.write(metalchat::basic_message("user", "What is the capital of France?"));

       std::cout << chat.read_text() << std::endl;
       // Prints: The capital of France is Paris.
       return 0;
   }


Building an executable
^^^^^^^^^^^^^^^^^^^^^^

The last step is building the executable to launch a Llama-based chat. For this purpose we are
going to use Conan and CMake. At first, you would need to define requirements in the Conan file:

.. code-block:: ini
   :caption: conanfile.txt

   [requires]
   metalchat/[>=1.0.0]

   [tool_requires]
   cmake/[>=3.31.0]
   ninja/[>=1.30.0]

   [generators]
   CMakeDeps
   CMakeToolchain

Then create a ``CMakeLists.txt`` file with the project configuration:

.. code-block:: cmake
   :caption: CMakeLists.txt

   cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

   project(chat LANGUAGES CXX)
   set(CMAKE_CXX_STANDARD 23)

   find_package(metalchat CONFIG REQUIRED)

   add_executable(${PROJECT_NAME} main.cc)
   target_link_libraries(${PROJECT_NAME} PRIVATE MetalChat::MetalChat)
   target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)

Then, you could build a binary using ``conan`` command, and run the executable. If you downloaded
the model into the local directory ``Llama-3.2-1B-Instruct``, then the sequence of compilation
commands are like following:

.. code:: console

   $ conan install . --output-folder=build
   $ cd build
   $ cmake -GNinja .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
   $ cmake --build .

Then run the compiled binary by specifying a path to the downloaded location of the model:

.. code:: console

   $ ./chat '../Llama-3.2-1B-Instruct'
   The capital of France is Paris.
