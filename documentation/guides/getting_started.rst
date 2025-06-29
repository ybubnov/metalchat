Getting Started
===============

In this guide will walk through the most basic usage of the library, precisely - how to build a
simple chat application with `metalchat` library.

Prerequisites to walk through this guide is like in the table below:

.. list-table:: Minimum Requirements
   :widths: 35 65

   * - Operating System
     - MacOS 15
   * - CPU
     - Apple M1
   * - Metal Framework Version
     - 3.2

Downloading Model
^^^^^^^^^^^^^^^^^

In our application we will show how programmatically to send questions to a Llama 3.2 chat. For
this purpose you will need a copy of the model weights and model vocabulary in the
`tiktoken <https://github.com/openai/tiktoken>`_ format. Both of those could downloaded from the
official `Llama website <https://www.llama.com/llama-downloads/>`_. For the purpose of this guide
we would need *Instruct* type of the model, which is trained to answer questions in chat format.

Meta distributes Llama weights in PyTorch tensor format, which is not supported by `metalchat`.
In order to use Llama weights, we need to convert them to
`safetensor <https://huggingface.co/docs/safetensors/index>`_ format.

Here is an example Python script showing how it code be done:

.. code-block:: python
   :caption: convert.py
   :linenos:
   :emphasize-lines: 6

   import torch
   import sys
   from safetensors.torch import save_file

   tensors = torch.load(sys.argv[1], "cpu", weights_only=True)
   tensors["output.weight"] = tensors["output.weight"].clone()
   save_file(tensors, "instruct.safetensors")


.. code-block:: bash

   % python convert.py consolidated.00.pth


Please, note the highlighted line that instructs that output weight should be copied. This tensor
for performance reasons is a reference to the embedding tensor of the same model. Safetensors do
not support references due to memory safety reasons, so in order to make this model work with
`metalchat`, we need to create a materialized copy of that tensor.

Writing Application
^^^^^^^^^^^^^^^^^^^

The next step is to write a C++ application. In the example below we use method
`construct_llama3_1b` that uses various optimizations to load model, and uses default Llama 3.2
configuration.

At the highlighted line in the code snipped below we use model weights converted in the previous
section.

.. code-block:: c++
   :caption: main.cc
   :linenos:
   :emphasize-lines: 5

   #include <metalchat/metalchat.h>

   int main()
   {
       std::filesystem::path weights_path("instruct.safetensors");
       std::filesystem::path tokens_path("tokenizer.model");

       auto chat = metalchat::construct_llama3_1b(weights_path, tokens_path);

       chat.send(metalchat::basic_message("system", "You are a helpful assistant"));
       chat.send(metalchat::basic_message("user", "What is the capital of France?"));

       std::cout << chat.receive_text() << std::endl;
       // Prints: The capital of France is Paris.
       return 0;
   }


Building Executable
^^^^^^^^^^^^^^^^^^^

The last step is building the executable to launch a Llama-based chat and receive an answer
to the question. Here we assume that `MetalChat` framework is located within a working directory,
therefore we set framework lookup path (`-F`) as a current path.

.. code-block:: bash

   % clang++ -std=c++23 -rpath . -F. -framework MetalChat main.cc -o chat
   % ./chat
   The capital of France is Paris.
