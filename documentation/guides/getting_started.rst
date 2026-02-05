Getting started
===============

.. warning::

   Work is in progress.

This guide walks through the most basic usage of the library, precisely - how to build a
simple chat application with MetalChat framework.

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

   int main()
   {
       using Transformer = metalchat::huggingface::llama3;
       using Repository = metalchat::filesystem_repository<Transformer>;

       Repository repository("./Llama-3.2-1B-Instruct");
       auto tokenizer = repository.retrieve_tokenizer();
       auto transformer = repository.retreive_transformer();

       metalchat::interpreter chat(transformer, tokenizer);

       chat.send(metalchat::basic_message("system", "You are a helpful assistant"));
       chat.send(metalchat::basic_message("user", "What is the capital of France?"));

       std::cout << chat.receive_text() << std::endl;
       // Prints: The capital of France is Paris.
       return 0;
   }


Building an executable
^^^^^^^^^^^^^^^^^^^^^^

The last step is building the executable to launch a Llama-based chat and receive an answer
to the question. Here we assume that MetalChat framework is located within a working directory,
therefore we set framework lookup path (``-F``) as a current path.

.. code:: console

   $ clang++ -std=c++23 -rpath . -F. -framework MetalChat main.cc -o run
   $ ./run
   The capital of France is Paris.
