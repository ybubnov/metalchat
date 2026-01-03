Language models
===============

These are the building blocks of language models. In order to use them, include the header like
in the example below:

.. code-block:: c++

   #include <metalchat/nn.h>

   using namespace metalchat::nn;


Meta Llama 3
------------

.. doxygenclass:: metalchat::nn::llama3
   :members:


Key-value caching
-----------------

In autoregressive language models, key-value tensors power the attention mechanism that determines
how tokens relate to each other. These models generate text one token at a time, and each new
prediction requires attention calculations across all previous tokens in the sequence. Without
optimization, this creates a costly cycle of redundant work. Every time the model predicts the
next token, it recalculates the same key-value tensors for tokens it has already processed.

The cache is a layer and is registered as a layer as well within a language model, therefore it
is possible to access cache tensors through regular :cpp:class:`metalchat::basic_layer` api.
For example, to access cache for the 2-nd layer use the following approach:

.. code-block:: c++

   using namespace metalchat;
   using namespace metalchat::dtype;

   hardware_accelerator accelerator;
   nn::llama3<bf16> llm(default_llama3_1b_options(), accelerator);

   std::cout << llm.get_parameter("caches.2.keys")->sizes() << std::endl;:
   std::cout << llm.get_parameter("caches.2.values")->sizes() << std::endl;
   // out:
   // 1, 1024, 8, 64
   // 1, 1024, 8, 64


.. doxygenstruct:: metalchat::nn::caching_options
   :members:


.. doxygenstruct:: metalchat::nn::caching_result
   :members:

.. doxygenclass:: metalchat::nn::sink_cache
   :members:


Sampling
--------

.. doxygenclass:: metalchat::nn::basic_sampler
   :members:


.. doxygenclass:: metalchat::nn::nucleus_sampler
   :members:
