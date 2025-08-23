Large language models
=====================


Meta Llama
----------

.. doxygenclass:: metalchat::nn::llama3
   :members:


Key-value caching
-----------------

In autoregressive language models, key-value tensors power the attention mechanism that determines
how tokens relate to each other. These models generate text one token at a time, and each new
prediction requires attention calculations across all previous tokens in the sequence. Without
optimization, this creates a costly cycle of redundant work. Every time the model predicts the
next token, it recalculates the same key-value tensors for tokens it has already processed.

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
