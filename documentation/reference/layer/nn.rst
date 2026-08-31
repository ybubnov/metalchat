Neural network layers
=====================

These are the basic building blocks for neural networks. In order to use them, include the header
like in the example below:

.. code-block:: c++

   #include <metalchat/nn.h>

   using namespace metalchat::nn;


Attention
---------

.. doxygenclass:: metalchat::nn::multihead_attention
   :members:


.. doxygenclass:: metalchat::nn::recurrent_attention
   :members:


Embedding
---------

.. doxygenclass:: metalchat::nn::embedding
   :members:


Convolution
-----------

.. doxygenclass:: metalchat::nn::conv1d
   :members:


Rotary positional embedding
---------------------------

.. doxygenclass:: metalchat::nn::rope
   :members:


Linear
------

.. doxygenclass:: metalchat::nn::linear
   :members:


Root mean square normalization
------------------------------

.. doxygenclass:: metalchat::nn::rmsnorm
   :members:


Transformer
-----------

.. doxygenclass:: metalchat::nn::transformer
   :members:
