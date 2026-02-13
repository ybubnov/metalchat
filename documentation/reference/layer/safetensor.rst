Safetensors format
==================

These are methods to initialize neural networks from safetensors files. In order to use them,
include the hear like in the example below:

.. code-block:: c++

   #include <metalchat/safetensor.h>


Safetensor document
^^^^^^^^^^^^^^^^^^^

.. seealso::

   For more details on the format of the safetensors and other implementations refer
   to the `huggingface page <https://huggingface.co/docs/safetensors/index>`_.

.. doxygenclass:: metalchat::safetensor_document
   :members:


Sharded safetensor document
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: metalchat::sharded_safetensor_document
   :members:


Safetensor
^^^^^^^^^^

.. doxygenclass:: metalchat::safetensor
   :members:


Safetensor allocator
^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: metalchat::safetensor_allocator
   :members:
