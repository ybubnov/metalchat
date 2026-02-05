Tensor library
==============

.. toctree::
   :maxdepth: 1

   create
   future
   traits

This is the fundamental library of MetalChat. It provides a convenient interface for management
of multi-dimensional matrices. This library could be used like following:

.. code-block:: c++

   #include <metalchat/tensor.h>

Tensor
------

.. doxygenclass:: metalchat::tensor
   :members:


Tensor layout
-------------

.. doxygenstruct:: metalchat::tensor_layout
   :members:


Tensor iterator
---------------

.. doxygenclass:: metalchat::tensor_iterator
   :members:


Basic tensor
------------

.. doxygenclass:: metalchat::basic_tensor
   :members:


Shared tensor
-------------

.. doxygenclass:: metalchat::shared_tensor_ptr
   :members:
