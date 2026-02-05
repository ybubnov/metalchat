Kernel library
==============

.. toctree::
   :maxdepth: 1

   metal

This library provides low-level operations accelerated with Metal framework. This
library could be used like following:

.. code-block:: c++

   #include <metalchat/kernel.h>

Hardware accelerator
--------------------

.. doxygenclass:: metalchat::hardware_accelerator
   :members:


Basic kernel
------------

.. doxygenclass:: metalchat::basic_kernel
   :members:


Binary kernel wrapper
---------------------

.. doxygenclass:: metalchat::binary_kernel_wrapper
   :members:


Kernel task
-----------

.. doxygenstruct:: metalchat::dim3
   :members:

.. doxygenclass:: metalchat::kernel_task
   :members:


Kernel thread
-------------

.. doxygenclass:: metalchat::kernel_thread
   :members:


Recursive kernel thread
-----------------------

.. doxygenclass:: metalchat::recursive_kernel_thread
   :members:
