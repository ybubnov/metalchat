Metal accelerated kernels
=========================


Arithmetic kernels
------------------

.. doxygenclass:: metalchat::kernel::add
   :members:

.. doxygenclass:: metalchat::kernel::add2
   :members:

.. doxygenclass:: metalchat::kernel::sub
   :members:

.. doxygenclass:: metalchat::kernel::cumsum
   :members:

.. doxygenclass:: metalchat::kernel::hadamard
   :members:

.. doxygenclass:: metalchat::kernel::scalar_mul
   :members:


Comparison kernels
------------------

.. doxygenclass:: metalchat::kernel::gt
   :members:

.. doxygenclass:: metalchat::kernel::sort
   :members:


Batched matrix multiplication
-----------------------------

.. doxygenclass:: metalchat::kernel::bmm
   :members:


Copying kernels
---------------

.. doxygenclass:: metalchat::kernel::clone
   :members:

.. doxygenclass:: metalchat::kernel::gather
   :members:

.. doxygenclass:: metalchat::kernel::roll
   :members:

.. doxygenclass:: metalchat::kernel::scatter
   :members:


Sparse kernels
--------------

.. doxygenclass:: metalchat::kernel::embedding
   :members:

.. doxygenclass:: metalchat::kernel::rope
   :members:

.. doxygenclass:: metalchat::kernel::rope_freqs
   :members:


Non-linear activation kernels
-----------------------------

.. doxygenclass:: metalchat::kernel::rmsnorm
   :members:

.. doxygenclass:: metalchat::kernel::silu
   :members:

.. doxygenclass:: metalchat::kernel::softmax
   :members:
