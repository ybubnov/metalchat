Tensor library
==============


Tensor
------

.. doxygenclass:: metalchat::tensor
   :project: metalchat


Tensor Iterator
---------------

.. doxygenclass:: metalchat::tensor_iterator
   :project: metalchat


Shared Tensor
-------------

.. doxygenclass:: metalchat::shared_tensor
   :project: metalchat


Future Tensor
-------------

.. doxygenclass:: metalchat::future_tensor
   :project: metalchat
   :members:


Scalar
------

.. doxygenfunction:: metalchat::scalar
   :project: metalchat


Empty
------

.. doxygenfunction:: metalchat::empty(std::size_t (&&sizes)[N])
   :project: metalchat

.. doxygenfunction:: metalchat::empty(std::size_t (&&sizes)[N], hardware_accelerator& gpu)
   :project: metalchat

.. doxygenfunction:: metalchat::empty(std::size_t (&&sizes)[N], Allocator alloc)
   :project: metalchat
