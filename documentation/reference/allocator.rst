.. _allocator_library:


Allocator library
=================

This library provides primitives to efficiently manage dynamic allocation of memory containers
used for scheduling accelerated kernel execution. Use this library as following:

.. code-block:: c++

   #include <metalchat/allocator.h>


Hardware memory allocator
-------------------------

.. doxygenclass:: metalchat::hardware_memory_allocator
   :members:


Hardware heap allocator
-----------------------

.. doxygenclass:: metalchat::hardware_heap_allocator
   :members:


Hardware resident allocator
---------------------------

.. doxygenclass:: metalchat::hardware_resident_allocator
   :members:


Polymorphic hardware allocator
------------------------------

.. doxygenstruct:: metalchat::basic_hardware_allocator
   :members:

.. doxygenclass:: metalchat::polymorphic_hardware_allocator
   :members:


Scalar memory allocator
-----------------------

.. doxygenstruct:: metalchat::scalar_memory_allocator
   :members:


Random memory allocator
-----------------------

.. doxygenstruct:: metalchat::random_memory_allocator
   :members:


Aliasing allocator
------------------

.. doxygenclass:: metalchat::aliasing_allocator
   :members:


Rebind allocator
----------------

.. doxygenstruct:: metalchat::rebind_allocator
   :members:


Pooling allocator adapter
-------------------------

.. doxygenclass:: metalchat::pooling_allocator_adapter
   :members:


No-copy allocator
-----------------

.. doxygenclass:: metalchat::nocopy_allocator
   :members:


File-buffered memory allocator
------------------------------

.. doxygenstruct:: metalchat::filebuf_memory_allocator
   :members:


Paginated allocator adapter
---------------------------

.. doxygenclass:: metalchat::paginated_allocator_adapter
   :members:


Allocator concepts
------------------

.. doxygenconcept:: metalchat::allocator

.. doxygenconcept:: metalchat::allocator_t

.. doxygenconcept:: metalchat::hardware_allocator

.. doxygenconcept:: metalchat::hardware_allocator_t
