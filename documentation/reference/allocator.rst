.. _allocator_library:


Allocator library
=================

Allocator concepts
------------------

.. doxygenconcept:: metalchat::allocator

.. doxygenconcept:: metalchat::allocator_t

.. doxygenconcept:: metalchat::hardware_allocator

.. doxygenconcept:: metalchat::hardware_allocator_t


Hardware memory allocator
-------------------------

.. doxygenclass:: metalchat::hardware_memory_allocator
   :members:


Hardware buffer allocator
-------------------------

.. doxygenclass:: metalchat::hardware_buffer_allocator
   :members:


Hardware heap allocator
-----------------------

.. doxygenclass:: metalchat::hardware_heap_allocator
   :members:


Hardware resident allocator
---------------------------

.. doxygenclass:: metalchat::hardware_resident_allocator
   :members:


Hardware no-copy allocator
--------------------------

.. doxygenclass:: metalchat::hardware_nocopy_allocator
   :members:


Hardware aliasing allocator
---------------------------

.. doxygenclass:: metalchat::hardware_aliasing_allocator
   :members:


Rebind hardware allocator
----------------------------

.. doxygenclass:: metalchat::rebind_hardware_allocator
   :members:


Polymorphic hardware allocator
------------------------------

.. doxygenstruct:: metalchat::basic_hardware_memory_allocator
   :members:

.. doxygenclass:: metalchat::polymorphic_hardware_memory_allocator
   :members:


Scalar memory allocator
-----------------------

.. doxygenstruct:: metalchat::scalar_memory_allocator
   :members:


Random memory allocator
-----------------------

.. doxygenstruct:: metalchat::random_memory_allocator
   :members:


Paginated allocator adapter
---------------------------

.. doxygenclass:: metalchat::paginated_allocator_adapter
   :members:
