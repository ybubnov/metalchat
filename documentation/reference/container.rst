Container library
=================

This library provides basic building blocks to manage memory backing tensor locations. This
library could be used like following:

.. code-block:: c++

   #include <metalchat/container.h>


Memory container
----------------

.. doxygenstruct:: metalchat::memory_container
 :members:


Hardware memory container
-------------------------

.. doxygenstruct:: metalchat::hardware_memory_container
   :members:


Random memory container
-----------------------

.. doxygenstruct:: metalchat::random_memory_container
   :members:


Vector memory container
-----------------------

.. doxygenstruct:: metalchat::vector_memory_container
   :members:


Scalar memory container
-----------------------

.. doxygenstruct:: metalchat::scalar_memory_container
   :members:


File buffer container
---------------------

.. doxygenstruct:: metalchat::filebuf_memory_container
   :members:


Offsetted container adapter
---------------------------

.. doxygenclass:: metalchat::offsetted_container_adapter
   :members:


Container concepts
------------------

.. doxygenconcept:: metalchat::contiguous_container

.. doxygenstruct:: metalchat::container_remove_type
   :members:

.. doxygenstruct:: metalchat::container_offset
   :members:

.. doxygenstruct:: metalchat::container_rebind
   :members:

.. doxygenstruct:: metalchat::container_traits
   :members:
