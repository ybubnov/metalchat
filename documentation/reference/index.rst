:html_theme.sidebar_secondary.remove:


API Reference
=============

TBD

.. toctree::
   :hidden:
   :maxdepth: 2

   allocator
   container
   tensor
   kernel
   layer/index
   chat


Allocator library
-----------------

.. list-table::
   :width: 100%
   :widths: 45 55

   * - :cpp:class:`metalchat::hardware_memory_allocator`
     - Metal hazard-tracking shared memory allocator.

   * - :cpp:class:`metalchat::hardware_buffer_allocator`
     - An allocator that returns the same hardware container (buffer).

   * - :cpp:class:`metalchat::hardware_heap_allocator`
     - Metal heap allocator of shared memory without hazard tracking.

   * - :cpp:class:`metalchat::hardware_nocopy_allocator`
     - Allocates hardware containers without actually copying underlying memory.

   * - :cpp:class:`metalchat::hardware_resident_allocator`
     - Moves all hardware allocations to the residency set.

   * - :cpp:class:`metalchat::hardware_aliasing_allocator`
     - Provides a way to assign ownership of an existing resource to the allocated
       container.

   * - :cpp:class:`metalchat::rebind_hardware_allocator`
     - Provides a way to obtain a hardware allocator of a different type.

   * - :cpp:class:`metalchat::random_memory_allocator`
     - Allocates containers from random access memory.

   * - :cpp:class:`metalchat::scalar_memory_allocator`
     - Allocates containers that holds a single scalar value (int, float, double, etc.).

   * - :cpp:class:`metalchat::paginated_allocator_adapter`
     - Provides a way to allocate multiple small-sized containers backing large
       allocation request.

   * - :cpp:class:`metalchat::polymorphic_hardware_memory_allocator`
     - An allocator that exhibits different allocation behaviour depending on particular
       implementation.


Container library
-----------------

.. list-table::
   :width: 100%
   :widths: 45 55

   * - :cpp:class:`metalchat::hardware_memory_container`
     - TBD.

   * - :cpp:class:`metalchat::random_memory_container`
     - TBD.

   * - :cpp:class:`metalchat::vector_memory_container`
     - TBD.

   * - :cpp:class:`metalchat::scalar_memory_container`
     - TBD.

   * - :cpp:class:`metalchat::reference_memory_container`
     - TBD.


Tensor library
--------------

.. list-table::
   :width: 100%
   :widths: 45 55

   * - :cpp:class:`metalchat::tensor`
     - A multi-dimensional matrix containing elements of a single data type.

   * - :cpp:class:`metalchat::tensor_iterator`
     - A sequential accessor to the tensor's elements.

   * - :cpp:class:`metalchat::shared_tensor_ptr`
     - A tensor that could be shared by multiple owners.

   * - :cpp:class:`metalchat::future_tensor`
     - A tensor associated with a computation task, which result is not ready yet.


Tensor functions
^^^^^^^^^^^^^^^^

.. list-table::
   :width: 100%
   :widths: 45 55

   * - :cpp:func:`metalchat::scalar`
     - TBD

   * - :cpp:func:`metalchat::empty`
     - TBD


Kernel library
--------------

.. list-table::
   :width: 100%
   :widths: 45 55

   * - :cpp:class:`metalchat::hardware_accelerator`
     - TBD

   * - :cpp:class:`metalchat::basic_kernel`
     - TBD

   * - :cpp:class:`metalchat::binary_kernel_wrapper`
     - TBD

   * - :cpp:class:`metalchat::kernel_task`
     - TBD

   * - :cpp:class:`metalchat::kernel_thread`
     - TBD

   * - :cpp:class:`metalchat::recursive_kernel_thread`
     - TBD


Layer library
-------------

.. list-table::
   :width: 100%
   :widths: 45 55

   * - :cpp:class:`metalchat::basic_layer`
     - A basic building block of neural networks in MetalChat.

   * - :cpp:class:`metalchat::shared_layer_ptr`
     - A convenience class for a :cpp:class:`metalchat::layer` type to share layer ownership.

   * - :cpp:class:`metalchat::nn::llama`
     - TBD


Chat library
-------------

.. list-table::
   :width: 100%
   :widths: 45 55

   * - :cpp:class:`metalchat::byte_pair_encoder`
     - Byte-pair encoder.

   * - :cpp:class:`metalchat::chat`
     - A language model adapter to receive and send messages.

   * - :cpp:class:`metalchat::polymorphic_chat`
     - TBD

   * - :cpp:class:`metalchat::language_transformer`
     - A language estimator adapter to predict the next token.
