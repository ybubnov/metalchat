:html_theme.sidebar_secondary.remove:


API reference
=============

TBD

.. toctree::
   :hidden:
   :maxdepth: 2

   text/index
   layer/index
   kernel/index
   tensor/index
   allocator
   container


Allocator library
-----------------

.. list-table::
   :width: 100%
   :widths: 45 55

   * - :cpp:class:`metalchat::hardware_memory_allocator`
     - Metal hazard-tracking shared memory allocator.

   * - :cpp:class:`metalchat::hardware_heap_allocator`
     - Metal heap allocator of shared memory without hazard tracking.

   * - :cpp:class:`metalchat::hardware_resident_allocator`
     - Moves all hardware allocations to the residency set.

   * - :cpp:class:`metalchat::aliasing_allocator`
     - Provides a way to assign ownership of an existing resource to the allocated
       container.
   * - :cpp:class:`metalchat::random_memory_allocator`
     - Allocates containers from random access memory.

   * - :cpp:class:`metalchat::scalar_memory_allocator`
     - Allocates containers that holds a single scalar value (int, float, double, etc.).

   * - :cpp:class:`metalchat::nocopy_allocator`
     - Allocates containers without actually copying underlying memory.

   * - :cpp:class:`metalchat::pooling_allocator_adapter`
     - An allocator that returns the same backing container.

   * - :cpp:class:`metalchat::rebind_allocator`
     - Provides a way to obtain an allocator of a different type.

   * - :cpp:class:`metalchat::paginated_allocator_adapter`
     - Provides a way to allocate multiple small-sized containers backing large
       allocation request.

   * - :cpp:class:`metalchat::polymorphic_hardware_allocator`
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

   * - :cpp:class:`metalchat::nn::llama3`
     - TBD


Chat library
-------------

.. list-table::
   :width: 100%
   :widths: 45 55

   * - :cpp:class:`metalchat::byte_pair_encoder`
     - Byte-pair encoder.

   * - :cpp:class:`metalchat::agent`
     - A language model adapter to receive and send messages.

   * - :cpp:class:`metalchat::transformer`
     - A language estimator adapter to predict the next token.
