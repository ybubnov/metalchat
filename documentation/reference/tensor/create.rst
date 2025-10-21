Advanced creation
=================

Scalar
------

.. doxygenfunction:: metalchat::scalar


Empty
------

.. doxygenfunction:: metalchat::empty(std::size_t (&&sizes)[N])

.. doxygenfunction:: metalchat::empty(std::size_t (&&sizes)[N], Allocator alloc)

.. doxygenfunction:: metalchat::empty(std::size_t (&&sizes)[N], const hardware_accelerator& accelerator);

.. doxygenfunction:: metalchat::empty(InputIt begin, InputIt end)

.. doxygenfunction:: metalchat::empty_like(const Tensor& like)

.. doxygenfunction:: metalchat::empty_like(const Tensor& like, Allocator alloc)


Full
----

.. doxygenfunction:: metalchat::full(std::size_t (&&sizes)[N], const T& fill_value)

.. doxygenfunction:: metalchat::full(std::size_t (&&sizes)[N], const T& fill_value, Allocator alloc)

.. doxygenfunction:: metalchat::full(std::size_t (&&sizes)[N], const T& fill_value, const hardware_accelerator& accelerator)

.. doxygenfunction:: metalchat::zeros(std::size_t (&&sizes)[N])
