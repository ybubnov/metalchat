:html_theme.sidebar_secondary.remove: true

Llama inference for Apple Devices
=================================

MetalChat is a `Metal <https://developer.apple.com/metal/>`_-accelerated C++ framework and command
line interpreter for inference of `Meta Llama <https://www.llama.com/>`_ models. MetalChat is
designed as a full-stack framework, allowing to provide access to both low-level GPU kernels and
high-level LLM interpreter API.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Open source

      Distributed under a copy-left GPLv3 license, MetalChat is developed and maintained
      `publicly on GitHub <https://github.com/ybubnov/metalchat>`_.

   .. grid-item-card:: Lightweight

      MetalChat supports only Apple hardware with little external dependencies.

   .. grid-item-card:: HuggingFace compatible

      MetalChat supports Llama models distributed through
      `HuggingFace Hub <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`_ out of the box.


You can install both Framework and binary by adding a
`third-party repository <https://docs.brew.sh/Taps>`_, and running ``brew install``. You can
get more details how to use ``metalchat`` binary in the :doc:`command line <guides/command_line>`
guide.

.. prompt::

   brew tap ybubnov/metalchat https://github.com/ybubnov/metalchat
   brew install --HEAD metalchat


User Guide
^^^^^^^^^^

Information about installation and usage of the MetalChat library and the binary utility.


.. toctree::
   :maxdepth: 2

   User Guide <guides/index>


Development notes and contribution guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Information about the development principles of this library and how you can contribute.


.. toctree::
   :maxdepth: 2

   Development <development/index>


MetalChat reference
^^^^^^^^^^^^^^^^^^^

The programming interface exposed by the MetalChat library.

.. toctree::
   :maxdepth: 2

   Reference <reference/index>


.. toctree::
   :hidden:

   Sponsor <https://github.com/sponsors/ybubnov>
