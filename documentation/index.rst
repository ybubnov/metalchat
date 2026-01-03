:html_theme.sidebar_secondary.remove: true

Llama inference for Apple Devices
=================================

MetalChat is a `Metal <https://developer.apple.com/metal/>`_-accelerated C++ framework and command
line interpreter for inference of `Meta Llama <https://www.llama.com/>`_ models. MetalChat is
designed as a full-stack framework, allowing to provide access to both low-level GPU kernels and
high-level LLM interpreter API.

.. toctree::
   :hidden:

   Install <install>
   User Guide <guides/index>
   Development <development/index>
   Reference <reference/index>
   Sponsor <https://github.com/sponsors/ybubnov>

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
