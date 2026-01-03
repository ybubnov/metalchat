String tokenization
===================

These are the building blocks of the tokenization mechanism: chopping the input character sequence
into the grouped units of characters. This library could be used like following:

.. code-block:: c++

    #include <metalchat/text.h>

    using namespace metalchat::text;


Byte pair encoder
-----------------

.. doxygenclass:: metalchat::text::byte_pair_encoder
   :members:

.. doxygentypedef:: metalchat::text::tokenkind

.. seealso::

   The list of available token kinds is presented below and essentially are inherited from the
   Meta Llama Prompt Template Guide.

   More more details on prompt engineering, refer to the `Llama 3.1 Prompt Template Guide
   <https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#prompt-template)>`_.

.. doxygenvariable:: metalchat::text::token::regular
.. doxygenvariable:: metalchat::text::token::begin_text
.. doxygenvariable:: metalchat::text::token::end_text
.. doxygenvariable:: metalchat::text::token::reserved
.. doxygenvariable:: metalchat::text::token::finetune_right_pad
.. doxygenvariable:: metalchat::text::token::begin_header
.. doxygenvariable:: metalchat::text::token::end_header
.. doxygenvariable:: metalchat::text::token::end_message
.. doxygenvariable:: metalchat::text::token::end_turn
.. doxygenvariable:: metalchat::text::token::ipython


Regular expression
------------------

The implementation of regular expressions from the standard C++ library does not support Perl
syntax, used in the Tiktoken library. This implementation uses Perl Compatible Regular Expressions
library (`PCRE <https://www.pcre.org/>`_) to make available Perl syntax.

.. doxygenclass:: metalchat::text::regexp
   :members:

.. doxygenclass:: metalchat::text::regexp_iterator
   :members:
