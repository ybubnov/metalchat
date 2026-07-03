Message formatting
==================

These are building blocks of the message formatting in a dialogue with large language model. The
basic types could be included into the project like following:

.. code-block:: c++

   #include <metalchat/format.h>

   using namespace metalchat::text;


Basic message formatting
------------------------

.. doxygenclass:: metalchat::basic_message
   :members:


.. doxygenstruct:: metalchat::basic_formatter
   :members:


Token scanning
--------------

.. doxygenclass:: metalchat::basic_token_scanner
   :members:

.. doxygenclass:: metalchat::match_token_scanner
   :members:

.. doxygenclass:: metalchat::limit_token_scanner
   :members:

.. doxygenclass:: metalchat::composite_token_scanner
   :members:


Meta Llama 3 formatting
-----------------------

The implementation of the Meta Llama 3 formatting is implemented in the `huggingface` library,
and could be imported like in the example below:

.. code-block:: c++

   #include <metalchat/huggingface/llama.h>

   using namespace metalchat::huggingface;


.. doxygenclass:: metalchat::huggingface::llama3_formatter
   :members:

.. seealso::

   More details on prompt engineering, refer to the `Llama 3.1 Prompt Template Guide
   <https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#prompt-template>`_.


Google Gemma 3 formatting
-------------------------

The implementation of the Google Gemma 3formatting is implemented in the `huggingface` library,
and could be imported like in the example below:

.. code-block:: c++

   #include <metalchat/huggingface/gemma.h>

   using namespace metalchat::huggingface;


.. doxygenclass:: metalchat::huggingface::gemma3_formatter
   :members:

.. seealso::

   More details on prompt engineering, refer to the `Gemma formatting and system instructions
   <https://ai.google.dev/gemma/docs/core/prompt-structure>`_.
