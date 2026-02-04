Installing MetalChat
====================

The project is distributed as both a Framework, and a binary utility to run the model inference.
Currently, the installation implies compilation from sources using `Homebrew <https://brew.sh/>`_
package manager.

In order to install both Framework and binary, add a
`third-party repository <https://docs.brew.sh/Taps>`_:

.. prompt::

   brew tap ybubnov/metalchat https://github.com/ybubnov/metalchat

Then install the latest version of the project:

.. prompt::

   brew install --HEAD metalchat

Downloading and running a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the successful installation, you could pull a Llama model and run it. By the moment of
writing this documentation, MetalChat supports only pulling images from
`HuggingFace <https://huggingface.co>`_ repository. Meta's models there are gated, and therefore
you need to get access to those models at first. After that you could pull models using MetalChat
by an `access token <https://huggingface.co/docs/hub/en/security-tokens>`_.

Once you obtained an access token and granted an access to a gated Llama3 model, you are ready
to pull the model from the repository. MetalChat stores the provided secrets in
`Keychain Access <https://support.apple.com/en-ca/guide/keychain-access>`_ in a secured way.

.. prompt::

   metalchat credential add -H huggingface.co -u $HF_USERNAME -s $HF_ACCESS_TOKEN

After that you can pull a model:

.. prompt::

   metalchat model pull https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

And then run it:

.. prompt::

   echo 'What is the capital of Germany?' | metalchat - e37f2dfbbef2a9dcad4e1d83274b8ff5d55c5481
