Command line
============

.. warning::

   Work in progress


Credentials management
^^^^^^^^^^^^^^^^^^^^^^

You can use ``metalchat credential`` command to store the access tokens. On MacOS
the credential is stored in `Keychain Access <https://support.apple.com/en-ca/guide/keychain-access>`_
in a secure way and only queried by the ``metalchat`` command, when accessing remote resources.

.. hint::

   You can create a HuggingFace access token by following through the
   `User Access Tokens Guide <https://huggingface.co/docs/hub/en/security-tokens>`_.

.. code:: console

   $ metalchat credential add --host huggingface.co --username $HF_USERNAME --secret $HF_ACCESS_TOKEN

Then you could list access tokens using the ``list`` sub-command. Here the hostname is defined
as part of the URL. If the same URL prefix is used in the model pulling command, the access
token will be automatically pulled from the secrets provider and used to authenticate requests.

.. code:: console

   $ metalchat credential list
   https://huggingface  username @keychain


Models management
^^^^^^^^^^^^^^^^^

.. note::

   You will need the access to the gated Meta Llama3 model model in order to run ``metalchat pull``
   command. You can do this by creating an account at `HuggingFace <https://huggingface.co/>`_.
   And then requesting access to a
   `Llama-3.2-1B-Instruct <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`_ model.

You can use ``metalchat model`` command to pull models from remote repositories, list or remove
then from the repository. By default all models are stored into ``$HOME/.metalchat/models``
directory.

.. code:: console

   $ metalchat model pull https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

This command assigns to each model a SHA-1 identifier comprised of a repository URL, model
architecture, model variant, and weights partitioning. You can use this identifier to switch
between models.

.. code:: console

   $ metalchat model list --abbrev
   e37f2df  llama3  consolidated  https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct


Switching models
^^^^^^^^^^^^^^^^

The ``metalchat`` utility uses ``metalchat.toml`` manifest file to keep the currently used
model version and all respective options of that model, as well as environment parameters.
The utility distinguishes three scopes: local, global, and model. You can find the manifest
file in each of those scopes.

The scopes correspond to the following locations:

- ``local`` - the current working directory.
- ``global`` - a directory located at ``$HOME/.metalchat``.
- ``model`` - a directory in the ``$HOME/.metalchat/models``.

You can use ``metalchat checkout`` command to switch models use either in local or global scope.
By default, this command switches a model in the local scope.

.. code:: console
   :force:

   $ metalchat checkout e37f2dfbbef2a9dcad4e1d83274b8ff5d55c5481
   $ cat metalchat.toml
   [model]
   repository = "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"
   architecture = "llama3"
   partitioning = "consolidated"
   variant = "huggingface"

In a similar way, you can switch a model in the ``global`` scope:

.. code:: console

   $ metalchat checkout --global e37f2dfbbef2a9dcad4e1d83274b8ff5d55c5481


Configuring options
^^^^^^^^^^^^^^^^^^^

TBD.


Prompting models
^^^^^^^^^^^^^^^^

TBD.
