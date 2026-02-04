Command line
============

.. warning::

   Work in progress


Credentials management
^^^^^^^^^^^^^^^^^^^^^^

You can use ``metalchat credential`` command to store the access tokens. On MacOS
the credential is stored in Keychain Access in a secure way and only queried by the ``metalchat``
command, when accessing remote resources.

.. prompt::

   metalchat credential add --host huggingface.co --username $HF_USERNAME --secret $HF_ACCESS_TOKEN

Then you could list access tokens using the ``list`` sub-command. Here the hostname is defined
as part of the URL. If the same URL prefix is used in the model pulling command, the access
token will be automatically pulled from the secrets provider and used to authenticate requests.

.. prompt::

   metalchat credential list
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

.. prompt::

   metalchat model pull https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

This command assigns to each model a SHA-1 identifier comprised of a repository URL, model
architecture, model variant, and weights partitioning. You can use this identifier to switch
between models.

.. prompt::

   metalchat model list --abbrev
   e37f2df  llama3  consolidated  https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct


Switching models
^^^^^^^^^^^^^^^^

TBD.


Configuring options
^^^^^^^^^^^^^^^^^^^

TBD.


Prompting models
^^^^^^^^^^^^^^^^

TBD.
