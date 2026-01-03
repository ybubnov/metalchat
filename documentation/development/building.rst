Building from source
====================

In this guide we will walk through the configuration of development environment: installation of
necessary tools and packagase, and configuring buids.

Obtaining source code
^^^^^^^^^^^^^^^^^^^^^

The source code of the project is served on GitHub, and could be retrieved in the following way:

.. prompt:: bash

   git clone https://github.com/ybubnov/metalchat
   cd metalchat


Creating an environment
^^^^^^^^^^^^^^^^^^^^^^^

The building of the library and tests is implemented using a `conan <https://conan.io/>`_ to
resolve C++ dependencies, and `CMake <https://cmake.org/>`_ plus `ninja <https://ninja-build.org/>`_
as a build system.

All of those tools are available through `brew <https://brew.sh/>`_ package manager on MacOS, so
you could install them like following:

.. prompt:: bash

   brew install cmake conan ninja


Configuring a development build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the next step you need to setup conan profile, usually a default conan profile would work
without modifications to compile a library on MacOS, nevertheless here we present an example
profile (let's call it `metalchat-debug`). In the highlighted line we explicitly state that
we need a C++ compiler with 23 standard support.

.. code-block:: ini
   :caption: ~/.conan2/profiles/metalchat-debug
   :linenos:
   :emphasize-lines: 5

   [settings]
   arch=armv8
   build_type=Debug
   compiler=apple-clang
   compiler.cppstd=gnu23
   compiler.libcxx=libc++
   compiler.version=17
   os=Macos


After that you could use this profile to install missing C++ dependencies and create a build
environment. Run this command from the project directory root (it will create `build` directory):

.. prompt:: bash

   conan build --build=missing --output-folder build --profile:host=metalchat-debug .


Building a library
^^^^^^^^^^^^^^^^^^

On the last step, compile the library and all related unit tests, and then optionally launch unit
tests, like in the following snippet:

.. prompt:: bash

   ninja
   ninja test
