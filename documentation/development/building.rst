Building from source
====================

In this guide we will walk through the configuration of development environment: installation of
necessary tools and packagase and configuring builds.

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

These building instruments are installed by the conan itself, and therefore do not require manual
installation. The primary way of building the framework is using the Python environment. You can
the Python environment in the following steps:

1. Install `pipenv` instrument:

.. prompt:: bash

   brew install pipenv


2. Install project dependencies:

.. prompt:: bash

   pipenv sync --dev

3. Enter pipenv-shell:

.. prompt:: bash

   pipenv shell


Configuring a development build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the next step you need to setup conan profile, usually a default conan profile would work
without modifications to compile a library on MacOS, nevertheless here we present an example
profile (let's call it `metalchat-debug`). In the highlighted line we explicitly state that
we need a C++ compiler with 23 standard support.

.. code-block:: ini
   :caption: ~/.conan2/profiles/metalchat-debug
   :emphasize-lines: 5

   [settings]
   arch=armv8
   build_type=Debug
   compiler=apple-clang
   compiler.cppstd=gnu23
   compiler.libcxx=libc++
   compiler.version=17
   os=Macos


Building binaries
^^^^^^^^^^^^^^^^^

After that you could use this profile to install missing C++ dependencies and create a build
environment. Run this command from the project directory root (it will create `build` directory):

.. prompt:: bash

   conan build --build=missing --profile:host=metalchat-debug


Optionally, you could build a framework without running tests:

.. prompt:: bash

   conan build --build=missing --profile:host=metalchat-debug -c tools.build:skip_test=True

Conan builds both, a framework and executable command-line program. The build of the program
could be disable using conan package option `build_executable`:

.. prompt:: bash

   conan build --build=missing --profile:host=metalchat-build -o build_executable=False

Building a conan package
^^^^^^^^^^^^^^^^^^^^^^^^

MetalChat could be used as a conan dependency, for this purpose you could build a conan package
in the following way:

.. prompt:: bash

   conan export-pkg
