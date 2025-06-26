import subprocess
import sys
import pathlib

# Project information
project = "MetalChat"
author = "Yakau Bubnou"
copyright = f"2024-present, {author}"


# General configuration
need_sphinx = "4.4"
extensions = [
    "breathe",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]


# Breathe configuration.
breathe_projects = {"metalchat": "_xml"}
breathe_default_project = "metalchat"
breathe_show_include = True
breathe_domain_by_extension = {"h": "cpp"}


# The documentation is in English language.
language = "en"


# Options for HTML output
pygments_style = "default"
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
templates_path = ["_templates"]


# Doxygen writes output XMLs to the `_xml` path, exclude it from sphinx result.
exclude_patterns = ["_xml"]


def doxygen(path):
    """Run the doxygen command in the designated path."""
    try:
        retcode = subprocess.call("cd %s; doxygen" % path, shell=True)
        if retcode < 0:
            sys.stderr.write("doxygen terminated by signal %s" % (-retcode))
    except OSError as e:
        sys.stderr.write("doxygen execution failed: %s" % e)


def generate_doxygen_xml(app):
    doxygen(pathlib.Path(__file__).parent.absolute())


def setup(app):
    """Generate doxygen documentation on `sphinx-build` application setup."""
    app.connect("builder-inited", generate_doxygen_xml)
