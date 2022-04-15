"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

# -- Project information -----------------------------------------------------

project = 'SparseOperationKit'
copyright = '2022, NVIDIA'
author = 'NVIDIA'

# version
def _GetSOKVersion():
    _version_path = os.path.join(os.getcwd(), "../../sparse_operation_kit/core/")
    sys.path.append(_version_path)
    from _version import __version__
    version = __version__
    del __version__
    sys.path.pop(-1)
    return version

version = _GetSOKVersion()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_multiversion"
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# multiversion settings ---------------------
# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = r'^sok_v.*$'

# Whitelist pattern for branches (set to None to ignore all branches)
# smv_branch_whitelist = r'^sok_docs|master$'
smv_branch_whitelist = r'^None$'

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = r'^master$'

# Pattern for released versions
smv_released_pattern = r'^.*$'

# Format for versioned output directories inside the build directory
# smv_outputdir_format = 'versions/{config.version}'

# Determines whether remote or local git branches/tags are preferred if their output dirs conflict
smv_prefer_remote_refs = False

# options for autodoc
autodoc_typehints = "signature"
autoclass_content = "both"
autodoc_class_signature = "mixed"

autodoc_default_options = {
    "exclude-members": "__new__"
}
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_sidebars = {"**": ["versions.html"]}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

source_suffix = [".rst", ".md", ".txt"]
