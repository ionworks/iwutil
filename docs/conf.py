import os
import sys
import iwutil as iu

# Path for repository root
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "Ionworks Utility Functions"
copyright = "2025, Ionworks Technologies Inc"
author = "Ionworks Technologies Inc"

# Note: Both version and release are used in the build
version = iu.__version__
release = iu.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
]
myst_dmath_double_inline = True
templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_permalinks_icon = "<span>¶</span>"
