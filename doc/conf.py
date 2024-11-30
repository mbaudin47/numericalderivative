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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sys
import os
import subprocess
from distutils.version import LooseVersion
import sphinx
import sphinx_gallery

sys.path.insert(0, os.path.abspath('../../..'))
# -- Project information -----------------------------------------------------

project = 'numericalderivative'
copyright = 'M. Baudin'
author = 'M. Baudin'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
]


sphinx_gallery_conf = {
    'examples_dirs': ['examples'], # # path to
    # example scripts
    'gallery_dirs': ['auto_example'],
    #path to where to save gallery gen. output
    'run_stale_examples':True,
    'show_signature': False
    }

if LooseVersion(sphinx.__version__) >= '1.8':
    autodoc_default_options = {'members': None, 'inherited-members': None}
else:
    autodoc_default_flags =  ['members', 'inherited-members']

intersphinx_mapping = {'openturns': ('http://openturns.github.io/openturns/latest', None)}
autosummary_generate = True

numpydoc_show_class_members = True
numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'python'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []



# -- Options for HTML output ----------------------------------------------

html_theme = 'openturns'
html_theme_path = ['themes']
# html_sidebars = {
#     '**': [
#         # 'about.html',
#         'navigation.html',
#         'relations.html',
#         'searchbox.html',
#         'donate.html',
#     ]
# }


# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/Icon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


html_show_sourcelink = True

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/Icon.ico'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'a4paper',
    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    'preamble': r'\usepackage{math_notations},\usepackage{stackrel}',
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = ['.rst']



man_pages = [
    ('index', 'numericalderivative', u'numericalderivative Documentation',
     [u'Michaël Baudin'], 1)
]

texinfo_documents = [
    ('index', 'numericalderivative', u'numericalderivative Documentation',
   u'Michaël Baudin', 'numericalderivative', 'Numerical Differentiation',
   'Miscellaneous'),
]
