# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import os
import datetime
from pathlib import Path
from packaging.version import Version

# -- Read the Docs Specific Configuration --------------------------------------

# This needs to be done before anything is imported
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    os.environ["SUNPY_CONFIGDIR"] = "/home/docs/"
    os.environ["HOME"] = "/home/docs/"
    os.environ["LANG"] = "C"
    os.environ["LC_ALL"] = "C"
    os.environ["PARFIVE_HIDE_PROGRESS"] = "True"

# -- Project information -----------------------------------------------------

# The full version, including alpha/beta/rc tags
from irispy import __version__

_version = Version(__version__)
version = release = str(_version)
# Avoid "post" appearing in version string in rendered docs
if _version.is_postrelease:
    version = release = _version.base_version
# Avoid long githashes in rendered Sphinx docs
elif _version.is_devrelease:
    version = release = f"{_version.base_version}.dev{_version.dev}"
is_development = _version.is_devrelease
is_release = not (_version.is_prerelease or _version.is_devrelease)

project = "irispy"
author = "IRIS Instrument Team @ LMSAL"
copyright = f"{datetime.datetime.now(datetime.UTC).year}, {author}"  # NOQA: A001

# -- General configuration ---------------------------------------------------

import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
from matplotlib import MatplotlibDeprecationWarning
from sunpy.util.exceptions import SunpyDeprecationWarning, SunpyPendingDeprecationWarning

# Need to make sure that our documentation does not raise any of these
warnings.filterwarnings("error", category=SunpyDeprecationWarning)
warnings.filterwarnings("error", category=SunpyPendingDeprecationWarning)
warnings.filterwarnings("error", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("error", category=AstropyDeprecationWarning)

# Wrap large function/method signatures
maximum_signature_line_length = 80

extensions = [
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_changelog",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# Register the template for the robots.txt
html_extra_path = ["robots.txt"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
source_suffix = {".rst": "restructuredtext"}

# The master toctree document.
master_doc = "index"

# Treat everything in single ` as a Python reference.
default_role = "py:obj"

# Enable and configure nitpicky mode
nitpicky = True
# This is not used. See docs/nitpick-exceptions file for the actual listing.
nitpick_ignore = []
with Path("nitpick-exceptions").open() as nitpick_exceptions:
    for line in nitpick_exceptions:
        if line.strip() == "" or line.startswith("#"):
            continue
        dtype, target = line.split(None, 1)
        target = target.strip()
        nitpick_ignore.append((dtype, target))

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "aiapy": ("https://aiapy.readthedocs.io/en/latest/", None),
    "astropy": ("https://docs.astropy.org/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "mpl_animators": ("https://docs.sunpy.org/projects/mpl-animators/en/latest/", None),
    "ndcube": ("https://docs.sunpy.org/projects/ndcube/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "parfive": ("https://parfive.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3/", None),
    "reproject": ("https://reproject.readthedocs.io/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "sunpy": ("https://docs.sunpy.org/en/latest/", None),
    "sunraster": ("https://docs.sunpy.org/projects/sunraster/en/latest/", None),
}

# -- Options for sphinxext-opengraph ------------------------------------------

ogp_image = "https://raw.githubusercontent.com/sunpy/sunpy-logo/master/generated/sunpy_logo_word.png"
ogp_use_first_image = True
ogp_description_length = 160
ogp_custom_meta_tags = ('<meta property="og:ignore_canonical" content="true" />',)

# -- Options for sphinx-copybutton ---------------------------------------------

# Python Repl + continuation, Bash, ipython and qtconsole + continuation, jupyter-console + continuation
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sunpy"

# Render inheritance diagrams in SVG
graphviz_output_format = "svg"

graphviz_dot_args = [
    "-Nfontsize=10",
    "-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Efontsize=10",
    "-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Gfontsize=10",
    "-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# By default, when rendering docstrings for classes, sphinx.ext.autodoc will
# make docs with the class-level docstring and the class-method docstrings,
# but not the __init__ docstring, which often contains the parameters to
# class constructors across the scientific Python ecosystem. The option below
# will append the __init__ docstring to the class-level docstring when rendering
# the docs. For more options, see:
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autoclass_content = "both"

# -- Other options ----------------------------------------------------------

# Configuration for sphinx-gallery
from sunpy_sphinx_theme import PNG_ICON

sphinx_gallery_conf = {
    "backreferences_dir": Path("generated") / "modules",
    "filename_pattern": "^((?!skip_).)*$",
    "examples_dirs": Path("..") / "examples",
    "within_subsection_order": "ExampleTitleSortKey",
    "gallery_dirs": Path("generated") / "gallery",
    "default_thumb_file": PNG_ICON,
    "abort_on_example_error": False,
    "plot_gallery": "True",
    "remove_config_comments": True,
    "doc_module": ("sunpy"),
    "only_warn_on_example_error": True,
    "matplotlib_animations": True,
}
