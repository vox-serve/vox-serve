import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from vox_serve import __version__ as package_version
except Exception:  # noqa: BLE001
    package_version = "0.0.0"

project = "VoxServe"
author = "VoxServe Team"
current_year = datetime.now().year
copyright = f"2025-{current_year}, {author}"
version = package_version
release = package_version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "nbsphinx",
    "sphinx.ext.mathjax",
]

autosectionlabel_prefix_document = True
autodoc_typehints = "description"
autosummary_generate = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3
myst_ref_domains = ["std", "py"]

nbsphinx_allow_errors = True
nbsphinx_execute = "never"
nbsphinx_allow_directives = True
nbsphinx_kernel_name = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nb_render_priority = {
    "html": (
        "application/vnd.jupyter.widget-view+json",
        "application/javascript",
        "text/html",
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "text/markdown",
        "text/latex",
        "text/plain",
    )
}

templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = f"{project} v{version} Documentation"
html_copy_source = True
html_last_updated_fmt = ""

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "fastapi": ("https://fastapi.tiangolo.com", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

html_theme_options = {
    "repository_url": "https://github.com/vox-serve/vox-serve",
    "repository_branch": "main",
    "show_navbar_depth": 3,
    "max_navbar_depth": 4,
    "collapse_navbar": False,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
}

html_context = {
    "display_github": True,
    "github_user": "vox-serve",
    "github_repo": "vox-serve",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_css_files = ["css/custom_log.css"]


def setup(app):
    app.add_css_file("css/custom_log.css")


copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
autodoc_preserve_defaults = True
navigation_with_keys = False

autodoc_mock_imports = [
    "torch",
    "transformers",
    "triton",
    "onnxruntime",
]

nbsphinx_prolog = """
.. raw:: html

    <style>
        .output_area.stderr, .output_area.stdout {
            color: #d3d3d3 !important; /* light gray */
        }
    </style>
"""
