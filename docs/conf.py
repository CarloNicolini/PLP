"""Sphinx configuration for the PLP documentation."""
from importlib.metadata import version as package_version

project = "Probabilistic Language Programming"
release = package_version("plp")
version = release
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]
html_theme = "pydata_sphinx_theme"
myst_enable_extensions = ["colon_fence", "dollarmath"]
exclude_patterns = ["_build", "presentation"]
# Research notes remain available in the repository; the curated site starts at index.
exclude_patterns += [
    "critics.md", "gemini_proposal_pymc.md", "inference_zoo.md",
    "llm_probabilistic_inference.md", "openrouter_gemini_gpt51*.md",
    "plp_inference_murphy_audit.md", "plp_introduction.md",
    "plp_theoretical_foundations.md",
]
autodoc_member_order = "bysource"
