VoxServe Documentation
======================

.. image:: https://img.shields.io/pypi/v/vox-serve?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI&color=3775A9
   :target: https://pypi.org/project/vox-serve/
   :alt: PyPI

.. image:: https://img.shields.io/badge/arXiv-2602.00269-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white
   :target: https://arxiv.org/abs/2602.00269
   :alt: arXiv

.. image:: https://img.shields.io/github/stars/vox-serve/vox-serve?style=for-the-badge&logo=github&logoColor=white
   :target: https://github.com/vox-serve/vox-serve
   :alt: GitHub Stars

VoxServe is a streaming-centric serving system for Speech Language Models (SpeechLMs), supporting both
text-to-speech (TTS) and speech-to-speech (STS) workloads.

VoxServe v2 serves models with a Rust runtime and splits the GPU spatially by the urgency of audio work
(:doc:`concepts`); read the `release blog post <https://vox-serve.github.io/2026/11/01/voxserve-v2.html>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   concepts

.. toctree::
   :maxdepth: 2
   :caption: Usage Guides

   usage/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   cli
   runtimes
   models
   architecture

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   development
   adding-a-model
