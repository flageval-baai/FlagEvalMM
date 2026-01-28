Installation
============

This guide will help you install FlagEvalMM and its dependencies.

Basic Installation
------------------

The easiest way to install FlagEvalMM is from the source repository:

.. code-block:: bash

   git clone https://github.com/flageval-baai/FlagEvalMM.git
   cd FlagEvalMM
   pip install -e .

System Requirements
-------------------

* Python 3.10 or higher
* PyTorch 2.0.0 or higher
* CUDA 11.7+ (for GPU acceleration)

Backend Installations
---------------------

FlagEvalMM supports multiple backend engines for inference. Choose and install the ones you plan to use:

VLLM Backend
~~~~~~~~~~~~

Currently (July 28, 2025), we recommend using vllm==0.8.5.post1 and torch==2.6.0 for optimal inference performance and stability.

.. code-block:: bash

   pip install vllm==0.8.5.post1

.. warning::
   Make sure to use the recommended versions for the best compatibility and performance.

SGLang Backend
~~~~~~~~~~~~~~

.. code-block:: bash

   pip install --upgrade pip
   pip install "sglang[all]"
   pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

For detailed installation instructions, please refer to the `official SGLang documentation <https://sgl-project.github.io/start/install.html>`_.

LMDeploy Backend
~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install lmdeploy

For detailed installation instructions, please refer to the `official LMDeploy documentation <https://lmdeploy.readthedocs.io/en/latest/>`_.

FlagScale Backend
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/FlagOpen/FlagScale.git
   cd FlagScale/install
   ./install-requirements.sh --env inference
   cd vllm
   pip install .

For detailed installation instructions, please refer to the `official FlagScale documentation <https://lmdeploy.readthedocs.io/en/latest/>`_.

Transformers Backend
~~~~~~~~~~~~~~~~~~~~

For optimal performance with transformers, we recommend installing flash-attention:

.. code-block:: bash

   pip install flash-attn --no-build-isolation

API Keys Configuration
----------------------

If you want to evaluate tasks using GPT models (like charxiv, math_verse, etc.), you need to set the following environment variables:

.. code-block:: bash

   export FLAGEVAL_API_KEY=$YOUR_OPENAI_API_KEY
   export FLAGEVAL_BASE_URL="https://api.openai.com/v1"

For other API providers, adjust the base URL accordingly.

Verification
------------

To verify your installation, run:

.. code-block:: bash

   flagevalmm --help

You should see the help message with available command-line options.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Error**: If you encounter import errors, make sure all dependencies are installed correctly and your Python environment is activated.

**CUDA Issues**: For GPU-related problems, verify that your CUDA version is compatible with your PyTorch installation.

**Memory Issues**: For large models, ensure you have sufficient GPU memory. Consider using model sharding or reducing batch sizes.

Getting Help
~~~~~~~~~~~~

If you encounter issues during installation:

1. Check the `GitHub Issues <https://github.com/flageval-baai/FlagEvalMM/issues>`_
2. Join our community discussions
3. Contact the development team at flageval@baai.ac.cn 