[![DOI](https://zenodo.org/badge/970263838.svg)](https://doi.org/10.5281/zenodo.15270437)

# A Comparative Study on LLM-enabled HPC User Support

This git repository contains the code for preproducing the results in our submitted paper.

The train and eval datasets contain TACC user private email information, so they are not included here.

## Hardware requirement
This work used 64 GH200 GPUs for fine-tuning Llama-3.1-8B model, each of which has 96 GB HBM and 120 GB RAM.

## Dataset
We cleaned the user tickets stored in TACC database to form question-answer pairs. According to TACC's user privacy policy, we cannot release the train and evaluation dataset in public. If you are interested, please register a TACC account [here](https://accounts.tacc.utexas.edu/register?_gl=1*15ri7bv*_ga*ODMzMTMwOTkwLjE3MDc3NTMzNjA.*_ga_TRRRQZ0EHX*MTc0NTMzMzg0Mi4xMjAuMC4xNzQ1MzMzODQyLjAuMC4w) and contact us for access to the datasets on TACC machines.

## Software dependencies

Please use `pip install -r requirements.txt` to install the required packages. 

If you are using ARM machines (e.g., NVIDIA GH200 GPUs), please follow the instruction below to install gpt4all:

```bash
# https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-bindings/python/README.md
# MUST USE v3.4.2
# THERE IS A SERIOUS BUG FOR THE LATEST VERSION (TypeError: expected CFunctionType instance instead of function in _pyllmodel.py)

git clone --recurse-submodules https://github.com/nomic-ai/gpt4all.git –branch v3.4.2
cd gpt4all/gpt4all-backend

# Add “include(CheckCXXCompilerFlag)” to /gpt4all/gpt4all-backend/CMakeLists.txt

# If you are using a supercomputer and see an error caused by the gcc or g++ version, 
# use conda to install your desired version of gcc
conda install -c conda-forge cxx-compiler gcc

CXX=/path/to/miniconda3/bin/g++ CC=/path/to/miniconda3/bin/gcc cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=/path/to/miniconda3/bin/g++ -DCMAKE_C_COMPILER=/path/to/miniconda3/bin/gcc -DLLMODEL_KOMPUTE=OFF

CXX=/path/to/miniconda3/bin/g++ CC=/path/to/miniconda3/bin/gcc cmake --build build --parallel
```

## Experiment workflow
fine-tuning -> inference -> evaluation