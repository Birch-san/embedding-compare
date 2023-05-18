# Embedding Compare

Gonna try and make ways to tabulate the results of embedding comparisons.

## Setup

Clone repository (including ImageBind submodule):

```bash
git clone --recursive --depth 1 https://github.com/Birch-san/embedding-compare.git
cd embedding-compare
```

Create a Conda environment. I'm naming this after Python 3.11 and CUDA 12.1:

```bash
conda create -n p311-cu121 python=3.11
conda activate p311-cu121
```

Install dependencies listed in `requirements.txt`.  
I've configured it to use `--pre`, and told it (via `--extra-index-url ...`) to source from the CUDA 12.1 nightly channel. You could delete those options if you prefer a stable release.

```bash
pip install -r requirements.txt --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu121
```

## Run:

From root of repository:

```bash
python -m scripts.embed_play
```