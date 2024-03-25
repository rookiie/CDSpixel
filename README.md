# CDSpixel
The official implementation for aaai24 [Learning Invariant Inter-pixel Correlations for Superpixel Generation'](https://arxiv.org/abs/2402.18201)

This repo is based on [SCN](https://github.com/fuy34/superpixel_fcn). So the installation and data preparation is pretty similar.

# Installation

**Step 0.** Install PyTorch and Torchvision following [official instructions](https://pytorch.org/get-started/locally/), e.g.,

```shell
pip install torch torchvision
# The training code was mainly developed and tested with python 3.7, PyTorch 1.8, CUDA 10 
```

**Step 1.** Follow the protocol of SCN and SSN, compile the component connection method in SSN to enforce the connectivity in superpixels. 
```shell
cd third_party/cython/
python setup.py install --user
cd ../..
```

# Dataset Preparation


