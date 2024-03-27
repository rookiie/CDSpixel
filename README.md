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

## BSDS500 

Download and convert the primary training dataset, i.e., [BSDS](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500), and put it in <BSDS_DIR> folder.
```shell
mkdir -p data/BSDS
# Put the downloaded archive in this folder

python ./tools/pre_process_bsd500.py --dataset=<BSDS_DIR> --dump_root=<DUMP_DIR>
# E.g., python ./tools/pre_process_bsd500.py --dataset='./data/BSDS' --dump_root='./data/BSDS'

python pre_process_bsd500_ori_sz.py --dataset=<BSDS_DIR> --dump_root=<DUMP_DIR>
# E.g., python ./tools/pre_process_bsd500_ori_sz.py --dataset='./data/BSDS' --dump_root='./data/BSDS'
```
The code will generate three folders under the <DUMP_DIR>, named as /train, /val, and /test, and three .txt files record the absolute path of the images, named as train.txt, val.txt, and test.txt.

## Custom Datasets 

we evaluate our methods mainly on NYUv2, VOC2012, KITTI. Since training or evaluation requires csv labels, use the following command to convert the label format, e.g.,
```shell
python ./tools/trans_label.py --dataset='VOC2012' --label_path=<LABEL_DIR> --trans_path=<CSV_DIR>
```
<LABEL_DIR> is the segmentation label path.


# Citation

If you find this project useful in your research, please consider cite:
```shell
@inproceedings{xu2024leaning,
  title={Learning Invariant Inter-pixel Correlations for Superpixel Generation},
  author={Xu, Sen and Wei, Shikui and Ruan, Tao and Liao, Lixin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={6351-6359},
  year={2024},
  DOI={10.1609/aaai.v38i6.28454},
}
```
