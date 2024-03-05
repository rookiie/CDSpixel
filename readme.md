# Learning Invariant Inter-pixel Correlations for Superpixel Generation
## This code is built based on SCN original sources and we promise that this code has no information about authors of CDSpixel (Submission 4385, AAAI2024).

## Prerequisites
The training code was mainly developed and tested with python 3.7, PyTorch 1.6, CUDA 10.2, and Ubuntu 16.04.

In the test, we make use of the component connection method in SSN to enforce the connectivity in superpixels. The code has been included in ```/third_paty/cython```. To compile it:
 ```
cd third_party/cython/
python setup.py install --user
cd ../..
```

## Demo
The demo script ```demo.py``` provides the superpixels with a grid size of ```16 x 16``` using our pre-trained model (in ```./weights/model_best.tar```).
Please feel free to provide your own images by copying them into ```/demo/inputs```, and run 
```
python demo.py --data_dir=./demo/inputs --data_suffix=jpg --output=./demo 
```
The results will be generated in a new folder under ```/demo``` called ```spixel_viz```.

## Data preparation 
To generate the training and test dataset, please first download the data from the original [BSDS500 dataset](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz), 
and extract it to  ```<BSDS_DIR>```. Then, run 
```
cd data_preprocessing
python pre_process_bsd500.py --dataset=<BSDS_DIR> --dump_root=<DUMP_DIR>
python pre_process_bsd500_ori_sz.py --dataset=<BSDS_DIR> --dump_root=<DUMP_DIR>
cd ..
```
The code will generate three folders under the ```<DUMP_DIR>```, named as ```/train```, ```/val```, and ```/test```, and three ```.txt``` files 
record the absolute path of the images, named as ```train.txt```, ```val.txt```, and ```test.txt```. Here, we generate the edge label for training data only.

## Testing
We provide test code to generate: 1) superpixel visualization and 2) the```.csv``` files  for evaluation. 

To test on BSDS500, run
```
python test_bsds.py --data_dir=<DUMP_DIR> --output=<TEST_OUTPUT_DIR> --pretrained=<PATH_TO_THE_CKPT>
```
