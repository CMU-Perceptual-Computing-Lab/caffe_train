# Caffe_train

Our modified caffe for training multi-person pose estimator. The original caffe version is in July 2016. This repository at least runs on Ubuntu 14.04, OpenCV 2.4.10, CUDA 7.5/8.0, and CUDNN 5. 

The [full project repo](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) includes detailed training steps and the testing code in matlab, C++ and python.

We add customized caffe layer for data augmentation: [cpm_data_transformer.cpp](https://github.com/CMU-Perceptual-Computing-Lab/caffe_train/blob/master/src/caffe/cpm_data_transformer.cpp), including scale augmentation e.g., in the range of 0.7 to 1.3, rotation augmentation, e.g., in the range of -40 to 40 degrees, flip augmentation and image cropping. This augmentation strategy makes the method capable of dealing with a large range of scales and orientations. You can set the augmentation parameters in [setLayers.py](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/training/setLayers.py). Example data layer parameters in the [training prototxt](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/training/example_proto/pose_train_test.prototxt) is:

```
layer {
  name: "data"
  type: "CPMData"
  top: "data"
  top: "label"
  data_param {
    source: "/home/zhecao/COCO_kpt/lmdb_trainVal"
    batch_size: 10
    backend: LMDB
  }
  cpm_transform_param {
    stride: 8
    max_rotate_degree: 40
    visualize: false
    crop_size_x: 368
    crop_size_y: 368
    scale_prob: 1
    scale_min: 0.5
    scale_max: 1.1
    target_dist: 0.6
    center_perterb_max: 40
    do_clahe: false
    num_parts: 56
    np_in_lmdb: 17
  }
}
```
This project is licensed under the terms of the GPL v3 license [![License](https://img.shields.io/aur/license/yaourt.svg)](LICENSE). We will merge it with the caffe testing version (https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose) later.

## Citation
Please cite the paper in your publications if it helps your research:



    @article{cao2016realtime,
	  title={Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
	  author={Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
	  journal={arXiv preprint arXiv:1611.08050},
	  year={2016}
	  }

    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
      }

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
