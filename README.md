# get docker image #
```
https://hub.docker.com/r/patrickchrist/cascadedfcn/
docker pull patrickchrist/cascadedfcn
```

# get python script from notebook #
```
jupyter nbconvert --to script notebooks/cascaded_unet_inference.ipynb
ipython qtconsole  --> run cascaded_unet_inference.py
```

# get caffe version from docker
```
root@maeda:/opt/caffe# cat .git/config
[core]
        repositoryformatversion = 0
        filemode = true
        bare = false
        logallrefupdates = true
[remote "origin"]
        url = https://github.com/mohamed-ezz/caffe.git
        fetch = +refs/heads/*:refs/remotes/origin/*
[branch "master"]
        remote = origin
        merge = refs/heads/master
[branch "jonlong"]
        remote = origin
        merge = refs/heads/jonlong
root@maeda:/opt/caffe# git log
commit 876e387a6d7f8974f68f42beacd3728b4fc92ff7
Author: Mohamed Ezz <moh.ezz8@gmail.com>
Date:   Thu Feb 18 01:24:04 2016 +0100

    Add class weighting feature for softmax_loss layer

$ git clone https://github.com/mohamed-ezz/caffe.git caffe/docker
$ cd caffe/docker;  git branch docker 876e387a6d7f8974f68f42beacd3728b4fc92ff7 ; git checkout docker
```


# Cascaded-FCN #
This repository contains the pre-trained models for a Cascaded-FCN in caffe and tensorflow that segments the liver and its lesions out of axial CT images and a python wrapper for dense 3D Conditional Random Fields 3D CRFs.


This work was published in MICCAI 2016 paper ([arXiv link](https://arxiv.org/abs/1610.02177)) titled : 

```
Automatic Liver and Lesion Segmentation in CT Using Cascaded Fully Convolutional 
Neural Networks and 3D Conditional Random Fields
```
## Caffe ##
### Quick Start ###
If you want to use our code we offer an docker image, which runs our code and has all dependencies installed including the correct caffe version. After having installed docker and nvidia docker:
```
sudo GPU=0 nvidia-docker run -v $(pwd):/data -P --net=host --workdir=/Cascaded-FCN -ti --privileged patrickchrist/cascadedfcn bash
```
And than start jupyter notebook and browse to localhost:8888
```
jupyter notebook
```
## Tensorflow ## 
Please look at Readme and Documentation at https://github.com/FelixGruen/tensorflow-u-net
### Citation ###

If you have used these models in your research please use the following BibTeX for citation :
```
@Inbook{Christ2016,
title="Automatic Liver and Lesion Segmentation in CT Using Cascaded Fully Convolutional Neural Networks and 3D Conditional Random Fields",
author="Christ, Patrick Ferdinand and Elshaer, Mohamed Ezzeldin A. and Ettlinger, Florian and Tatavarty, Sunil and Bickel, Marc and Bilic, Patrick and Rempfler, Markus and Armbruster, Marco and Hofmann, Felix and D'Anastasi, Melvin and Sommer, Wieland H. and Ahmadi, Seyed-Ahmad and Menze, Bjoern H.",
editor="Ourselin, Sebastien and Joskowicz, Leo and Sabuncu, Mert R. and Unal, Gozde and Wells, William",
bookTitle="Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II",
year="2016",
publisher="Springer International Publishing",
address="Cham",
pages="415--423",
isbn="978-3-319-46723-8",
doi="10.1007/978-3-319-46723-8_48",
url="http://dx.doi.org/10.1007/978-3-319-46723-8_48"
}
```
```
@ARTICLE{2017arXiv170205970C,
   author = {{Christ}, P.~F. and {Ettlinger}, F. and {Gr{\"u}n}, F. and {Elshaera}, M.~E.~A. and 
	{Lipkova}, J. and {Schlecht}, S. and {Ahmaddy}, F. and {Tatavarty}, S. and 
	{Bickel}, M. and {Bilic}, P. and {Rempfler}, M. and {Hofmann}, F. and 
	{Anastasi}, M.~D and {Ahmadi}, S.-A. and {Kaissis}, G. and {Holch}, J. and 
	{Sommer}, W. and {Braren}, R. and {Heinemann}, V. and {Menze}, B.},
    title = "{Automatic Liver and Tumor Segmentation of CT and MRI Volumes using Cascaded Fully Convolutional Neural Networks}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1702.05970},
 primaryClass = "cs.CV",
 keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Artificial Intelligence},
     year = 2017,
}
```
```
@inproceedings{Christ2017SurvivalNetPP,
  title={SurvivalNet: Predicting patient survival from diffusion weighted magnetic resonance images using cascaded fully convolutional and 3D convolutional neural networks},
  author={Patrick Ferdinand Christ and Florian Ettlinger and Georgios Kaissis and Sebastian Schlecht and Freba Ahmaddy and Felix Gr{\"{u}n and Alexander Valentinitsch and Seyed-Ahmad Ahmadi and Rickmer Braren and Bjoern H. Menze},
  booktitle={ISBI},
  year={2017}
}
```
### Description ###
This work uses 2 cascaded UNETs, 

 1. In step1, a UNET segments the liver from an axial abdominal CT slice. The segmentation output is a binary mask with bright pixels denoting the segmented object. By segmenting all slices in a volume we obtain a 3D segmentation. 
 2. (Optional) We enhance the liver segmentation using 3D dense CRF (conditional random field). The resulting enhanced liver segmentation is then used further for step2.
 3. In step2 another UNET takes an enlarged liver slice and segments its lesions. 
 
The input to both networks is 572x572 generated by applying reflection mirroring at all 4 sides of a 388x388 slice. The boundary 92 pixels are reflecting, resulting in (92+388+92)x(92+388+92) = 572x572.

An illustration of the pipeline is shown below :

<img src="https://www.dropbox.com/s/rmqo0s82i8r1ihm/CascadedFCN_Pipeline.png?dl=1" width="512" alt="Illustration of the CascadedFCN pipeline">

For detailed Information have a look in our [presentation](Cascaded-FCN.pdf)

### 3D Conditional Random Field 3DCRF
You can find the 3D CRF at [3DCRF-python](https://github.com/mbickel/DenseInferenceWrapper). Please follow the installation description in the [Readme](https://github.com/mbickel/DenseInferenceWrapper/blob/master/readme.md).

### License 

These models are published with unrestricted use for research and educational purposes.
For commercial use, please refer to the paper authors.
