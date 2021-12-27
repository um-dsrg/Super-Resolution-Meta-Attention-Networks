# Deep-FIR Codebase - Super Resolution Meta Attention Networks [![macOS](https://svgshare.com/i/ZjP.svg)](https://svgshare.com/i/ZjP.svg) [![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg) [![Windows](https://svgshare.com/i/ZhY.svg)](https://svgshare.com/i/ZhY.svg) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

About 
-------------------

This repository contains the main coding framework accompanying our work on meta-attention in Single Image Super-Resolution (SISR), which has been published in the IEEE Signal Processing Letters (SPL) [here](https://doi.org/10.1109/LSP.2021.3116518).  A sample of the results obtained by our metadata-enhanced models is provided below:

![training_system](Documentation/SPL_results_sample.png)

Installation
--------------------
### Python and Virtual Environments

If installing from scratch, it is first recommended to set up a new Python virtual environment prior to installing this code.  With [Conda](https://docs.conda.io/en/latest/), this can be achieved through the following:

```conda create -n *environment_name* python=3.7 ```  (Python 3.7 recommended but not essential).

```conda activate *environment_name*```

Code testing was conducted in Python 3.7, but the code should work fine with Python 3.6+.

### Local Installation

Run the following commands from the repo base directory to fully install the package and all requirements: 

```cd Code```

**If using CPU only:** ```conda install --file requirements.txt --channel pytorch --channel conda-forge```

**If using CPU + GPU:**  First install Pytorch and Cudatoolkit for your specific configuration using instructions [here](https://pytorch.org/get-started/locally/).  Then, install requirements as above.

**If using [Aim](https://github.com/aimhubio/aim)** for metrics logging, install via ```pip install aim```.  The Aim GUI does not work on Windows, but metrics should still be logged in the .aim folder.

Finally:

```pip install -e .```  This installs all the command-line functions from Code/setup.py.

All functionality has been tested on Linux (CPU & GPU), Mac OS (CPU) and Windows (CPU & GPU). 

Requirements installation is only meant as a guide and all requirements can be installed using alternative means (e.g. using ```pip```).

Guidelines for Generating SR Data  <a name="data-generate"></a>
-----------------
### Setting up CelebA Dataset
Create a folder 'celeba' in the Data directory.  In here, download all files from the celeba [source](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).  
Unpack all archives in this location.  Run ```image_manipulate``` to generate LR images and corresponding metadata (check details in Documentation/data_prep.md for more info on how to do this).
### Setting up CelebA-HQ Dataset
CelebA-HQ files can be easily downloaded from [here](https://github.com/switchablenorms/CelebAMask-HQ).  To generate LR images, check Documentation/data_prep.md as with CelebA.  For our IEEE SPL paper (super-resolving by 4x), we generated images using the following two commands:

To generate 512x512 HR images: ```image_manipulate --source_dir *path_to_original_images* --output_dir *path_to_new_folder* --pipeline downscale --scale 2```

To generate 128x128 LR images: ```image_manipulate --source_dir *path_to_512x512_images* --output_dir *path_to_new_folder* --pipeline blur-downscale --scale 4```

To generate pre-upscaled 512x512 LR images for SPARNet: ```image_manipulate --source_dir *path_to_128x128_images* --output_dir *path_to_new_folder* --pipeline upscale --scale 4```

### Setting up DIV2K/Flickr2K Datasets
DIV2K training/validation downloadable from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/).  
Flickr2K dataset downloadable from [here](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar).

Similar to CelebA-HQ, for our IEEE SPL paper (super-resolving by 4x), we generated LR images using the following command:

```image_manipulate --source_dir *path_to_original_HR_images* --output_dir *path_to_new_folder* --pipeline blur-downscale --scale 4```

For blurred & compressed images, we used the following command (make sure to first install JM to be able to compress the images, as detailed [here](#jm-install)):

```image_manipulate --source_dir *path_to_original_HR_images* --output_dir *path_to_new_folder* --pipeline blur-downscale-jm_compress --scale 4 --random_compression```


### Setting up SR testing Datasets
All SR testing datasets are available for download from the LapSRN main page [here](http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip).  Generate LR versions of each image using the same commands as used for the DIV2K/Flickr2K datasets.    

### Additional Options
Further detail on generating LR data provided in Documentation/data_prep.md.

Training/Evaluating Models
----------------
### Training
To train models, prepare a configuration file (details in Documentation/model_training.md) and run:

```train_sisr --parameters *path_to_config_file*```
### Evaluation
Similarly, for evaluation, prepare an eval config file (details in Documentation/model_eval.md) and run:

```eval_sisr --config *path_to_config_file*```

### Standard SISR models available (code for each adapted from their official repository - linked within source code):
1. [SRCNN](https://arxiv.org/abs/1501.00092)
2. [VDSR](https://arxiv.org/abs/1511.04587)
3. [EDSR](https://arxiv.org/abs/1707.02921)
4. [RCAN](https://arxiv.org/abs/1807.02758)
5. [SPARNet](https://arxiv.org/abs/2012.01211)
6. [SFTMD](https://arxiv.org/abs/1904.03377)
7. [SRMD](https://arxiv.org/abs/1712.06116)
8. [SAN](https://openaccess.thecvf.com/content_CVPR_2019/html/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.html)
9. [HAN](https://arxiv.org/abs/2008.08767)

### Custom models available (all meta-models are marked with a Q-):
1. Q-RCAN (meta-RCAN)
2. Q-EDSR
3. Q-SAN
4. Q-HAN
5. Q-SPARNet
7. Various SFTMD variants (check SFTMD architectures file for options)

### IEEE SPL Pre-Trained Model Weights
All weights for the models presented in our paper are available for download [here](https://doi.org/10.5281/zenodo.5551061).  The models are split into three folders:

- Models trained on **blurry general images**:  These models were all trained on DIV2K/Flickr2K blurred/downsampled images.  These include:
  - SRMD
  - SFTMD
  - RCAN
  - EDSR
  - SAN
  - HAN
  - Meta-RCAN
  - Meta-EDSR
  - Meta-SAN
  - Meta-HAN
- Models trained on **blurry and compressed general images**:  These models were all trained on DIV2K/Flickr2K blurred/downsampled/compressed images.  These include:
  - RCAN
  - Meta-RCAN (accepting blur kernel data only)
  - Meta-RCAN (accepting compression QPI data only)
  - Meta-RCAN (accepting both blur kernels and compression QPI)
- Models trained on **blurry face images**:  These models were all trained on CelebA-HQ blurred/downsampled images.  These include:
  - RCAN
  - SPARNet (note that SPARNET only accepts pre-upsampled images)
  - Meta-RCAN
  - Meta-SPARNet
-  Testing config files for all of these models are available in Documentation/SPL_testing_files.  To use these, you need to first download and prepare the relevant datasets as shown [here](#data-generate).  Place the downloaded model folders in ./Results to use the config files as is, or adjust the ```model_loc``` parameter to point towards the directory containing the models.

Once downloaded, these models can be used directly with the eval command (```eval_sisr``) on any other input dataset as discussed in the evaluation documentation (Documentation/model_eval.md).

### Replicating SPL Results from Scratch

All training config files for models presented in our SPL paper are provided in Documentation/sample_config_files.  These configurations assume that your training/eval data is stored in the relevant directory within ./Data, so please check that you have downloaded and prepared your datasets (as detailed above) before training.

Additional/Advanced Setup
--------------
### <a name="jm-install"></a> Setting up JM (for compressing images)
Download the reference software from [here](http://iphome.hhi.de/suehring/tml/).  Place the software in the directory ```./JM```.  cd into this directory and compile the software using the commands ```. unixprep.sh``` and ```make```.  Some changes might be required for different OS versions.   
To compress images, simply add the ```jm_compress``` argument when specifying ```image_manipulate```'s pipeline.

### Setting up VGGFace (Pytorch)
Download pre-trained weights for the VGGFace model from [here](http://www.robots.ox.ac.uk/~albanie/pytorch-models.html) (scroll to VGGFace).  Place the weights file in the directory ```./external_packages/VGGFace/```.  The weights file should be called ```vgg_face_dag.pth```.

### Setting up lightCNN
Download pre-trained weights for the lightCNN model from [here](https://github.com/AlfredXiangWu/LightCNN) (LightCNN-29 v1).  Place the weights file in the directory ```./external_packages/LightCNN/```.  The weights file should be called ```LightCNN_29Layers_checkpoint.pth.tar```.

Creating Custom Models
------------------------
Information on how to develop and train your own models is available in Documentation/framework_development.md.

Full List of Commands Available
-------------------------------
The entire list of commands available with this repository is:

- train_sisr - main model training function.
- eval_sisr - main model evaluation function.
- image_manipulate - main bulk image converter.
- images_to_video - Helper function to convert a folder of images into a video.
- extract_best_model - Helper function to extract model config and best model checkpoint from a folder to a target location.
- clean_models - Helper function to remove unnecessary model checkpoints.
- model_report - Helper function to report on models available in specified directory.

Each command can be run with the --help parameter, which will print out the available options and docstrings.

Uninstall
-----------------
Simply run:

```pip uninstall Deep-FIR-SR```

from any directory, with the relevant virtual environment activated.

Citation
-----------------
```
@ARTICLE{Meta-Attention,
  author={Aquilina, Matthew and Galea, Christian and Abela, John and Camilleri, Kenneth P. and Farrugia, Reuben A.},
  journal={IEEE Signal Processing Letters}, 
  title={Improving Super-Resolution Performance Using Meta-Attention Layers}, 
  year={2021},
  volume={28},
  number={},
  pages={2082-2086},
  doi={10.1109/LSP.2021.3116518}}
```

License/Further Development
--------------------------
This code has been released via the GNU GPLv3 open-source license.  However, this code can also be made available via an alternative closed, permissive license.  Third-parties interested in this form of licensing should contact us separately.  

Usages of code from other repositories is properly referenced within the code itself.  

We are working on a number of different research tasks in super-resolution, we'll be updating this repo as we make further advancements!

Short-term upgrades planned:
- CI automated testing (alongside Pytest)
- Release of packaged version
- Other upgrades TBA
