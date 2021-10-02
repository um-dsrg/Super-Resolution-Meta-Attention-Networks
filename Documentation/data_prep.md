SISR Data Preparation
======================

## Details
SISR data preparation can be mostly facilitated through the ```image_manipulate``` function.  This function systematically applies a given image conversion pipeline to all images within a provided folder.  All possible parameters for this function are provided below:
### General Parameters
- ```source_dir``` - Input directory from which to source images.
- ```output_dir``` - Output directory within which to save new images.  Final directory name does not need to be created beforehand.
- ```pipeline``` - Pipeline of operations to perform, separated by "-".  Available operations include:
  - jm_compress - JM H.264 compression.  Requires JM to be installed (see main documentation for details)
  - jpeg_compress - JPEG compression.
  - downscale - Bicubic downscaling.
  - upscale - Bicubic upscaling.
  - blur - Gaussian blur (currently only uses Isotropic Gaussians).
- ```seed``` - Random seed used to initialize libraries.  Default 8.
- ```scale``` - Scale to use when downsampling.  Default 4.
- ```h5_filename``` - H5 filename (WIP).
- ```recursive``` - Set this flag to signal data converter to seek out all images in all sub-directories of main directory specified.
- ```skip_images``` - Set this flag to not save any images but only generate metadata.
### JM-Specific Parameters
- ```jm_qpi``` - Quality value for JM compression.  Higher is worse (up to 51).  Default 28.
- ```verbose``` - Turn on/off JM output verbosity.  Default off.
- ```random_compression``` - Set this flag to compress images with random QPI values.
- ```compression_range``` - QPI compression range to extract possible QPI from during compression.
- ```qpi_divisions``` - Number of divisions to create from compression range (i.e. number of compressed images to create per input image).
### JPEG-Specific Parameters
- ```jpeg_quality``` - Quality value for jpeg compression.  Higher is better (up to 100).  Default 60.
### Blurring Parameters (code for this function based on that of [IKC](https://github.com/yuanjunchai/IKC))
- ```blur_sig``` - Blur kernel width.  Specify this if random blur kernel selection is not required.  Default sigma minimum is 0.2 and maximum is 4.0.
- ```save_original_kernel``` - Set to save entire blur kernel apart from PCA'd version.

## Examples

To blur, downsample then compress a set of images with a randomly selected blur kernel and JM Compression QPI, the command would be specified as follows:

```image_manipulate --pipeline blur-downscale-jm_compress --random_compression --source_dir *path_to_input_folder* --output_dir *path_to_output*```

To simply upscale some images by a constant scale factor (e.g. x8), the command would be:

```image_manipulate --pipeline upscale --scale 8 --source_dir *path_to_input_folder* --output_dir *path_to_output*```
