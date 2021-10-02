import os
import numpy as np
import PIL.Image
import h5py
from tqdm import tqdm
from io import BytesIO
import skvideo.io
import subprocess
import click
import sys
import random
import torch
from torchvision import transforms
import pickle
from collections import defaultdict
import pandas as pd

import sr_tools.gaussian_utils as g_utils
import SISR.configuration.constants as sconst
from sr_tools.image_manipulation import downsample
from sr_tools.helper_functions import extract_image_names_from_folder, create_dir_if_empty, generate_range


class ConversionHub:
    def __init__(self, ref_dir, out_dir, recursive=False):

        self.image_names = []
        self.ref_dir = ref_dir
        self.out_dir = out_dir
        self.lossless_extension = '.png'
        create_dir_if_empty(out_dir)

        if os.path.isdir(ref_dir):
            self.image_names = extract_image_names_from_folder(ref_dir, recursive=recursive)
        elif os.path.isfile(ref_dir):
            self.image_names = [ref_dir]
        else:
            raise RuntimeError('Please provide a valid filename/folder.')

        self.base_names = [os.path.splitext(os.path.basename(loc))[0] for loc in self.image_names]

        self.function_library = {
            'jm_compress': (self.jm_compress, 'QPI'),
            'jpeg_compress': (self.jpeg_compress, 'jpeg_quality'),
            'downscale': (self.downscale, None),
            'upscale': (self.upscale, None),
            'blur': (self.blur, 'blur_kernel')
        }

        self.save_info_buffer = defaultdict(list)

    def create_h5(self, h5_filename):  # TODO: fix this function to match rest of framework (currently unusable)
        """
        Creates an h5 file from given images (images must all be of the same size)
        :param h5_filename: h5 filename to save
        :return: None
        """
        # Alternative option if upgrading:
        # https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/h5tool.py

        dummy_file = np.array(PIL.Image.open(self.image_names[0]))
        h, w, c = dummy_file.shape  # extract height, width and channels

        with h5py.File(os.path.join(self.out_dir, h5_filename), 'w') as h5_file:
            images = np.zeros((len(self.image_names), h, w, c))
            dataset = h5_file.create_dataset("images", np.shape(images), 'uint8')

            # loop directly saves image to the h5 file
            for idx in tqdm(range(len(self.image_names))):
                dataset[idx, :, :, :] = np.asarray(PIL.Image.open(self.image_names[idx]))

    def _blur_setup(self, blur_sig):
        batch_ker = g_utils.random_batch_kernel(batch=30000, tensor=False)
        b = np.size(batch_ker, 0)
        batch_ker = batch_ker.reshape((b, -1))
        pca_matrix = g_utils.PCA(batch_ker, k=10).float()
        torch.save(pca_matrix, os.path.join(self.out_dir, 'pca_matrix.pth'))
        prepro = g_utils.SRMDPreprocessing(pca_matrix, random=True if blur_sig is None else False,
                                           noise=False, cuda=False, noise_high=0.0, sig=blur_sig)
        return prepro

    def _jm_setup(self):

        # creation of temporary locations for video transfer to JM and back
        temp_yuv_loc = os.path.join(self.out_dir, 'vid_temp.yuv')
        temp_comp_loc = os.path.join(self.out_dir, 'vid_comp.yuv')
        temp_h264_loc = os.path.join(self.out_dir, 'vid_comp.h264')
        temp_stats_loc = os.path.join(self.out_dir, 'comp_stats.dat')
        temp_leakybucket = os.path.join(self.out_dir, 'leakybucketparam.cfg')
        temp_data = os.path.join(os.getcwd(), 'data.txt')
        temp_log = os.path.join(os.getcwd(), 'log.dat')

        # Setting up constant JM params
        jm_params = {'InputFile': temp_yuv_loc, 'OutputFile': temp_h264_loc,
                     'ReconFile': temp_comp_loc, 'StatsFile': temp_stats_loc,
                     'LeakyBucketParamFile': temp_leakybucket,
                     'NumberBFrames': 0, 'IDRPeriod': 1, 'IntraPeriod': 1, 'QPISlice': 0,
                     'SourceHeight': 0, 'SourceWidth': 0, 'FramesToBeEncoded': 1}

        jm_bin = os.path.join(os.path.dirname(sconst.base_directory), 'JM/bin')

        jm_command = jm_bin + '/lencod.exe -d ' + jm_bin + '/encoder_baseline.cfg'

        for key, val in jm_params.items():
            jm_command += ' -p ' + str(key) + '=' + str(val)

        return jm_command, [temp_yuv_loc, temp_comp_loc, temp_h264_loc, temp_stats_loc, temp_leakybucket, temp_data, temp_log]

    def _jm_cleanup(self, locations):
        for location in locations:
            os.remove(location)

    def run_conversion(self, pipeline, blur_sig, skip_images=False, **kwargs):

        if 'jm_compress' in pipeline:
            jm_command, temp_locations = self._jm_setup()
            kwargs = {**kwargs, **{'jm_command': jm_command, 'yuv_loc': temp_locations[0],
                                   'comp_loc': temp_locations[1], 'jm': True}}
        if 'blur' in pipeline:
            blur_system = self._blur_setup(blur_sig)
            kwargs = {**kwargs, **{'blur_system': blur_system}}

        diag_string = 'Running conversion with the following pipeline: input '
        metadata = ['image']
        for command in pipeline:
            diag_string += '-> %s ' % command
            c_data = self.function_library[command][1]  # extracting expected metadata
            if c_data is not None:
                metadata.append(c_data)
        print(diag_string)

        saveable_metadata = defaultdict(list)  # pd.DataFrame(columns=metadata)

        for index, image_name in enumerate(tqdm(self.image_names)):
            flux_images = [PIL.Image.open(image_name)]

            self.save_info_buffer = defaultdict(list)  # temporary buffer for storing extracted metadata

            for func in pipeline:  # Running all functions in provided pipeline
                flux_images = self.function_library[func][0](*flux_images, **kwargs)

            # file saving and information extraction
            # TODO: I still haven't found a way to handle multiple images being produced from a single input image.
            #  I have had to use the rather verbose patch below to fix the current issues.
            #  Need to find a more elegant way to handle all exceptions.
            #  Also, it should be possible to convert all functions to being static rather than depending on self.
            if len(flux_images) == 1:
                base_name = self.base_names[index] + self.lossless_extension
                saveable_metadata['image'].append(base_name)
                for key, val in self.save_info_buffer.items():
                    saveable_metadata[key].append(val[0])
                if not skip_images:
                    flux_images[0].save(os.path.join(self.out_dir, base_name))
            else:
                for flux_index, final_image in enumerate(flux_images):
                    base_name = self.base_names[index] + '_q%d' % flux_index + self.lossless_extension
                    saveable_metadata['image'].append(base_name)
                    for key, val in self.save_info_buffer.items():
                        if len(val) == 1:
                            saveable_metadata[key].append(val[0])
                        else:
                            saveable_metadata[key].append(val[flux_index])
                    if not skip_images:
                        final_image.save(os.path.join(self.out_dir, base_name))

        if len(metadata) > 1:
            if 'scaled_landmarks' in saveable_metadata:
                landmarks_dict = {'name': saveable_metadata['image'],
                                  'landmarks': saveable_metadata.pop('scaled_landmarks')}
                pickle.dump(landmarks_dict, open(os.path.join(self.out_dir, 'scaled_landmarks.pkl'), 'wb'))

            saveable_metadata = pd.DataFrame.from_dict(saveable_metadata).set_index(['image'])
            saveable_metadata.to_csv(os.path.join(self.out_dir, 'degradation_metadata.csv'))

        if 'jm_compress' in pipeline:
            self._jm_cleanup(temp_locations)

    def jpeg_compress(self, *images, jpeg_quality=60, **kwargs):
        """
        Compresses data into jpg images
        :param images: list/set/single PIL image/s
        :param jpeg_quality: JPEG compression quality value (higher = better quality, max 100)
        :param names: Corresponding image names
        :return: None
        """
        compressed_images = []
        for index, image in enumerate(images):
            buffer = BytesIO()  # TODO: check if subsampling needs to be changed for JPEG....
            image.save(buffer, "JPEG", subsampling=0, quality=jpeg_quality)  # main compression step
            buffer.seek(0)
            compressed_images.append(PIL.Image.open(buffer))
            self.save_info_buffer['jpeg_quality'].append(jpeg_quality)

        return compressed_images

    def jm_compress(self, *images, jm_command, yuv_loc, comp_loc, verbose=False, jm_qpi=28, compression_range=(10, 50),
                    random_compression=False, qpi_divisions=None, **kwargs):
        compressed_images = []
        if random_compression:
            if qpi_divisions is not None:
                qpi_list = []
                splits = generate_range(compression_range, qpi_divisions)
                for _ in images:
                    selected_qpi = [random.randint(splits[index], splits[index+1]-1) for index, _ in enumerate(splits[:-1])]
                    qpi_list.append(selected_qpi)
            else:
                qpi_list = [[random.randint(*compression_range)] for _ in images]
        else:
            qpi_list = [[jm_qpi] for _ in images]

        init_params = 'QPISlice=%d -p SourceHeight=%d -p SourceWidth=%d' % (0, 0, 0)

        for index, image in enumerate(images):
            # Adjusting JM command for image dimensions and selected QPI
            l_w, l_h = image.size
            vid = skvideo.utils.vshape(np.array(image))
            # convert im to YUV file
            skvideo.io.vwrite(yuv_loc, vid, verbosity=1 if verbose else 0, outputdict={"-pix_fmt": "yuv420p"})

            for qpi in qpi_list[index]:
                new_params = 'QPISlice=%d -p SourceHeight=%d -p SourceWidth=%d' % (qpi, l_h, l_w)
                command_full = jm_command.replace(init_params, new_params)

                # Perform JM compression
                process = subprocess.Popen(command_full,
                                           shell=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

                stdout, stderr = process.communicate()
                if verbose:
                    print('%%%%%%%%%%%%%%%%%')
                    print('JM Error Output:')
                    print(stderr)
                    print('%%%%%%%%%%%%%%%%%')

                # Restore image from video
                vid = skvideo.io.vread(comp_loc, height=l_h, width=l_w,
                                       inputdict={"-pix_fmt": "yuv420p"}, verbosity=1 if verbose else 0)
                # recover resulting image to PIL array
                output = PIL.Image.fromarray(vid[0, :, :, :].astype(np.uint8))
                compressed_images.append(output)
                self.save_info_buffer['QPI'].append(qpi)

        return compressed_images

    def blur(self, *images, blur_system, save_original_kernel=False, **kwargs):
        blurred_images = []
        for index, image in enumerate(images):
            tensor_image, kernel_map, unreduced_kernel = blur_system(transforms.ToTensor()(image))
            pil_image = transforms.ToPILImage()(tensor_image.squeeze(0).cpu())
            self.save_info_buffer['blur_kernel'].append(kernel_map.cpu().tolist()[0])
            if save_original_kernel:
                unreduced_kernel = unreduced_kernel.numpy().squeeze().flatten()
                self.save_info_buffer['unmodified_blur_kernel'].append(unreduced_kernel.tolist())
            blurred_images.append(pil_image)
        return blurred_images

    def downscale(self, *images, scale=4, jm=False, **kwargs):
        downsampled_images = []
        for image in images:
            _, lr_im = downsample(image, scale=scale, jm=jm)
            downsampled_images.append(lr_im)
        return downsampled_images

    def upscale(self, *images, scale=4, **kwargs):
        upsampled_images = []
        for image in images:
            hr_im = image.resize((image.width * scale, image.height * scale), resample=PIL.Image.BICUBIC)
            upsampled_images.append(hr_im)
        return upsampled_images

@click.command()
@click.option("--source_dir", default=sconst.data_directory,
              help='Input directory to source images.')
@click.option("--output_dir", default=os.path.join(sconst.data_directory, 'new_images'),
              help='Output directory to save new images.')
@click.option("--pipeline", default='downscale-jm_compress', help='Pipeline of operations to perform, separated by "-". '
                                                                  'Available operations: jm_compress, '
                                                                  'jpeg_compress, downscale, upscale, blur',
              show_default=True)
@click.option("--seed", default=8,
              help='Random seed.')
@click.option("--scale", default=4,
              help='Scale to use when downsampling.  Default 4.')
@click.option("--jm_qpi", default=28,
              help='Quality value for JM compression.  Higher is worse (up to 51).  Default 28.')
@click.option("--verbose", default=False,
              help='Turn on/off JM output verbosity.  Default off.')
@click.option("--random_compression", is_flag=True,
              help='Set this flag to compress images with random QPI values.')
@click.option("--compression_range", default=(20, 40), type=(int, int),
              help='QPI compression range to extract possible QPI from during compression.')
@click.option("--qpi_divisions", type=int, help='Number of divisions to create from compression range '
                                                '(i.e. number of compressed images to create per input image)')
@click.option("--jpeg_quality", default=60,
              help='Quality value for jpeg compression.  Higher is better (up to 100).  Default 60.')
@click.option("--h5_filename", default='images.h5',
              help='H5 filename.')
@click.option('--blur_sig', default=None, type=float,
              help='Blur kernel width.  Specify this if random blur kernel selection is not required.')
@click.option('--save_original_kernel', default=False, is_flag=True,
              help='Set this parameter to save the original blur kernel used, without any reductions or modifications.')
@click.option('--recursive', is_flag=True,
              help='Set this flag to signal data converter to seek out all images in all sub-directories of '
                   'directory specified.')
@click.option('--skip_images', default=False, is_flag=True,
              help='Set this flag to skip all image saving, and instead only save metadata generated.')
def manipulation_hub(source_dir, output_dir, pipeline, seed, recursive, **kwargs):
    """
    Main function for degrading and preparing images for SR.
    """
    # TODO: find a way to implement multiprocessing for compression systems...
    # TODO: fix up arguments to be more intuitive

    random.seed(seed)
    if 'blur' in pipeline:  # TODO: check if this is necessary
        g_utils.set_random_seed(seed)

    if kwargs['jm_qpi'] > 51 or kwargs['compression_range'][1] > 51:
        raise RuntimeError('QPI cannot be larger than 51.')

    pipeline = pipeline.split('-')
    converter = ConversionHub(source_dir, output_dir, recursive=recursive)
    converter.run_conversion(pipeline, **kwargs)


if __name__ == '__main__':
    manipulation_hub(sys.argv[1:])  # for use when debugging with pycharm

