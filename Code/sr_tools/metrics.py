import numpy as np
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim


def psnr(img1, img2, max_value=255.0):
    """
    Calculates peak signal-to-noise ratio (PSNR) between two images.
    :param img1: 2D image array (numpy or torch)
    :param img2: 2D image array (numpy or torch)
    :param max_value: maximum pixel value for arrays
    :return:
    """
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


class Metrics:
    """
    Main metrics class that takes care of all necessary calculations for images specified.
    """
    def __init__(self, metrics, delimeter='-', **kwargs):
        """
        :param metrics: List of metrics to calculate.
        :param delimeter: Delimeter to use when outputting results.
        """

        self.metrics = metrics
        self.delimeter = delimeter

    def run_image_metric(self, metric, im_a, im_ref=None, single_values=False, max_value=1, multichannel=False):
        """
        Main metric calculation function.
        :param metric: Metric to be calculated (string).
        :param im_a: Batch of query images.
        :param im_ref: Batch of reference images.
        :param single_values: Request metric result for each provided image individually.
        :param max_value: Max image pixel value.
        :param multichannel: Calculate metrics using all provided channels.
        :return: Metric results.
        """
        if len(im_a.shape) == 3:  # ensure N, C, H, W format
            im_a = np.expand_dims(im_a, axis=0)

        if im_ref is not None and len(im_ref.shape) == 3:
            im_ref = np.expand_dims(im_ref, axis=0)

        if metric == 'PSNR':  # TODO: multichannel could also be relevant to single values area....
            if im_ref is None:
                raise Exception('Need a reference to calculate PSNR.')
            if single_values:
                indiv_psnr = []
                for ind in range(im_a.shape[0]):
                    indiv_psnr.append(psnr(im_a[ind, 0, :, :], im_ref[ind, 0, :, :], max_value=max_value))
                return indiv_psnr
            else:
                if multichannel:
                    return psnr(im_a, im_ref, max_value=max_value)
                else:
                    return psnr(im_a[:, 0, :, :], im_ref[:, 0, :, :], max_value=max_value)

        elif metric == 'SSIM':
            if im_ref is None:
                raise Exception('Need a reference to calculate SSIM.')

            if multichannel:
                im_a = im_a.transpose((0, 2, 3, 1))
                im_ref = im_ref.transpose((0, 2, 3, 1))

                ssim_vals = []
                for i in range(im_a.shape[0]):
                    ssim_vals.append(ssim(im_a[i, :], im_ref[i, :], data_range=max_value, gaussian_weights=True,
                                          use_sample_covariance=False, sigma=1.5, multichannel=True))
                return sum(ssim_vals)/len(ssim_vals)

            else:
                im_a = im_a.transpose((1, 2, 3, 0))[0, :]
                im_ref = im_ref.transpose((1, 2, 3, 0))[0, :]

                if single_values:
                    indiv_ssim = []
                    for ind in range(im_a.shape[-1]):
                        indiv_ssim.append(ssim(im_a[..., ind], im_ref[..., ind], data_range=max_value, gaussian_weights=True,
                                               use_sample_covariance=False, sigma=1.5))
                    return indiv_ssim
                else:
                    return ssim(im_a, im_ref, data_range=max_value, gaussian_weights=True,
                                use_sample_covariance=False, sigma=1.5, multichannel=True)

    def run_metrics(self, images, references=None, key='',
                    metrics=None, probe_names=None, max_value=1):
        """
        Function that runs multiple metrics for images specified.
        :param images: Images to evaluate.
        :param references: Reference images.
        :param key: Additional key to append to results dictionary.
        :param metrics: Metrics to calculate.
        :param probe_names: Image names.
        :param max_value: Images maximum pixel value.
        :return: Dictionary of results, and a quick diagnostic string.
        """
        #TODO: also accept rgb images to prevent double-take for Face Rec Performance
        if metrics is None:
            metrics = self.metrics
        diag_string = ''
        output = defaultdict(list)

        for metric in metrics:
            value = self.run_image_metric(metric, images, references, max_value=max_value,
                                                     single_values=True,)
            if type(value) == dict:
                for metric_key in value.keys():
                    output['%s%s%s' % (key, self.delimeter, metric)] = value[metric_key]
            else:
                output['%s%s%s' % (key, self.delimeter, metric)] = value
            if metric.upper() == 'PSNR':
                diag_string = '{} {}: {:.4f}, '.format(key, metric, np.average(value))

        return output, diag_string
