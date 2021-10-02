""" Setup script for package. """
from setuptools import setup, find_packages

setup(
    name="Deep-FIR-SR",
    author="Matthew Aquilina",
    version='1.0',
    url="https://github.com/um-dsrg/Super-Resolution-Meta-Attention-Networks",
    description="SR package containing functionality for creating, "
                "training and validating a variety of SR models.",
    packages=find_packages(),
    include_package_data=True,
    entry_points='''
        [console_scripts]
        train_sisr=SISR.net_train:experiment_setup
        eval_sisr=SISR.net_eval:eval_run
        image_manipulate=sr_tools.data_converter:manipulation_hub
        images_to_video = sr_tools.helper_functions:click_image_sequence_to_movie
        extract_best_model = sr_tools.helper_functions:extract_best_models
        clean_models = sr_tools.helper_functions:click_clean
        model_report = sr_tools.helper_functions:model_compare
    ''',
)

