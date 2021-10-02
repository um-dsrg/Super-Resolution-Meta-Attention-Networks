import os

base_directory = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
results_directory = os.path.join(os.path.dirname(base_directory), 'Results')
data_directory = os.path.join(os.path.dirname(base_directory), 'Data')
scratch_directory = os.path.join(os.path.dirname(base_directory), 'Scratch')

# Dataset Splits
data_splits = {'celeba': {'train': (0, 162770),
                          'eval': (162770, 182637),
                          'test': (182637, 202599)},
               'div2k': {'train': (0, 800),
                         'eval': (800, 900)},
               'flickr2k': {'train': (0, 2650)}}

# Other Configs
temp_dump = os.path.join(results_directory, 'temp')  # Temporary image dump
vggface_weights = os.path.join(base_directory, 'Code/external_packages/VGGFace/vgg_face_dag.pth')
lightcnn_weights = os.path.join(base_directory, 'Code/external_packages/LightCNN/LightCNN_29Layers_checkpoint.pth.tar')
