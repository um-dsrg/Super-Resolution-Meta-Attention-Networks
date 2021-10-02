from torch import nn


class ParaCALayer(nn.Module):
    """
    Main Meta-Attention module.

    This module accepts as input both the current set of channels within a CNN and a vector containing
    relevant metadata for the image under analysis.  The metadata vector will be used to modulate the network channels,
     using a channel attention scheme.
    """
    def __init__(self, network_channels, num_metadata, nonlinearity=False, num_layers=2):
        """
        :param network_channels: Number of feature channels expected in network.
        :param num_metadata: Metadata vector size.
        :param nonlinearity: Set to True to add ReLUs between each FC layer in the module.
        :param num_layers: Number of fully-connected layers to introduce.  With 2 or more layers, the unit size
        is increased/decreased consistently from the input to the final modulation vector.
        """
        super(ParaCALayer, self).__init__()

        layers = []
        multiplier = num_layers
        inputs = [num_metadata]

        for i in range(num_layers):
            if num_metadata > 15:
                inputs.append((network_channels-num_metadata)//multiplier + num_metadata)
            else:
                inputs.append(network_channels//multiplier)
            layers.append(nn.Conv2d(inputs[i], inputs[i+1], 1, padding=0, bias=True))
            if nonlinearity and multiplier != 1:
                layers.append(nn.ReLU(inplace=True))
            multiplier -= 1

        layers.append(nn.Sigmoid())
        self.attribute_integrator = nn.Sequential(*layers)

    def forward(self, x, attributes):

        y = self.attribute_integrator(attributes)

        return x * y

    def forensic(self, x, attributes):

        y = self.attribute_integrator(attributes)

        return x * y, y.cpu().data.numpy().squeeze()
