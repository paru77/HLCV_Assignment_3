import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        super(ConvNet, self).__init__()

        ############## TODO ###############################################
        # Initialize the different model parameters from the config file  #
        # (basically store them in self)                                  #
        ###################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.drop_prob = drop_prob
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module if drop_prob > 0          #
        # Do NOT add any softmax layers.                                                #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        in_channels=self.input_size
        for out_channels in self.hidden_layers[:-1]:
          layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
          if self.norm_layer:
            layers.append(nn.BatchNorm2d(out_channels))
          layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
          layers.append(self.activation())
          if self.drop_prob > 0:
            layers.append(nn.Dropout(self.drop_prob))
          in_channels = out_channels

        self.conv_layers=nn.Sequential(*layers)
        self.fc = nn.Linear(self.hidden_layers[-1], self.num_classes)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        filters = self.conv_layers[0].weight.data.cpu().numpy()
        fig, axarr = plt.subplots(filters.shape[0] // 8, 8, figsize=(10, 10))
        for idx in range(filters.shape[0]):
            ax = axarr[idx // 8, idx % 8]
            filter_img = self._normalize(filters[idx].transpose(1, 2, 0))
            ax.imshow(filter_img)
            ax.axis('off')
        plt.show()

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass computations                             #
        # This can be as simple as one line :)
        # Do not apply any softmax on the logits.                                   #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        
        out = None
        x=self.conv_layers(x)
        x=x.view(x.size(0),-1)
        out=self.fc(x)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
