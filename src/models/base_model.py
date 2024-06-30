import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod # To be implemented by child classes.
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()
    
        #### TODO #######################################
        # Print the number of **trainable** parameters  #
        # by appending them to ret_str                  #
        #################################################
        
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                ret_str += f'\n{name}: {num_params}'
                total_params += num_params
        ret_str += f'\nTotal Trainable Parameters: {total_params}'
        
        
        return ret_str