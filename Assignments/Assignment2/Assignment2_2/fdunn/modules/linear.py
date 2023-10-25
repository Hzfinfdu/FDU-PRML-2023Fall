"""
Linear Layer

ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from base import Module

class Linear(Module):
    """Applies a linear transformation to the incoming data: :math:`Y = XW^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Parameters:
        W: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        b: the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    """
    def __init__(self, in_features, out_features, bias = True):

        # input and output
        self.input = None
        self.in_features = in_features
        self.out_features = out_features

        # params
        self.params = {}
        k= 1/in_features
        self.params['W'] = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=(out_features,in_features))
        self.params['b'] = None
        if bias:
            self.params['b'] = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k), size=(out_features))

        # grads of params
        self.grads = {}

    def forward(self, input):
        self.input = input
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. math:  Y = XW^T + b                              #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return output

    def backward(self, output_grad):
        """
        Input:
            - output_grad：(*, H_{out})
            partial (loss function) / partial (output of this module)

        Return：
            - input_grad：(*, H_{in})
            partial (loss function) / partial (input of this module)
        """
        ###########################################################################
        # TODO:                                                                   #
        # Calculate and store the grads of self.params['W'] and self.params['b']  #
        # in self.grads['W'] and self.grads['b'] respectively.                    #
        # Calculate and return the input_grad.                                    #
        # Notice: You have to deal with high-dimensional tensor inputs            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        assert self.grads['W'].shape == self.params['W'].shape
        assert self.grads['b'].shape == self.params['b'].shape
        assert input_grad.shape == self.input.shape

        return input_grad

def unit_test():
    np.random.seed(2333)

    model = Linear(20,30)
    input = np.random.randn(4, 2, 8, 20)
    output = model(input)
    print (output.shape)

    output_grad = output
    input_grad = model.backward(output_grad)
    print (model.grads['W'].shape)
    print (model.grads['b'].shape)
    print (input_grad.shape)

if __name__ == '__main__':
    unit_test()