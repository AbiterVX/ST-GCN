# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
import numpy as np

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A,Type="Origin"):
        if Type == "Origin":
            assert A.size(0) == self.kernel_size
            # A.shape   [3, 18, 18]   3是分组数。
            # x.shape   batch=16:[32, 64, 150, 18]
            x = self.conv(x)
            n, kc, t, v = x.size()
            x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)  # batch=16:  shape [32, 3, 256, 38, 18][n,k,c,t,v]
            x = torch.einsum('nkctv,kvw->nctw', (x, A))  # kv乘积和
        elif Type == "TemporalWeight":
            x = self.conv(x)
            n, kc, t, v = x.size()
            x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)

            x_current = torch.einsum('nkctv,kvw->tncw', (x, A[1]))
            x_previous = torch.einsum('nkctv,kvw->tncw', (x, A[0]))
            x_next = torch.einsum('nkctv,kvw->tncw', (x, A[2]))


            tempEmpty = torch.zeros_like(x_current[0])
            tempEmpty = tempEmpty.unsqueeze(0)
            x_current = torch.cat((tempEmpty,x_current,tempEmpty), 0)
            x_previous = torch.cat((tempEmpty,tempEmpty,x_previous), 0)
            x_next = torch.cat((x_next,tempEmpty,tempEmpty), 0)

            x = 0.8*x_current + 0.1*x_previous + 0.1*x_next

            x = x.narrow(0,  1, len(x)-2)
            x = x.permute(1, 2, 0, 3)



        return x.contiguous(), A,Type
