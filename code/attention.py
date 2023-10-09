import math 
from einops import rearrange, repeat
from functools import partial

import torch
import torch.nn as nn

from layers import Conv1d
from hparam import HPDurationExtractor as hp

from functional import scaled_dot_attention
from attn_utils import default, empty, exists
from attn_utils import gaussian_orthogonal_random_matrix
from attn_utils import linear_attention, softmax_kernel


class ScaledDotAttention(nn.Module):
    """Scaled dot attention with positional encoding preconditioning"""

    def __init__(self):
        super(ScaledDotAttention, self).__init__()

        self.noise = hp.att_noise
        self.fc_query = Conv1d(hp.channels, hp.att_hidden_channels)
        self.fc_keys = Conv1d(hp.channels, hp.att_hidden_channels)

        # share parameters
        self.fc_keys.weight = torch.nn.Parameter(self.fc_query.weight.clone())
        self.fc_keys.bias = torch.nn.Parameter(self.fc_query.bias.clone())

        self.fc_values = Conv1d(hp.channels, hp.att_hidden_channels)
        self.fc_out = Conv1d(hp.att_hidden_channels, hp.channels)

    def forward(self, q, k, v, mask=None):
        """
        :param q: queries, (batch, time1, channels1)
        :param k: keys, (batch, time2, channels1)
        :param v: values, (batch, time2, channels2)
        :param mask: boolean mask, (batch, time1, time2)
        :return: (batch, time1, channels2), (batch, time1, time2)
        """

        noise = self.noise if self.training else 0

        alignment, weights = scaled_dot_attention(self.fc_query(q),
                                                  self.fc_keys(k),
                                                  self.fc_values(v),
                                                  mask, noise=noise)
        alignment = self.fc_out(alignment)
        return alignment, weights
    

class FastAttention(nn.Module):
    def __init__(
            self, dim_heads, nb_features = None, ortho_scaling = 0,
            generalized_attention = False, kernel_fn = nn.ReLU()):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v, mask=None):
        device = q.device

        q = q[None, :]
        k = k[None, :]

        create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
        q = create_kernel(q, is_query = True)
        k = create_kernel(k, is_query = False)

        weights = torch.matmul(q, k.transpose(2, 3))

        if mask is not None:
            weights = torch.squeeze(weights, 0)

        alignment = linear_attention(q, k, v)
        alignment = torch.squeeze(alignment, 0)
        alignment.detach()

        # try:
        #     print(round(torch.mps.current_allocated_memory() / torch.mps.driver_allocated_memory(), 2)*100, '%')
        # except:
        #     pass

        return alignment, weights