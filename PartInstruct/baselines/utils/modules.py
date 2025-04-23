import math
import numpy as np
from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# >>> fn = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=8, p2=8)

import robomimic
from robomimic.models.base_nets import RNN_Base, Randomizer
import torch.distributions as D

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

USE_GPU = torch.cuda.is_available()
DEVICE = TorchUtils.get_torch_device(try_to_use_cuda=True)
def safe_cuda(x):
    if USE_GPU:
        return x.cuda()
    return x

def get_activate_fn(activation):
    if activation == 'relu':
        activate_fn = torch.nn.ReLU
    elif activation == 'leaky-relu':
        activate_fn = torch.nn.LeakyReLU
    return activate_fn

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = torch.nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(torch.from_numpy(np.linspace(-1, 1, in_w)).float(),
                                      torch.from_numpy(np.linspace(-1, 1, in_h)).float())

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h
        
    def forward(self, x):
        assert(x.shape[1] == self._in_c)
        assert(x.shape[2] == self._in_h)
        assert(x.shape[3] == self._in_w)

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)
        attention = F.softmax(h, dim=-1)

        keypoint_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True).view(-1, self._num_kp)
        keypoint_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True).view(-1, self._num_kp)

        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints

class ResnetConv(torch.nn.Module):
    def __init__(self,
                 pretrained=False,
                 no_training=False,
                 activation='relu',
                 remove_layer_num=2,
                 img_c=3,
                 last_c=None,
                 no_stride=False):

        super().__init__()

        assert(remove_layer_num <= 5)
        # For training policy
        layers = list(torchvision.models.resnet18(pretrained=pretrained).children())[:-remove_layer_num]
        if img_c != 3:
            # If use eye_in_hand, we need to increase the channel size
            conv0 = torch.nn.Conv2d(img_c, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            layers[0] = conv0

        self.no_stride = no_stride
        if self.no_stride:
            layers[0].stride = (1, 1)
            layers[3].stride = 1
        self.resnet18_embeddings = torch.nn.Sequential(*layers)

        if no_training:
            for param in self.resnet18_embeddings.parameters():
                param.requires_grad = False

        self.remove_layer_num = remove_layer_num

    def forward(self, x):
        h = self.resnet18_embeddings(x)
        return h

    def output_shape(self, input_shape):
        assert(len(input_shape) == 3)
        
        if self.remove_layer_num == 2:
            out_c = 512
            scale = 32.
        elif self.remove_layer_num == 3:
            out_c = 256
            scale = 16.
        elif self.remove_layer_num == 4:
            out_c = 128
            scale = 8.
        elif self.remove_layer_num == 5:
            out_c = 64
            scale = 4.

        if self.no_stride:
            scale = scale / 4.
        out_h = int(math.ceil(input_shape[1] / scale))
        out_w = int(math.ceil(input_shape[2] / scale))
        return (out_c, out_h, out_w)

class ResnetKeypoints(nn.Module):
    def __init__(self,
                 input_shape,
                 pretrained=False,
                 no_training=False,
                 activation='relu',
                 remove_layer_num=2,
                 num_kp=32,
                 visual_feature_dimension=64):
        super().__init__()
        self._resnet_conv = ResnetConv(pretrained=pretrained,
                                       no_training=no_training,
                                       activation=activation,
                                       remove_layer_num=remove_layer_num,
                                       img_c=input_shape[0])

        self._resnet_output_shape = self._resnet_conv.output_shape(input_shape)
        self._spatial_softmax = SpatialSoftmax(in_c=self._resnet_output_shape[0],
                                               in_h=self._resnet_output_shape[1],
                                               in_w=self._resnet_output_shape[2],
                                               num_kp=num_kp)
        self._visual_feature_dimension = visual_feature_dimension
        self._fc = torch.nn.Sequential(torch.nn.Linear(num_kp * 2, visual_feature_dimension))

    def forward(self, x):
        out = self._resnet_conv(x)
        out = self._spatial_softmax(out)
        out = self._fc(out)
        return out

    def output_shape(self, input_shape):
        return (self._resnet_output_shape[0], self._visual_feature_dimension)
        
    
# Simple components for building transformer model
    
class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head_output=64, dropout=0.):
        super().__init__()
        
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = dim_head_output ** (-0.5) 
        self.attention_fn = nn.Softmax(dim=-1)
        self.linear_projection_kqv = nn.Linear(dim, num_heads * dim_head_output * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(nn.Linear(num_heads * dim_head_output, dim), nn.Dropout(dropout))


    def forward(self, x):
        # qkv should be (..., seq_len, num_heads * dim_head_output)
        qkv = self.linear_projection_kqv(x).chunk(3, dim=-1)

        # We need to convert to (..., num_heads, seq_len, dim_head_output)
        # By doing this operation, we assume q, k, v have the same dimension. But this is not necessarily the case
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # q.dot(k.transpose)
        qk_dot_product = torch.matmul(q, k.transpose(-1, -2)) * self.att_scale
        self.att_weights = self.attention_fn(qk_dot_product)

        # (..., num_heads, seq_len, dim_head_output)
        out = torch.matmul(self.att_weights, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        # Merge the output from heads to get single vector
        return self.output_layer(out)

class SelfAttentionMasked(nn.Module):
    def __init__(self, dim, num_heads=2, dim_head_output=64, dropout=0.):
        super().__init__()
        
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = dim_head_output ** (-0.5) 
        self.attention_fn = nn.Softmax(dim=-1)
        self.linear_projection_kqv = nn.Linear(dim, num_heads * dim_head_output * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(nn.Linear(num_heads * dim_head_output, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        # qkv should be (..., seq_len, num_heads * dim_head_output)
        qkv = self.linear_projection_kqv(x).chunk(3, dim=-1)
        # We need to convert to (..., num_heads, seq_len, dim_head_output)
        # By doing this operation, we assume q, k, v have the same dimension. But this is not necessarily the case
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        # q.dot(k.transpose)
        qk_dot_product = torch.matmul(q, k.transpose(-1, -2)) * self.att_scale
        # import ipdb; ipdb.set_trace()
        qk_dot_product = qk_dot_product.masked_fill(mask==1., -torch.inf)
        self.att_weights = self.attention_fn(qk_dot_product)
        # (..., num_heads, seq_len, dim_head_output)
        out = torch.matmul(self.att_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # Merge the output from heads to get single vector
        return self.output_layer(out)

class TransformerFeedForwardNN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # Remember the residual connection
        self.layers = [nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)]
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class TransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 num_layers,
                 num_heads,
                 dim_head_output,
                 mlp_dim,
                 dropout,
                 # position_embedding_type,
                 **kwargs):
        super().__init__()

        # self.position_embedding_fn = get_position_embedding(position_embedding_type, **kwargs)
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

        self.attention_output = {}
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Norm(input_dim),
                SelfAttention(input_dim, num_heads=num_heads, dim_head_output=dim_head_output, dropout=dropout),
                Norm(input_dim),
                TransformerFeedForwardNN(input_dim, mlp_dim, dropout=dropout)
                ]))

            self.attention_output[_] = None

    def forward(self, x):
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            x = x + drop_path(att(att_norm(x)))
            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 num_layers,
                 num_heads,
                 dim_head_output,
                 mlp_dim,
                 dropout,
                 T=1,
                 # position_embedding_type,
                 **kwargs):
        super().__init__()

        # self.position_embedding_fn = get_position_embedding(position_embedding_type, **kwargs)
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

        self.attention_output = {}
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Norm(input_dim),
                SelfAttentionMasked(input_dim, num_heads=num_heads, dim_head_output=dim_head_output, dropout=dropout),
                Norm(input_dim),
                TransformerFeedForwardNN(input_dim, mlp_dim, dropout=dropout)
                ]))

            self.attention_output[_] = None

        self.seq_len = None
        self.num_elements = None

    def compute_mask(self, input_shape):
        if self.num_elements is None or self.seq_len is None or self.num_elements != input_shape[2] or self.seq_len != input_shape[1]:
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (torch.triu(torch.ones(self.seq_len, self.seq_len)) - torch.eye(self.seq_len, self.seq_len)).to(DEVICE)
            # self.mask = self.original_mask.repeat_interleave(self.num_elements, dim=-1).repeat_interleave(self.num_elements, dim=-2)
            self.mask = self.original_mask
            # import ipdb; ipdb.set_trace()
        
    def forward(self, x, mask=None):
        assert(mask is not None or self.mask is not None)
        if mask is not None:
            self.mask = mask
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            x = x + drop_path(att(att_norm(x), self.mask))
            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x
    
class TemporalSinusoidalPositionEncoding(nn.Module):
    def __init__(self,
                 input_shape,
                 inv_freq_factor=10,
                 factor_ratio=None):
        super().__init__()
        self.input_shape = input_shape
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_shape[-1]
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels))
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)
        
    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        # emb = torch.zeros((x.shape[1], self.channels), device=x.device).type(x.type())
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape

class TemporalZeroPositionEncoding(nn.Module):
    def __init__(self,
                 input_shape,
                 inv_freq_factor=10,
                 factor_ratio=None):
        super().__init__()
        self.input_shape = input_shape
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_shape[-1]
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels))
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)
        
    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        # print(pos_x.shape)
        # print(self.inv_freq.shape)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * 0.

    def output_shape(self, input_shape):
        return input_shape
    
class PatchBBoxSinsuisoidalPositionEncoding(torch.nn.Module):
    def __init__(
            self,
            input_shape,
            scaling_ratio=128.,
            dim=4,
            factor_ratio=1.,
            num_proposals=20,
    ):
        super().__init__()
        self.input_shape = input_shape
        channels = self.input_shape[1]
        channels = int(np.ceil(channels / (dim * 2)) * dim)
        self.channels = channels
        inv_freq = 1.0 / (10 ** (torch.arange(0, channels, dim).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        
        x_steps = int(np.sqrt(num_proposals)) + 1
        y_steps = num_proposals // (x_steps - 1) + 1

        px = torch.linspace(0, 128, steps=x_steps)
        py = torch.linspace(0, 128, steps=y_steps)
        bbox_list = []
        for j in range(len(py) - 1):
            for i in range(len(px) - 1):
                bbox_list.append(torch.tensor([px[i], py[j], px[i+1], py[j+1]]))
        
        bbox_tensor = torch.stack(bbox_list, dim=0)
        print(bbox_tensor.shape)
        x = torch.divide(bbox_tensor, scaling_ratio)
        pos_embed = torch.matmul(x.unsqueeze(-1), self.inv_freq.unsqueeze(0))
        pos_embed_sin = pos_embed.sin()
        pos_embed_cos = pos_embed.cos()
        spatial_pos_embedding = torch.cat([pos_embed_sin, pos_embed_cos], dim=-1)
        spatial_pos_embedding = torch.flatten(spatial_pos_embedding, start_dim=-2)
        self.register_buffer("bbox", bbox_tensor)
        self.register_buffer("spatial_pos_embedding", spatial_pos_embedding)
        self.scaling_ratio = scaling_ratio
        factor = torch.nn.Parameter(torch.ones(1) * factor_ratio)
        self.register_parameter("factor", factor)

    def forward(self):
        return self.spatial_pos_embedding * self.factor
    
    def get_bbox_list(self, batch_size):
        return [self.bbox] * batch_size


class MLPLayer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=2,
                 num_dim=1024,
                 activation='relu'):
        super().__init__()
        self.output_dim = output_dim
        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        if num_layers > 0:
            self._layers = [torch.nn.Linear(input_dim, num_dim),
                            activate_fn()]
            for i in range(1, num_layers):
                self._layers += [torch.nn.Linear(num_dim, num_dim),
                                activate_fn()]

            self._layers += [torch.nn.Linear(num_dim, output_dim)]
        else:
            self._layers += [torch.nn.Linear(input_dim, output_dim)]
        self.layers = torch.nn.Sequential(*self._layers)

    def forward(self, x):
        h = self.layers(x)
        return h
    
    
class PolicyMLPLayer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=2,
                 num_dim=1024,
                 activation='relu',
                 action_scale=1.,
                 action_squash=True):
        super().__init__()
        self.output_dim = output_dim
        self.action_scale = 1.0
        self.action_squash = action_squash
        if activation == 'relu':
            activate_fn = torch.nn.ReLU
        elif activation == 'leaky-relu':
            activate_fn = torch.nn.LeakyReLU

        if num_layers > 0:
            self._layers = [torch.nn.Linear(input_dim, num_dim),
                            activate_fn()]
            for i in range(1, num_layers):
                self._layers += [torch.nn.Linear(num_dim, num_dim),
                                activate_fn()]

            self._layers += [torch.nn.Linear(num_dim, output_dim)]
        else:
            self._layers += [torch.nn.Linear(input_dim, output_dim)]
        self.layers = torch.nn.Sequential(*self._layers)

    def forward(self, x):
        h = self.layers(x)
        if self.action_squash:
            h = torch.tanh(h) * self.action_scale
        return h

class GMMPolicyMLPLayer(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 num_modes=5, 
                 min_std=0.0001,
                 num_layers=2,
                 num_dim=1024,              
                 mlp_activation="relu",
                 std_activation="softplus", 
                 low_noise_eval=True, 
                 use_tanh=False):
        super().__init__()
        self.num_modes = num_modes
        self.output_dim = output_dim
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.std_activation = std_activation
        self.use_tanh = use_tanh
        
        if mlp_activation == 'relu':
            mlp_activate_fn = torch.nn.ReLU
        elif mlp_activation == 'leaky-relu':
            mlp_activate_fn = torch.nn.LeakyReLU
        
        out_dim = self.num_modes * self.output_dim
        if num_layers > 0:
            self._layers = [torch.nn.Linear(input_dim, num_dim),
                            mlp_activate_fn()]
            for i in range(1, num_layers):
                self._layers += [torch.nn.Linear(num_dim, num_dim),
                                mlp_activate_fn()]

        else:
            self._layers += [torch.nn.Linear(input_dim, num_dim)]
        self.mlp_layers = torch.nn.Sequential(*self._layers)
        
        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }

        self.mean_layer = nn.Linear(num_dim, out_dim)
        self.scale_layer = nn.Linear(num_dim, out_dim)
        self.logits_layer = nn.Linear(num_dim, self.num_modes)
        
    def forward(self, x):
        x = self.mlp_layers(x)
        means = self.mean_layer(x).view(-1, self.num_modes, self.output_dim)
        scales = self.scale_layer(x).view(-1, self.num_modes, self.output_dim)
        logits = self.logits_layer(x)
        
        means = torch.tanh(means)
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        self.means = means
        self.scales = scales
        self.logits = logits

        return dist


class TemporalGMMPolicyMLPLayer(nn.Module):
    """This is a mlp layer that handles temporal sequence. (because of of restricted usage from robomimic)
    """
    def __init__(self, 
                 input_dim,
                 output_dim,
                 num_modes=5, 
                 min_std=0.0001,
                 num_layers=2,
                 num_dim=1024,              
                 mlp_activation="relu",
                 std_activation="softplus", 
                 low_noise_eval=True, 
                 use_tanh=False):
        super().__init__()
        self.num_modes = num_modes
        self.output_dim = output_dim
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.std_activation = std_activation
        self.use_tanh = use_tanh
        
        if mlp_activation == 'relu':
            mlp_activate_fn = torch.nn.ReLU
        elif mlp_activation == 'leaky-relu':
            mlp_activate_fn = torch.nn.LeakyReLU
        
        out_dim = self.num_modes * self.output_dim
        if num_layers > 0:
            self._layers = [torch.nn.Linear(input_dim, num_dim),
                            mlp_activate_fn()]
            for i in range(1, num_layers):
                self._layers += [torch.nn.Linear(num_dim, num_dim),
                                mlp_activate_fn()]

        else:
            self._layers += [torch.nn.Linear(input_dim, num_dim)]
        self.mlp_layers = torch.nn.Sequential(*self._layers)
        
        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }

        self.mean_layer = nn.Linear(num_dim, out_dim)
        self.scale_layer = nn.Linear(num_dim, out_dim)
        self.logits_layer = nn.Linear(num_dim, self.num_modes)

    def forward_fn(self, x):
        out = self.mlp_layers(x)
        means = self.mean_layer(out).view(-1, self.num_modes, self.output_dim)
        scales = self.scale_layer(out).view(-1, self.num_modes, self.output_dim)
        logits = self.logits_layer(out)

        means = torch.tanh(means)
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std
        
        return means, scales, logits
        
    def forward(self, x):
        # x = self.mlp_layers(x)
        # means = self.mean_layer(x).view(-1, self.num_modes, self.output_dim)
        # scales = self.scale_layer(x).view(-1, self.num_modes, self.output_dim)
        # logits = self.logits_layer(x)

        means, scales, logits = TensorUtils.time_distributed(x, self.forward_fn)

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        self.means = means
        self.scales = scales
        self.logits = logits
    
        return dist    

class RNNBackbone(nn.Module):
    def __init__(self,
                 input_dim=64,
                 rnn_hidden_dim=1000,
                 rnn_num_layers=2,
                 rnn_type="LSTM",
                 per_step_net=None,
                 rnn_kwargs={"bidirectional": False},
                 *args,
                 **kwargs):
        super().__init__()
        self.per_step_net = eval(per_step_net.network)(**per_step_net.network_kwargs)
        self.rnn_model = RNN_Base(
            input_dim=64,
            rnn_hidden_dim=1000,
            rnn_num_layers=2,
            rnn_type="LSTM",
            per_step_net=self.per_step_net,
            rnn_kwargs=rnn_kwargs
        )

    def forward(self, x, *args, **kwargs):
        return self.rnn_model(x, *args, **kwargs)

    def get_rnn_init_state(self, *args, **kwargs):
        return self.rnn_model.get_rnn_init_state(*args, **kwargs)

    def forward_step(self, *args, **kwargs):
        return self.rnn_model.forward_step(*args, **kwargs)

class GMMPolicyOutputHead(robomimic.models.base_nets.Module):
    """GMM policy output head without any nonlinear MLP layers."""
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_modes=5, 
                 min_std=0.0001,
                 std_activation="softplus", 
                 low_noise_eval=False, 
                 use_tanh=False
    ):
        super().__init__()

        self.num_modes = num_modes
        self.output_dim = output_dim
        self.min_std = min_std

        self.low_noise_eval = low_noise_eval
        self.std_activation = std_activation
        self.use_tanh = use_tanh
        
        out_dim = self.num_modes * output_dim

        self.mean_layer = nn.Linear(input_dim, out_dim)
        self.scale_layer = nn.Linear(input_dim, out_dim)
        self.logits_layer = nn.Linear(input_dim, self.num_modes)
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }

        
    def forward(self, x):
        means = self.mean_layer(x).view(-1, self.num_modes, self.output_dim)
        scales = self.scale_layer(x).view(-1, self.num_modes, self.output_dim)
        logits = self.logits_layer(x)
        
        means = torch.tanh(means)
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        self.means = means
        self.scales = scales
        self.logits = logits

        return {"means": self.means,
                "scales": self.scales,
                "logits": self.logits}

