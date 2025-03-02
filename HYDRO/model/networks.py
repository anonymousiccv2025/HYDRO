import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import einsum
from einops import rearrange, repeat
from typing import Tuple, Dict


def concatenate_feature_maps(fm_0, fm_1):
    return torch.cat([fm_0, fm_1], dim=1)


def add_feature_maps(fm_0, fm_1):
    return (fm_0 + fm_1) * 0.5


def exists(val):
    return val is not None


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0


class Resample(nn.Module):
    def __init__(self,
                 h_in,
                 resample_type='conv_stride',
                 gain=0.5):
        super(Resample, self).__init__()

        if resample_type == 'conv_stride':
            self.resample = nn.Conv2d(h_in, h_in, kernel_size=2, stride=2)
        elif resample_type == 'pixel_shuffle':
            self.resample = nn.Sequential(nn.Conv2d(h_in, h_in * 4, kernel_size=3, padding=1),
                                          nn.PixelShuffle(2))
        elif resample_type == 'conv_interpolate_up':
            self.resample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                          nn.Conv2d(h_in, h_in, kernel_size=3, padding=1))
        self.gain = gain

    def forward(self, h):
        return self.resample(h) * self.gain


class AdaptiveSkip(nn.Module):
    def __init__(self,
                 h_in,
                 c_in, ):
        super(AdaptiveSkip, self).__init__()

        self.adaptive = nn.Linear(c_in, h_in)

    def forward(self, h, hs, c):
        gate = self.adaptive(c)[..., None, None]
        return h + hs * gate


class ResBlock(nn.Module):
    def __init__(self,
                 h_in,
                 h_out,
                 c_in,
                 norm=nn.InstanceNorm2d,
                 adaptive=True,
                 adaptive_type='scale_shift_gate',
                 adaptive_layer='linear',
                 attention=False,
                 kv_compress=2,
                 skip_type: str = 'addition', ):
        super(ResBlock, self).__init__()
        self.h_out = h_out
        self.norm = norm

        self.norm_0 = norm(h_in)
        self.conv_0 = nn.Conv2d(h_in, h_out, kernel_size=3, padding=1)

        self.norm_1 = norm(h_out)
        self.conv_1 = nn.Conv2d(h_out, h_out, kernel_size=3, padding=1)

        self.res = nn.Conv2d(h_in, h_out, kernel_size=1, stride=1) if h_in != h_out else nn.Identity()

        self.skip_type = skip_type
        if skip_type == 'concatenate':
            self.skip_op = concatenate_feature_maps
        elif skip_type == 'adaptive':
            self.skip_op = add_feature_maps
        else:
            self.skip_op = add_feature_maps

        self.adaptive = None
        self.adaptive_size = 1
        if adaptive:
            self.adaptive_type = adaptive_type
            if adaptive_type == 'scale_shift':
                self.adaptive_size = 2
            elif adaptive_type == 'scale_shift_gate':
                self.adaptive_size = 3
            if attention:
                self.adaptive_size *= 3
            if skip_type == 'adaptive':
                self.adaptive_size += 1

            if adaptive_layer == 'linear':
                self.adaptive = nn.Linear(c_in, h_out * self.adaptive_size)
            elif adaptive_layer == 'mlp':
                self.adaptive = nn.Sequential(nn.Linear(c_in, h_out * self.adaptive_size),
                                              nn.SiLU(),
                                              nn.Linear(h_out * self.adaptive_size, h_out * self.adaptive_size))
        self.attention = None
        if attention:
            self.attention_norm = nn.InstanceNorm2d(h_out)
            self.attention = SelfAttention(h_out, dim_head=h_out // 8, kv_compress=kv_compress)

    def forward(self, h, hs=None, c=None, noise_gain=None):
        scale, shift, gate = 0.0, 0.0, 1.0
        scale_attention, shift_attention, gate_attention = 0.0, 0.0, 1.0
        gate_skip = 1.0

        if c is not None and self.adaptive is not None:
            c = self.adaptive(c)[..., None, None]
            adaptives = c.chunk(self.adaptive_size, dim=1)
            if self.adaptive_type == 'scale_shift':
                scale, shift = adaptives[0:2]
            elif self.adaptive_type == 'scale_shift_gate':
                scale, shift, gate = adaptives[0:3]
            elif self.adaptive_type == 'scale_shift' and self.attention is not None:
                scale, shift, scale_attention, shift_attention = adaptives[0:4]
            elif self.adaptive_type == 'scale_shift_gate' and self.attention is not None:
                scale, shift, gate, scale_attention, shift_attention, gate_attention = adaptives[0:6]
            else:
                shift = c
            if self.skip_type == 'adaptive':
                gate_skip = adaptives[-1]

        if hs is not None:
            h = self.skip_op(h, hs * gate_skip)

        h_ = self.conv_0(F.silu(self.norm_0(h)))

        h_ = self.norm_1(h_)

        h_ = h_ * (1 + scale) + shift
        h_ = self.conv_1(F.silu(h_)) * gate

        h = h_ + self.res(h)

        if noise_gain is not None:
            h = h + torch.randn_like(h, device=h.device) * noise_gain

        if self.attention is not None:
            h_ = self.attention_norm(h)
            h_ = h_ * (1 + scale_attention) + shift_attention
            h = h + self.attention(h_) * gate_attention

        return h


# From https://github.com/lucidrains/gigagan-pytorch/blob/main/gigagan_pytorch/gigagan_pytorch.py#L512
class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            dot_product=True,
            kv_compress=2,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.dot_product = dot_product

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim_inner, kv_compress, stride=kv_compress, bias=False) if dot_product else None
        self.to_v = nn.Conv2d(dim, dim_inner, kv_compress, stride=kv_compress, bias=False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias=False)

    def forward(self, fmap):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch = fmap.shape[0]

        x, y = fmap.shape[-2:]

        h = self.heads

        q, v = self.to_q(fmap), self.to_v(fmap)

        k = self.to_k(fmap) if exists(self.to_k) else q

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=self.heads), (q, k, v))

        # add a null key / value, so network can choose to pay attention to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b=batch), self.null_kv)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # l2 distance or dot product

        if self.dot_product:
            sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            # using pytorch cdist leads to nans in lightweight gan training framework, at least
            q_squared = (q * q).sum(dim=-1)
            k_squared = (k * k).sum(dim=-1)
            l2dist_squared = rearrange(q_squared, 'b i -> b i 1') + rearrange(k_squared, 'b j -> b 1 j') - 2 * einsum(
                'b i d, b j d -> b i j', q, k)  # hope i'm mathing right
            sim = -l2dist_squared

        # scale

        sim = sim * self.scale

        # attention

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x=x, y=y, h=h)

        return self.to_out(out)


class Generator(nn.Module):
    def __init__(self,
                 i_in: int = 3,
                 c_in: int = 512,

                 resolution: int = 256,
                 channel_in: int = 3,
                 channel_base: int = 64,
                 channel_multiplier: float = 2.0,
                 channel_max: int = 1024,
                 num_resample_steps: int = 5,
                 channel_scale: float = 1.0,  # Scales the channel sizes directly
                 attention_resolutions: Tuple[int] = (8, 16, 32),
                 kv_compress: Dict[int, int] = {8: 1, 16: 2, 32: 2, 64: None, 128: None, 256: None},
                 skip_resolutions: Tuple[int] = (16, 32, 64, 128, 256),
                 resample_type: str = 'conv_interpolate_up',
                 adaptive_type: str = 'scale_shift_gate',
                 skip_type: str = 'addition',
                 adaptive_layer: str = 'linear',

                 mapping: bool = False,
                 resample_gain: float = 0.5,
                 mask_layer: bool = True,

                 multi_scale_output: bool = False,
                 ):
        super(Generator, self).__init__()

        self.resolution = resolution
        self.channel_in = channel_in
        self.channel_base = channel_base
        self.channel_multiplier = channel_multiplier
        self.channel_max = channel_max
        self.num_resample_steps = num_resample_steps
        self.channel_scale = channel_scale
        self.skip_resolutions = skip_resolutions
        self.adaptive_layer = adaptive_layer
        self.multi_scale_output = multi_scale_output

        self.skip_scale = 2 if skip_type == 'concatenate' else 1

        self.mapping = None
        if mapping:
            self.mapping = nn.Sequential(nn.Linear(c_in, c_in),
                                         nn.SiLU(),
                                         nn.Linear(c_in, c_in))

        # Calculate channels for each block
        ch = [min(int((channel_base * (channel_multiplier ** i)) * channel_scale), channel_max)
              for i in range(num_resample_steps + 1)]
        ch_up = [min(c * 2, channel_max) for c in ch]
        self.ch = ch

        # Inputs
        self.conv_in = nn.Conv2d(i_in, ch[0], kernel_size=3, padding=1)

        current_resolution = resolution
        self.encoder = nn.ModuleList()
        for idx in range(len(ch) - 1):
            encoder_block = nn.Module()
            encoder_block.block = ResBlock(ch[idx], ch[idx + 1], c_in,
                                           attention=current_resolution in attention_resolutions,
                                           kv_compress=kv_compress[current_resolution],
                                           adaptive_type=adaptive_type,
                                           adaptive_layer=adaptive_layer)
            encoder_block.resample = Resample(ch[idx + 1], 'conv_stride', resample_gain)
            self.encoder.append(encoder_block)

            current_resolution /= 2

        self.encoder_out = ResBlock(ch[-1], ch_up[-1], c_in,
                                    attention=current_resolution in attention_resolutions,
                                    kv_compress=kv_compress[current_resolution],
                                    adaptive_type=adaptive_type,
                                    adaptive_layer=adaptive_layer)

        self.decoder = nn.ModuleList()

        for idx in range(len(ch) - 1):
            decoder_block = nn.Module()
            skip_scale = self.skip_scale if current_resolution in skip_resolutions else 1
            decoder_block.block = ResBlock(ch_up[-1 - idx] * skip_scale, ch_up[-2 - idx], c_in,
                                           attention=current_resolution in attention_resolutions,
                                           kv_compress=kv_compress[current_resolution],
                                           adaptive_type=adaptive_type,
                                           adaptive_layer=adaptive_layer,
                                           skip_type=skip_type)
            decoder_block.resample = Resample(ch_up[-2 - idx], resample_type, resample_gain)
            self.decoder.append(decoder_block)

            if self.multi_scale_output:
                decoder_block.to_rgb = nn.Sequential(nn.InstanceNorm2d(ch_up[-2 - idx], ch_up[-2 - idx]),
                                                     nn.SiLU(),
                                                     nn.Conv2d(ch_up[-2 - idx], 3))

            current_resolution *= 2

        skip_scale = self.skip_scale if current_resolution in skip_resolutions else 1
        self.decoder_out = ResBlock(ch_up[-2 - idx] * skip_scale, ch[0], c_in,
                                    attention=current_resolution in attention_resolutions,
                                    kv_compress=kv_compress[current_resolution],
                                    adaptive_type=adaptive_type,
                                    adaptive_layer=adaptive_layer)
        self.norm_out = nn.InstanceNorm2d(ch[0])
        self.conv_out = nn.Conv2d(ch[0], i_in, kernel_size=3, padding=1)

        self.mask_out = None
        if mask_layer:
            self.mask_out = nn.Conv2d(ch[0], 1, kernel_size=3, padding=1)

        self.initialize_weights()

    def forward(self, h, c=None):

        if self.mapping is not None and c is not None:
            c = self.mapping(c)

        # Encode
        h = self.conv_in(h)
        hs = {}
        current_resolution = self.resolution
        for idx, encoder_block in enumerate(self.encoder):
            h = encoder_block.block(h, c=c)
            hs[current_resolution] = h
            h = encoder_block.resample(h)
            current_resolution /= 2

        h = self.encoder_out(h, c=c)
        hs[current_resolution] = h

        # Decode
        rgb = 0
        for idx, decoder_block in enumerate(self.decoder):

            if current_resolution in self.skip_resolutions:
                skip = hs[current_resolution]
            else:
                skip = None

            h = decoder_block.block(h, c=c, hs=skip)

            if self.multi_scale_output:
                rgb += decoder_block.to_rgb(h)
                rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear')

            h = decoder_block.resample(h)
            current_resolution *= 2

        if current_resolution in self.skip_resolutions:
            skip = hs[current_resolution]
        else:
            skip = None

        h = self.decoder_out(h, c=c, hs=skip)

        h = self.norm_out(h)
        h = F.silu(h)
        i = self.conv_out(h) + rgb
        if self.mask_out is not None:
            m = torch.sigmoid(self.mask_out(h))
            return i, m
        else:
            return i

    def forward_mix(self, h, c=None):

        if self.mapping is not None and c is not None:
            c = [self.mapping(_c) for _c in c]

        c_idx = 0

        # Encode
        h = self.conv_in(h)
        hs = {}
        current_resolution = self.resolution
        for idx, encoder_block in enumerate(self.encoder):
            h = encoder_block.block(h, c=c[c_idx])
            c_idx += 1
            hs[current_resolution] = h
            h = encoder_block.resample(h)
            current_resolution /= 2

        h = self.encoder_out(h, c=c[c_idx])
        c_idx += 1
        hs[current_resolution] = h

        # Decode
        rgb = 0
        for idx, decoder_block in enumerate(self.decoder):

            if current_resolution in self.skip_resolutions:
                skip = hs[current_resolution]
            else:
                skip = None

            h = decoder_block.block(h, c=c[c_idx], hs=skip)
            c_idx += 1

            if self.multi_scale_output:
                rgb += decoder_block.to_rgb(h)
                rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear')

            h = decoder_block.resample(h)
            current_resolution *= 2

        if current_resolution in self.skip_resolutions:
            skip = hs[current_resolution]
        else:
            skip = None

        h = self.decoder_out(h, c=c[c_idx], hs=skip)
        c_idx += 1

        h = self.norm_out(h)
        h = F.silu(h)
        i = self.conv_out(h) + rgb
        if self.mask_out is not None:
            m = torch.sigmoid(self.mask_out(h))
            return i, m
        else:
            return i


    def initialize_weights(self):

        def base_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight,
                                              #gain=nn.init.calculate_gain('relu')
                                              )
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight,
                                              #gain=nn.init.calculate_gain('relu')
                                              )
                torch.nn.init.zeros_(m.bias)

        self.apply(base_init)

        # Zero out the scale, shift, and gate layers
        for encoder_block in self.encoder:
            if encoder_block.block.adaptive is not None:
                if self.adaptive_layer == 'linear':
                    torch.nn.init.constant_(encoder_block.block.adaptive.weight, 0)
                    torch.nn.init.constant_(encoder_block.block.adaptive.bias, 0)
                else:
                    torch.nn.init.constant_(encoder_block.block.adaptive[-1].weight, 0)
                    torch.nn.init.constant_(encoder_block.block.adaptive[-1].bias, 0)
        if self.adaptive_layer == 'linear':
            torch.nn.init.constant_(self.encoder_out.adaptive.weight, 0)
            torch.nn.init.constant_(self.encoder_out.adaptive.bias, 0)
        else:
            torch.nn.init.constant_(self.encoder_out.adaptive[-1].weight, 0)
            torch.nn.init.constant_(self.encoder_out.adaptive[-1].bias, 0)

        for decoder_block in self.decoder:
            if decoder_block.block.adaptive is not None:
                if self.adaptive_layer == 'linear':
                    torch.nn.init.constant_(decoder_block.block.adaptive.weight, 0)
                    torch.nn.init.constant_(decoder_block.block.adaptive.bias, 0)
                else:
                    torch.nn.init.constant_(decoder_block.block.adaptive[-1].weight, 0)
                    torch.nn.init.constant_(decoder_block.block.adaptive[-1].bias, 0)
        if self.adaptive_layer == 'linear':
            torch.nn.init.constant_(self.decoder_out.adaptive.weight, 0)
            torch.nn.init.constant_(self.decoder_out.adaptive.bias, 0)
        else:
            torch.nn.init.constant_(self.decoder_out.adaptive[-1].weight, 0)
            torch.nn.init.constant_(self.decoder_out.adaptive[-1].bias, 0)


class FLOPWrapper(nn.Module):
    def __init__(self,
                 generator,
                 identity_encoder
                 ):
        super(FLOPWrapper, self).__init__()

        self.generator = generator
        self.identity_encoder =  identity_encoder

    def forward(self, x):
        return self.generator(x, self.identity_encoder(F.interpolate(x, size=112)))


class FLOPWrapperDiffusion(nn.Module):
    def __init__(self,
                 generator,
                 identity_encoder
                 ):
        super(FLOPWrapperDiffusion, self).__init__()

        self.generator = generator
        self.identity_encoder =  identity_encoder

    def forward(self, x):
        t = torch.randint(0, 10, (1,))
        return self.generator(x, self.identity_encoder(F.interpolate(x, size=112)), t)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiffusionGenerator(nn.Module):
    def __init__(self,
                 i_in: int = 3,
                 c_in: int = 512,
                 num_steps: int = 1,

                 resolution: int = 256,
                 channel_in: int = 3,
                 channel_base: int = 64,
                 channel_multiplier: float = 2.0,
                 channel_max: int = 1024,
                 num_resample_steps: int = 5,
                 channel_scale: float = 1.0,  # Scales the channel sizes directly
                 attention_resolutions: Tuple[int] = (8, 16, 32),
                 kv_compress: Dict[int, int] = {8: 1, 16: 2, 32: 2, 64: None, 128: None, 256: None},
                 skip_resolutions: Tuple[int] = (16, 32, 64, 128, 256),
                 resample_type: str = 'conv_interpolate_up',
                 adaptive_type: str = 'scale_shift_gate',
                 skip_type: str = 'addition',

                 mapping: bool = False,
                 condition_image_in: int = 0,
                 ):
        super(DiffusionGenerator, self).__init__()

        self.resolution = resolution
        self.channel_in = channel_in
        self.channel_base = channel_base
        self.channel_multiplier = channel_multiplier
        self.channel_max = channel_max
        self.num_resample_steps = num_resample_steps
        self.channel_scale = channel_scale
        self.skip_resolutions = skip_resolutions

        self.skip_scale = 2 if skip_type == 'concatenate' else 1

        self.mapping = None
        if mapping:
            self.mapping = nn.Sequential(nn.Linear(c_in, c_in),
                                         nn.SiLU(),
                                         nn.Linear(c_in, c_in))

        self.t_embedder = TimestepEmbedder(c_in)

        # Calculate channels for each block
        ch = [min(int((channel_base * (channel_multiplier ** i)) * channel_scale), channel_max)
              for i in range(num_resample_steps + 1)]
        ch_up = [min(c * 2, channel_max) for c in ch]
        self.ch = ch

        # Inputs
        self.conv_in = nn.Conv2d(i_in + condition_image_in, ch[0], kernel_size=3, padding=1)

        current_resolution = resolution
        self.encoder = nn.ModuleList()
        for idx in range(len(ch) - 1):
            encoder_block = nn.Module()
            encoder_block.block = ResBlock(ch[idx], ch[idx + 1], c_in,
                                           attention=current_resolution in attention_resolutions,
                                           kv_compress=kv_compress[current_resolution],
                                           adaptive_type=adaptive_type)
            encoder_block.resample = Resample(ch[idx + 1], 'conv_stride')
            self.encoder.append(encoder_block)

            current_resolution /= 2

        self.encoder_out = ResBlock(ch[-1], ch_up[-1], c_in,
                                    attention=current_resolution in attention_resolutions,
                                    kv_compress=kv_compress[current_resolution],
                                    adaptive_type=adaptive_type)

        self.decoder = nn.ModuleList()

        for idx in range(len(ch) - 1):
            decoder_block = nn.Module()
            skip_scale = self.skip_scale if current_resolution in skip_resolutions else 1
            decoder_block.block = ResBlock(ch_up[-1 - idx] * skip_scale, ch_up[-2 - idx], c_in,
                                           attention=current_resolution in attention_resolutions,
                                           kv_compress=kv_compress[current_resolution],
                                           adaptive_type=adaptive_type,
                                           skip_type=skip_type)
            decoder_block.resample = Resample(ch_up[-2 - idx], resample_type)
            self.decoder.append(decoder_block)

            current_resolution *= 2

        skip_scale = self.skip_scale if current_resolution in skip_resolutions else 1
        self.decoder_out = ResBlock(ch_up[-2 - idx] * skip_scale, ch[0], c_in,
                                    attention=current_resolution in attention_resolutions,
                                    kv_compress=kv_compress[current_resolution],
                                    adaptive_type=adaptive_type)
        self.norm_out = nn.InstanceNorm2d(ch[0])
        self.conv_out = nn.Conv2d(ch[0], i_in * num_steps, kernel_size=3, padding=1)

        self.initialize_weights()

    def forward(self, h, c, t):

        if self.mapping is not None:
            c = self.mapping(c)

        c += self.t_embedder(t)

        # Encode
        h = self.conv_in(h)
        hs = {}
        current_resolution = self.resolution
        for idx, encoder_block in enumerate(self.encoder):
            h = encoder_block.block(h, c=c)
            hs[current_resolution] = h
            h = encoder_block.resample(h)
            current_resolution /= 2

        h = self.encoder_out(h, c=c)
        hs[current_resolution] = h

        # Decode
        for idx, decoder_block in enumerate(self.decoder):

            if current_resolution in self.skip_resolutions:
                skip = hs[current_resolution]
            else:
                skip = None

            h = decoder_block.block(h, c=c, hs=skip)
            h = decoder_block.resample(h)
            current_resolution *= 2

        if current_resolution in self.skip_resolutions:
            skip = hs[current_resolution]
        else:
            skip = None

        h = self.decoder_out(h, c=c, hs=skip)

        h = self.norm_out(h)
        h = F.silu(h)
        i = self.conv_out(h)
        return i

    def initialize_weights(self):

        # Zero out the scale, shift, and gate layers
        for encoder_block in self.encoder:
            if encoder_block.block.adaptive is not None:
                torch.nn.init.constant_(encoder_block.block.adaptive.weight, 0)
                torch.nn.init.constant_(encoder_block.block.adaptive.bias, 0)

        torch.nn.init.constant_(self.encoder_out.adaptive.weight, 0)
        torch.nn.init.constant_(self.encoder_out.adaptive.bias, 0)

        for decoder_block in self.decoder:
            if decoder_block.block.adaptive is not None:
                torch.nn.init.constant_(decoder_block.block.adaptive.weight, 0)
                torch.nn.init.constant_(decoder_block.block.adaptive.bias, 0)

        torch.nn.init.constant_(self.decoder_out.adaptive.weight, 0)
        torch.nn.init.constant_(self.decoder_out.adaptive.bias, 0)

        torch.nn.init.constant_(self.conv_out.weight, 0)
        torch.nn.init.constant_(self.conv_out.bias, 0)

