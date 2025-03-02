import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, resample=None):
        super().__init__()

        self.register_buffer("gain", torch.sqrt(torch.tensor(2)))

        self.act = nn.LeakyReLU(0.2)

        self.residual_skip = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

        if resample == 'down':
            self.resample = nn.AvgPool2d(3, 2)
        else:
            self.resample = nn.Identity()

        self.norm_0 = nn.InstanceNorm2d(c_in, eps=0.001, affine=True, track_running_stats=True)
        self.conv_0 = nn.Conv2d(c_in, c_out, (3, 3), padding=1)
        self.norm_1 = nn.InstanceNorm2d(c_out, eps=0.001, affine=True, track_running_stats=True)
        self.conv_1 = nn.Conv2d(c_out, c_out, (3, 3), padding=1)

    def forward(self, x):
        x_ = self.residual_skip(x)
        x_ = self.resample(x_)

        x = self.norm_0(x)
        x = self.act(x)
        x = self.conv_0(x)

        x = self.resample(x)

        x = self.norm_1(x)
        x = self.act(x)
        x = self.conv_1(x)

        x = x + x_

        return x * 0.707017


class Discriminator(nn.Module):
    def __init__(self,
                 encoder_skip_idx=2,

                 channel_in=3,
                 channel_base=64,
                 channel_multiplier=2.0,
                 channel_max=512,
                 num_resample_steps=6,
                 channel_scale=1.0,  # Scales the channel sizes directly
                 ):
        super().__init__()

        self.encoder_skip_idx = encoder_skip_idx

        self.channel_in = channel_in
        self.channel_base = channel_base
        self.channel_multiplier = channel_multiplier
        self.channel_max = channel_max
        self.num_resample_steps = num_resample_steps
        self.channel_scale = channel_scale

        # Calculate channels for each block
        channels = [min(int((channel_base * (channel_multiplier ** i)) * channel_scale), channel_max)
                    for i in range(num_resample_steps)]

        # Prepare encoder layers with module list and an initial block
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Conv2d(channel_in, channels[0], 3, 1, 1))

        c_in = channels[0]
        for c_out in channels[1:num_resample_steps + 1]:
            self.encoder.append(ResidualBlock(c_in, c_out, resample='down'))
            c_in = c_out

        self.encoder.append(nn.Conv2d(c_out, c_out, 4, 1))
        self.encoder.append(nn.LeakyReLU(0.2))
        self.encoder.append(nn.Conv2d(c_out, 1, 4, 1))

    def forward(self, im):
        x = im
        for block in self.encoder:
            x = block(x)

        return x  # Skip final bottle


class EyeDiscriminator(nn.Module):
    def __init__(self,
                 encoder_skip_idx=2,

                 channel_in=3,
                 channel_base=64,
                 channel_multiplier=2.0,
                 channel_max=512,
                 num_resample_steps=3,
                 channel_scale=1.0,  # Scales the channel sizes directly
                 ):
        super().__init__()

        self.encoder_skip_idx = encoder_skip_idx

        self.channel_in = channel_in
        self.channel_base = channel_base
        self.channel_multiplier = channel_multiplier
        self.channel_max = channel_max
        self.num_resample_steps = num_resample_steps
        self.channel_scale = channel_scale

        # Calculate channels for each block
        channels = [min(int((channel_base * (channel_multiplier ** i)) * channel_scale), channel_max)
                    for i in range(num_resample_steps)]

        # Prepare encoder layers with module list and an initial block
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Conv2d(channel_in, channels[0], 3, 1, 1))

        c_in = channels[0]
        for c_out in channels[1:num_resample_steps + 1]:
            self.encoder.append(ResidualBlock(c_in, c_out, resample='down'))
            c_in = c_out

        self.encoder_discriminator_head = nn.Sequential(nn.Conv2d(c_out, c_out, 4, 1),
                                                        nn.LeakyReLU(0.2),
                                                        nn.Conv2d(c_out, 1, 4, 1))
        self.encoder_similarity_head = nn.Sequential(nn.Conv2d(c_out * 2, c_out, 4, 1),
                                                     nn.LeakyReLU(0.2),
                                                     nn.Conv2d(c_out, 1, 4, 1))

    def forward(self, left, right):
        for block in self.encoder:
            left = block(left)
            right = block(right)

        left_discrimination = self.encoder_discriminator_head(left)
        right_discrimination = self.encoder_discriminator_head(right)

        similarity = self.encoder_similarity_head(torch.concat((left, right), dim=1))
        dissimilarity = self.encoder_similarity_head(torch.concat((left, right.roll(1, 0)), dim=1))

        return left_discrimination, right_discrimination, similarity, dissimilarity



