import torch
import torch.nn as nn
import torch.nn.functional as F

# From https://github.com/protossw512/AdaptiveWingLoss
class AddCoordsTh(nn.Module):
    def __init__(self, x_dim=64, y_dim=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, heatmap=None):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).to(input_tensor.device)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).to(input_tensor.device)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).to(input_tensor.device)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).to(input_tensor.device)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range.float(), yy_ones.float())
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel / (self.x_dim - 1)
        yy_channel = yy_channel / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        if self.with_boundary and type(heatmap) != type(None):
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :],
                                           0.0, 1.0)

            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary_channel = torch.where(boundary_channel > 0.05,
                                              xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel > 0.05,
                                              yy_channel, zero_tensor)
        if self.with_boundary and type(heatmap) != type(None):
            xx_boundary_channel = xx_boundary_channel.to(input_tensor.device)
            yy_boundary_channel = yy_boundary_channel.to(input_tensor.device)
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            rr = rr / torch.max(rr)
            ret = torch.cat([ret, rr], dim=1)

        if self.with_boundary and type(heatmap) != type(None):
            ret = torch.cat([ret, xx_boundary_channel,
                             yy_boundary_channel], dim=1)
        return ret


class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""

    def __init__(self, x_dim, y_dim, with_r, with_boundary,
                 in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(x_dim=x_dim, y_dim=y_dim, with_r=with_r,
                                     with_boundary=with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel


def conv3x3(in_planes, out_planes, strd=1, padding=1,
            bias=False, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias,
                     dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4),
                             padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4),
                             padding=1, dilation=1)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(x_dim=64, y_dim=64,
                                     with_r=True, with_boundary=True,
                                     in_channels=256, first_one=first_one,
                                     out_channels=256,
                                     kernel_size=1,
                                     stride=1, padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.upsample(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


class FAN(nn.Module):

    def __init__(self, num_modules=1, end_relu=False, gray_scale=False,
                 num_landmarks=68):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.gray_scale = gray_scale
        self.end_relu = end_relu
        self.num_landmarks = num_landmarks

        # Base part
        if self.gray_scale:
            self.conv1 = CoordConvTh(x_dim=256, y_dim=256,
                                     with_r=True, with_boundary=False,
                                     in_channels=3, out_channels=64,
                                     kernel_size=7,
                                     stride=2, padding=3)
        else:
            self.conv1 = CoordConvTh(x_dim=256, y_dim=256,
                                     with_r=True, with_boundary=False,
                                     in_channels=3, out_channels=64,
                                     kernel_size=7,
                                     stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            if hg_module == 0:
                first_one = True
            else:
                first_one = False
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256,
                                                            first_one))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            num_landmarks + 1, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(num_landmarks + 1,
                                                                 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        # x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        boundary_channels = []
        tmp_out = None
        for i in range(self.num_modules):
            hg, boundary_channel = self._modules['m' + str(i)](previous,
                                                               tmp_out)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            if self.end_relu:
                tmp_out = F.relu(tmp_out)  # HACK: Added relu
            outputs.append(tmp_out)
            boundary_channels.append(boundary_channel)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs, boundary_channels

    def get_preds_fromhm(self, x):
        x = x * 0.5 + 0.5

        outputs, boundary_channels = self(x)
        hm = outputs[-1][:, :-1, :, :]

        max, idx = torch.max(
            hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
        idx += 1
        preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
        preds[..., 0] = (preds[..., 0] - 1) % hm.size(3) + 1
        preds[..., 1] = preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

        for i in range(preds.size(0)):
            for j in range(preds.size(1)):
                hm_ = hm[i, j, :]
                pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
                if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                    diff = torch.tensor(
                        [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                         hm_[pY + 1, pX] - hm_[pY - 1, pX]], device=hm.device, dtype=hm.dtype)
                    preds[i, j].add_(diff.sign_().mul_(.25))

        preds.add_(-0.5)

        return preds

    def detect_landmarks(self, x):
        x = x * 0.5 + 0.5

        outputs, boundary_channels = self(x)
        pred_heatmap = outputs[-1][:, :-1, :, :]
        #pred_landmarks = self.get_preds_fromhm(pred_heatmap)
        #landmarks = pred_landmarks * 4.0
        #eyes = torch.cat((landmarks[:, 96, :], landmarks[:, 97, :]), 1)
        return pred_heatmap[:, 96, :, :], pred_heatmap[:, 97, :, :]

    def extract_eyes(self, x, hm_left, hm_right, im_size=256):

        shift = im_size // 16

        hm_left = F.interpolate(hm_left[:, None, ...], size=im_size, mode='bilinear')
        hm_left_flat = torch.flatten(hm_left, start_dim=1)
        hm_left_argm = torch.argmax(hm_left_flat, dim=1)
        hm_left_xpos = hm_left_argm // im_size
        hm_left_ypos = hm_left_argm - hm_left_xpos * im_size

        hm_right = F.interpolate(hm_right[:, None, ...], size=im_size, mode='bilinear')
        hm_right_flat = torch.flatten(hm_right, start_dim=1)
        hm_right_argm = torch.argmax(hm_right_flat, dim=1)
        hm_right_xpos = hm_right_argm // im_size
        hm_right_ypos = hm_right_argm - hm_right_xpos * im_size

        hm_left_xpos += shift
        hm_left_ypos += shift
        hm_right_xpos += shift
        hm_right_ypos += shift

        eye_patches_left = []
        eye_patches_right = []

        for im, x_l, y_l, x_r, y_r in zip(x, hm_left_xpos, hm_left_ypos, hm_right_xpos, hm_right_ypos):
            im = F.pad(im, (shift, shift, shift, shift))
            eye_patches_left.append(im[:, x_l - shift: x_l + shift, y_l - shift: y_l + shift][None, ...])
            eye_patches_right.append(im[:, x_r - shift: x_r + shift, y_r - shift: y_r + shift][None, ...])

        return torch.concat(eye_patches_left, dim=0), torch.concat(eye_patches_right, dim=0)

