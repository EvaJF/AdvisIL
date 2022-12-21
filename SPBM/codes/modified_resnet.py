#import torch.nn as nn
#import torch.utils.model_zoo as model_zoo
#import modified_linear
#
#def conv3x3(in_planes, out_planes, stride=1):
#    """3x3 convolution with padding"""
#    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                     padding=1, bias=False)
#
#def _make_divisible(v, divisor, min_value=None):
#    """
#    This function is taken from the original tf repo.
#    It ensures that all layers have a channel number that is divisible by 8
#    It can be seen here:
#    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#    :param v:
#    :param divisor:
#    :param min_value:
#    :return:
#    """
#    if min_value is None:
#        min_value = divisor
#    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#    # Make sure that round down does not go down by more than 10%.
#    if new_v < 0.9 * v:
#        new_v += divisor
#    return int(new_v)
#
#class BasicBlock(nn.Module):
#    expansion = 1
#
#    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
#        super(BasicBlock, self).__init__()
#        self.conv1 = conv3x3(inplanes, planes, stride)
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.relu = nn.ReLU(inplace=True)
#        self.conv2 = conv3x3(planes, planes)
#        self.bn2 = nn.BatchNorm2d(planes)
#        self.downsample = downsample
#        self.stride = stride
#        self.last = last
#
#    def forward(self, x):
#        residual = x
#
#        out = self.conv1(x)
#        out = self.bn1(out)
#        out = self.relu(out)
#
#        out = self.conv2(out)
#        out = self.bn2(out)
#
#        if self.downsample is not None:
#            residual = self.downsample(x)
#
#        out += residual
#        if not self.last: #remove ReLU in the last layer
#            out = self.relu(out)
#
#        return out
#
#class ResNet(nn.Module):
#
#    def __init__(self, block, layers, width_mult=1.0, depth_mult=1.0, num_classes=1000):
#        self.inplanes = 64
#        super(ResNet, self).__init__()
#
#        layers = [_make_divisible(d*depth_mult, 1) for d in layers] 
#        channels = [64, 128, 256, 512] # previous hard-coded version
#        channels = [_make_divisible(c*width_mult, 8) for c in channels] # scaled
#        self.config = {"bloc" : "resnetBasic", "layers":layers, "channels": channels}
#
#        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                               bias=False)
#        self.bn1 = nn.BatchNorm2d(64)
#        self.relu = nn.ReLU(inplace=True)
#        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#        self.layer1 = self._make_layer(block, channels[0], layers[0])
#        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
#        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
#        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, last_phase=True)
#        self.avgpool = nn.AvgPool2d(7, stride=1)
#        self.fc = modified_linear.CosineLinear(channels[3] * block.expansion, num_classes)
#
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.constant_(m.weight, 1)
#                nn.init.constant_(m.bias, 0)
#
#    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
#        downsample = None
#        if stride != 1 or self.inplanes != planes * block.expansion:
#            downsample = nn.Sequential(
#                nn.Conv2d(self.inplanes, planes * block.expansion,
#                          kernel_size=1, stride=stride, bias=False),
#                nn.BatchNorm2d(planes * block.expansion),
#            )
#
#        layers = []
#        layers.append(block(self.inplanes, planes, stride, downsample))
#        self.inplanes = planes * block.expansion
#        if last_phase:
#            for i in range(1, blocks-1):
#                layers.append(block(self.inplanes, planes))
#            layers.append(block(self.inplanes, planes, last=True))
#        else: 
#            for i in range(1, blocks):
#                layers.append(block(self.inplanes, planes))
#
#        return nn.Sequential(*layers)
#
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.relu(x)
#        x = self.maxpool(x)
#
#        x = self.layer1(x)
#        x = self.layer2(x)
#        x = self.layer3(x)
#        x = self.layer4(x)
#
#        x = self.avgpool(x)
#        x = x.view(x.size(0), -1)
#        x = self.fc(x)
#
#        return x
#
#def resnet18(pretrained=False, **kwargs):
#    """Constructs a ResNet-18 model.
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#    return model
#
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import modified_linear
import torch 
### DEBUG 


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, last=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if not self.last: #remove ReLU in the last layer
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 16.)) * groups # modification : 16. instead of 64. for more flexibility in down-scaling
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# nb : default config of myResNet is ResNet50 with layers 3/4/6/3
# for a Resnet18 base network, set block=BasicBlock, layers = [2, 2, 2, 2] and keep width_per_group=64
# no scaling of width_per_group for basicblocks.

class ResNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000, 
                 zero_init_residual=False,
                 groups=1, width_per_group=64, width_mult=1.0, depth_mult=1.0,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        # scaling : depth --> layers tuple
        layers = [_make_divisible(d*depth_mult, 1) for d in layers] 
        # look at the make layers below --> replace the hard-coded channel numbers by scalable ones.
        channels = [64, 128, 256, 512] # previous hard-coded version
        channels = [_make_divisible(c*width_mult, 8) for c in channels] # scaled; 16 as divisor previously 
        # scaling of width_per_group for bottleneck blocs only
        if block == Bottleneck:
            width_per_group = _make_divisible(width_per_group*width_mult, 16) # for compatibility with Bottleneck layer
            block_type = "bottleneck"
        else : 
            block_type="basic"

        self.config = {"bloc" : block_type, "layers":layers, "width_per_group":width_per_group, "channels": channels}
        self.num_classes = num_classes
        
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels[0], layers[0]) #64 
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0]) # 128
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1]) # 256
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], last_phase=True) # 512
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) mine
        self.avgpool = nn.AvgPool2d(7, stride=1) # theirs (lucir)
        self.fc = modified_linear.CosineLinear(channels[-1] * block.expansion, num_classes) # todo modify hard-coded 512 ? channels[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, last_phase=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i < blocks or not last_phase :
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            else : 
                layers.append(block(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer, last=True))
            

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print("no pretraining implemented")
        pass
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)
