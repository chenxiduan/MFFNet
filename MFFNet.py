from torch import nn
import torch
import torch.nn.functional as F
from torchvision import models
from torch.nn import Module, Conv2d, Parameter, Softmax


def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)  # inplace=True
    )


class DecoderBlockAEM(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlockAEM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(True)

        self.AEM = AttentionEnhancementModule(in_channels // 4, n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        return self.AEM(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class KernelAttention(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(KernelAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        # self.exp_feature = exp_feature_map
        # self.tanh_feature = tanh_feature_map
        self.softplus_feature = softplus_feature_map
        self.eps = eps

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        K = self.softplus_feature(K)

        KV = torch.einsum("bmn, bcn->bmc", K, V)

        # att = torch.einsum("bnc, bcl->bnl", Q, K)
        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)

        # weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class AttentionEnhancementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionEnhancementModule, self).__init__()
        self.conv = conv3otherRelu(in_chan, out_chan)
        self.conv_atten = KernelAttention(out_chan)
        self.bn_atten = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        feat = self.conv(x)
        att = self.conv_atten(feat)
        return self.bn_atten(att)


class FeatureAggregationModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureAggregationModule, self).__init__()
        self.convblk = conv3otherRelu(in_chan, out_chan, 1, 1, 0)
        self.KAM = KernelAttention(out_chan)

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        # atten = self.CAM(self.KAM(feat))
        atten = self.KAM(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out





class MFFNet50(nn.Module):
    def __init__(self, band_num, class_num):
        super(MFFNet50, self).__init__()
        self.name = 'MFFNet50'
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = nn.Conv2d(band_num, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.FAM3 = FeatureAggregationModule(filters[3], filters[2])
        self.FAM2 = FeatureAggregationModule(filters[2], filters[1])
        self.FAM1 = FeatureAggregationModule(filters[1], filters[0])

        self.decoder4 = DecoderBlockAEM(filters[3], filters[2])
        self.decoder3 = DecoderBlockAEM(filters[2], filters[1])
        self.decoder2 = DecoderBlockAEM(filters[1], filters[0])
        self.decoder1 = DecoderBlockAEM(filters[0], filters[0] // 2)

        self.finaldeconv1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0] // 2, 64, 4, 2, 1),
            nn.ReLU(True)
        )
        self.finalconv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True)
        )
        self.finalconv3 = nn.Conv2d(32, class_num, 3, padding=1)

    def forward(self, x):
        cloud = x[:, 2:, :, :]
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.FAM3(self.decoder4(e4), e3)
        d3 = self.FAM2(self.decoder3(d4), e2)
        d2 = self.FAM1(self.decoder2(d3), e1)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalconv2(out)
        out = self.finalconv3(out)

        return out + cloud


class MFFNetBaseline(nn.Module):
    def __init__(self, band_num, class_num):
        super(MFFNetBaseline, self).__init__()
        self.name = 'MFFNetBaseline'
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = nn.Conv2d(band_num, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.conv3 = nn.Conv2d(filters[3], filters[2], 1)
        self.conv2 = nn.Conv2d(filters[2], filters[1], 1)
        self.conv1 = nn.Conv2d(filters[1], filters[0], 1)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0] // 2)

        self.finaldeconv1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0] // 2, 64, 4, 2, 1),
            nn.ReLU(True)
        )
        self.finalconv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True)
        )
        self.finalconv3 = nn.Conv2d(32, class_num, 3, padding=1)

    def forward(self, x):
        cloud = x[:, 2:, :, :]
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.conv3(torch.cat((self.decoder4(e4), e3), 1))
        d3 = self.conv2(torch.cat((self.decoder3(d4), e2), 1))
        d2 = self.conv1(torch.cat((self.decoder2(d3), e1), 1))
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalconv2(out)
        out = self.finalconv3(out)

        return out + cloud


class MFFNetBaselineAEM(nn.Module):
    def __init__(self, band_num, class_num):
        super(MFFNetBaselineAEM, self).__init__()
        self.name = 'MFFNetBaselineAEM'
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = nn.Conv2d(band_num, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.conv3 = nn.Conv2d(filters[3], filters[2], 1)
        self.conv2 = nn.Conv2d(filters[2], filters[1], 1)
        self.conv1 = nn.Conv2d(filters[1], filters[0], 1)

        self.decoder4 = DecoderBlockAEM(filters[3], filters[2])
        self.decoder3 = DecoderBlockAEM(filters[2], filters[1])
        self.decoder2 = DecoderBlockAEM(filters[1], filters[0])
        self.decoder1 = DecoderBlockAEM(filters[0], filters[0] // 2)

        self.finaldeconv1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0] // 2, 64, 4, 2, 1),
            nn.ReLU(True)
        )
        self.finalconv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True)
        )
        self.finalconv3 = nn.Conv2d(32, class_num, 3, padding=1)

    def forward(self, x):
        cloud = x[:, 2:, :, :]
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.conv3(torch.cat((self.decoder4(e4), e3), 1))
        d3 = self.conv2(torch.cat((self.decoder3(d4), e2), 1))
        d2 = self.conv1(torch.cat((self.decoder2(d3), e1), 1))
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalconv2(out)
        out = self.finalconv3(out)

        return out + cloud


class MAEUNet(nn.Module):
    def __init__(self, band_num, class_num):
        super(MAEUNet, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'MAEUNet'

        channels = [32, 32, 32, 32, 32]
        self.conv1 = nn.Sequential(
            conv3otherRelu(self.band_num, channels[0]),
            conv3otherRelu(channels[0], channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[0], channels[1]),
            conv3otherRelu(channels[1], channels[1]),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[1], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[2], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[3], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
        )

        self.AEM4 = AttentionEnhancementModule(channels[4] * 2, channels[3])
        self.AEM3 = AttentionEnhancementModule(channels[3] * 2, channels[2])
        self.AEM2 = AttentionEnhancementModule(channels[2] * 2, channels[1])
        self.AEM1 = AttentionEnhancementModule(channels[1] * 2, channels[0])

        self.decoder4 = DecoderBlock(channels[3], channels[2])
        self.decoder3 = DecoderBlock(channels[2], channels[1])
        self.decoder2 = DecoderBlock(channels[1], channels[0])
        self.decoder1 = DecoderBlock(channels[0], channels[0])

        self.conv10 = nn.Sequential(
            conv3otherRelu(channels[0], channels[0] // 2),
            nn.Conv2d(channels[0] // 2, self.class_num, kernel_size=1, stride=1),
        )

    def forward(self, x):
        cloud = x[:, 2:, :, :]
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.decoder4(conv5)
        conv6 = self.AEM4(torch.cat((deconv4, conv4), 1))

        deconv3 = self.decoder3(conv6)
        conv7 = self.AEM3(torch.cat((deconv3, conv3), 1))

        deconv2 = self.decoder2(conv7)
        conv8 = self.AEM2(torch.cat((deconv2, conv2), 1))

        deconv1 = self.decoder1(conv8)
        conv9 = self.AEM1(torch.cat((deconv1, conv1), 1))

        output = self.conv10(conv9)

        return output + cloud


class FFUNet(nn.Module):
    def __init__(self, band_num, class_num):
        super(FFUNet, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'FFUNet'

        channels = [32, 32, 32, 32, 32]
        self.conv1 = nn.Sequential(
            conv3otherRelu(self.band_num, channels[0]),
            conv3otherRelu(channels[0], channels[0]),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[0], channels[1]),
            conv3otherRelu(channels[1], channels[1]),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[1], channels[2]),
            conv3otherRelu(channels[2], channels[2]),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[2], channels[3]),
            conv3otherRelu(channels[3], channels[3]),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            conv3otherRelu(channels[3], channels[4]),
            conv3otherRelu(channels[4], channels[4]),
        )

        self.AEM4 = AttentionEnhancementModule(channels[4] * 2, channels[3])
        self.AEM3 = AttentionEnhancementModule(channels[3] * 2, channels[2])
        self.AEM2 = AttentionEnhancementModule(channels[2] * 2, channels[1])
        self.AEM1 = AttentionEnhancementModule(channels[1] * 2, channels[0])

        self.decoder4 = DecoderBlock(channels[3], channels[2])
        self.decoder3 = DecoderBlock(channels[2], channels[1])
        self.decoder2 = DecoderBlock(channels[1], channels[0])
        self.decoder1 = DecoderBlock(channels[0], channels[0])

        self.conv10 = nn.Sequential(
            conv3otherRelu(channels[0], channels[0] // 2),
            nn.Conv2d(channels[0] // 2, self.class_num, kernel_size=1, stride=1),
        )

        self.FAM = FeatureAggregationModule(class_num * 2, class_num)

    def forward(self, x):
        cloud = x[:, 2:, :, :]
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv4 = self.decoder4(conv5)
        conv6 = self.AEM4(torch.cat((deconv4, conv4), 1))

        deconv3 = self.decoder3(conv6)
        conv7 = self.AEM3(torch.cat((deconv3, conv3), 1))

        deconv2 = self.decoder2(conv7)
        conv8 = self.AEM2(torch.cat((deconv2, conv2), 1))

        deconv1 = self.decoder1(conv8)
        conv9 = self.AEM1(torch.cat((deconv1, conv1), 1))

        output = self.conv10(conv9)

        return self.FAM(output, cloud)


if __name__ == '__main__':
    num_classes = 10
    in_batch, inchannel, in_h, in_w = 4, 15, 256, 256
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = MFFNet50(15, 13)
    out = net(x)
    print(out.shape)
