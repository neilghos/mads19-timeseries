import torch

import torch.nn as nn

import torch.nn.functional as F

from torchlibrosa.stft import Spectrogram, LogmelFilterBank

from torchlibrosa.augmentation import SpecAugmentation

from pytorch_utils import do_mixup, interpolate, pad_framewise_output

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x

class ConvBlock5x5(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x

class Cnn6(nn.Module):

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super(Cnn6, self).__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size, window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict

class Cnn10(nn.Module):

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super(Cnn10, self).__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size, window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict

class MobileNetV1(nn.Module):

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super(MobileNetV1, self).__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size, window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)

        def conv_bn(inp, oup, stride):
            _layers = [nn.Conv2d(inp, oup, 3, 1, 1, bias=False), nn.AvgPool2d(stride), nn.BatchNorm2d(oup), nn.ReLU(inplace=True)]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_dw(inp, oup, stride):
            _layers = [nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False), nn.AvgPool2d(stride), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True)]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            return _layers
        self.features = nn.Sequential(conv_bn(1, 32, 2), conv_dw(32, 64, 1), conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2), conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 1024, 1))
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        x = self.features(x)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super(MobileNetV2, self).__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size, window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)
        width_mult = 1.0
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 2], [6, 160, 3, 1], [6, 320, 1, 1]]

        def conv_bn(inp, oup, stride):
            _layers = [nn.Conv2d(inp, oup, 3, 1, 1, bias=False), nn.AvgPool2d(stride), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True)]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        self.fc1 = nn.Linear(1280, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        x = self.features(x)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict

