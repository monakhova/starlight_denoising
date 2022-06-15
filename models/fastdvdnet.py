"""
Definition of the FastDVDnet model

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn

class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames*(4), num_in_frames*self.interm_ch, \
                      kernel_size=3, padding=1, groups=num_in_frames, bias=False),
            nn.BatchNorm2d(num_in_frames*self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)

class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)

class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)

class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.convblock(x)

class DenBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=3):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=4)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in0, in1, in2):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
        '''
        # Input convolution block

        #x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        x0 = self.inc(torch.cat((in0, in1, in2), dim=1))

        
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1+x2)
        # Estimation
        x = self.outc(x0+x1)

        # Residual
        x = in1 - x

        return x

class DenBlockUnet(nn.Module):
    """ Definition of the denosing block of FastDVDnet, adopted to Unet with single input
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=4):
        super(DenBlockUnet, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=4)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in0):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
        '''
        # Input convolution block

        #x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        x0 = self.inc(in0)

        
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1+x2)
        # Estimation
        x = self.outc(x0+x1)

        # Residual
        x = in0 - x

        return x

class FastDVDnet(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
    """

    def __init__(self, num_input_frames=5):
        super(FastDVDnet, self).__init__()
        self.num_input_frames = num_input_frames
        # Define models of each denoising stage
        self.temp1 = DenBlock(num_input_frames=3)
        self.temp2 = DenBlock(num_input_frames=3)
        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
        '''
        # Unpack inputs
        C = 4
        (x0, x1, x2, x3, x4) = tuple(x[:, C*m:C*m+C, :, :] for m in range(self.num_input_frames))
        x0 = x[:,:,0]
        x1 = x[:,:,1]
        x2 = x[:,:,2]
        x3 = x[:,:,3]
        x4 = x[:,:,4]
       
        # First stage
        x20 = self.temp1(x0, x1, x2)
        x21 = self.temp1(x1, x2, x3)
        x22 = self.temp1(x2, x3, x4)

        #Second stage
        x = self.temp2(x20, x21, x22)

        return x
    
class FastDVDnetHR(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
    """

    def __init__(self, num_input_frames=5):
        super(FastDVDnetHR, self).__init__()
        self.num_input_frames = num_input_frames
        
        import helper.hr_helper as hr
        self.temp1 = hr.load_2d_hrnet2(num_channels=4*3, num_classes = 4)
        self.temp2 = hr.load_2d_hrnet2(num_channels=4*3, num_classes = 4)
        
    def forward(self, x):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
        '''
        # Unpack inputs
        C = 4
        x0 = x[:,:,0]
        x1 = x[:,:,1]
        x2 = x[:,:,2]
        x3 = x[:,:,3]
        x4 = x[:,:,4]
       
        # First stage
        x20 = self.temp1(torch.cat((x0, x1, x2), dim=1))
        x21 = self.temp1(torch.cat((x1, x2, x3), dim=1))
        x22 = self.temp1(torch.cat((x2, x3, x4), dim=1))

        #Second stage
        x = self.temp2(torch.cat((x20, x21, x22), dim=1))

        return x
    
class FastDVDnetHRie(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
    """

    def __init__(self, num_input_frames=5):
        super(FastDVDnetHRie, self).__init__()
        self.num_input_frames = num_input_frames
        
        import models.ienet as ie
        self.temp1 = ie.make_ienet()
        self.temp2 = ie.make_ienet()
        
    def forward(self, x):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
        '''
        # Unpack inputs
        C = 4
        x0 = x[:,:,0]
        x1 = x[:,:,1]
        x2 = x[:,:,2]
        x3 = x[:,:,3]
        x4 = x[:,:,4]
       
        # First stage
        x20 = self.temp1(torch.cat((x0, x1, x2), dim=1))
        x21 = self.temp1(torch.cat((x1, x2, x3), dim=1))
        x22 = self.temp1(torch.cat((x2, x3, x4), dim=1))

        #Second stage
        x = self.temp2(torch.cat((x20, x21, x22), dim=1))

        return x

class FastDVDnetHR16(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
    """

    def __init__(self, num_input_frames=16):
        super(FastDVDnetHR16, self).__init__()
        self.num_input_frames = num_input_frames
        
        import helper.hr_helper as hr
        self.temp1 = hr.load_2d_hrnet2(num_channels=4*8, num_classes = 4)
        self.temp2 = hr.load_2d_hrnet2(num_channels=4*3, num_classes = 4)
        
    def forward(self, x):
        '''Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
        '''
       
        # First stage
        xs1 = []
        for i in range(0,3):
            j=i*4
            x_in = torch.cat((x[:,:,j], x[:,:,j+1],
                              x[:,:,j+2], x[:,:,j+3],
                              x[:,:,j+4], x[:,:,j+5],
                              x[:,:,j+6], x[:,:,j+7]), dim=1)
            
            xs1.append(self.temp1(x_in))

        # second stage    
        x = self.temp2(torch.cat((xs1), dim=1))

        return x