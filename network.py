import torch
import torch.nn as nn


class ComplexConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2D, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, input):
        real = self.conv_real(input.real) - self.conv_imag(input.imag)
        imag = self.conv_real(input.imag) + self.conv_imag(input.real)
        output = torch.stack([real, imag], dim=-1)

        return output


class PhaseEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,1), stride=1):
        super(PhaseEncoder, self).__init__()
        self.padding = nn.ConstantPad2d((0, 0, kernel_size[0]-1, 0), 0)
        self.conv = ComplexConv2D(in_channels, out_channels, kernel_size, stride)

    def forward(self, input):
        # (B, C, T, F) complex
        input = input.permute(0, 1, 3, 2)
        input = self.padding(input)
        
        # (B, C, T, F)
        output = self.conv(input)
        output = torch.pow(output[...,0]**2 + output[...,1]**2 + 1e-12, 0.25)
        return output 
    
    
class Chomp_T(nn.Module):
    def __init__(self, t):
        super(Chomp_T, self).__init__()
        self.t = t

    def forward(self, x):
        return x[:, :, :-self.t, :]


class FrequencyDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(FrequencyDown, self).__init__()
        self.fd = nn.Sequential(
            nn.ZeroPad2d((0, 0, kernel_size[0] -1,0)),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, input):
        return self.fd(input)


class FrequencyUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups):
        super(FrequencyUp, self).__init__()
        self.pconv1 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(in_channels),
            nn.Tanh(),
        )
        self.pconv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(in_channels),
        )
        
        k_t = kernel_size[0]
        if k_t > 1:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, output_padding=output_padding, groups=groups),
                Chomp_T(k_t-1),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
            )
        else:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, output_padding=output_padding, groups=groups),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
            )

    def forward(self, fu, fd):
        input = torch.cat([fu, fd], 1)
        output = self.pconv1(input) * fd
        output = self.pconv2(output)
        output = self.deconv(output)

        return output


class TFCM_block(nn.Module):
    def __init__(self, in_channels, dila, kernel_size=(3,3)):
        super(TFCM_block, self).__init__()
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )
        self.dila_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, dila * (kernel_size[0] - 1),0), 0),
            nn.Conv2d(in_channels, in_channels, kernel_size, dilation=(dila,1), groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )

        self.pconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))

    def forward(self, input):
        output = self.pconv1(input)
        output = self.dila_conv(input)
        output = self.pconv2(output)
        return input + output


class TFCM(nn.Module):
    def __init__(self, in_channels, layers=6):
        super(TFCM, self).__init__()
        self.tfcm_blks = nn.ModuleList()
        for i in range(layers):
            self.tfcm_blks.append(TFCM_block(in_channels, 2**i))

    def forward(self, input):
        output = input
        for block in self.tfcm_blks:
            output = block(output)

        return output


class GCAFA(nn.Module):
    def __init__(self, in_channels):
        super(GCAFA, self).__init__()
        self.att_channel = in_channels // 2
        self.gated_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.att_channel * 3, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(self.att_channel*3),
            nn.PReLU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(self.att_channel, in_channels, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )
        
        self.gated_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )
        
    def forward(self, input):
        # input: (B, C, T, F)
        f_qkv = self.gated_conv1(input) # (B, 3C, T, F)
        qf, kf, v = torch.chunk(f_qkv, chunks=3, dim=1)
        qf = qf.permute(0,2,3,1) # (B, T, F, C)
        kf = kf.permute(0,2,1,3) # (B, T, C, F)
        v = v.permute(0,2,3,1) # (B, T, F, C)
        f_score = torch.softmax(torch.matmul(qf, kf) / (self.att_channel**0.5), dim=-1) # (B, T, F, F)
        f_out = torch.matmul(f_score, v).permute(0,3,1,2) # (B, C, T, F)
        out = self.proj(f_out) + input
        out = self.gated_conv2(out) + out
        return out


class Encoder(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding, groups):
        super(Encoder, self).__init__()
        self.FD = FrequencyDown(in_channels, out_channels, kernel_size, stride, padding, groups)
        self.TFCM = TFCM(out_channels, layers=6)
        self.ASA = GCAFA(out_channels)
    
    def forward(self, input):
        fd_out = self.FD(input)
        output = self.TFCM(fd_out)
        output = self.ASA(output)
        return fd_out, output

class Decoder(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, groups, stride=(1,1), padding=(0,0), output_padding=(0,0)):
        super(Decoder, self).__init__()
        self.FU = FrequencyUp(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups)
        self.TFCM = TFCM(out_channels, layers=6)
        self.ASA = GCAFA(out_channels)
    
    def forward(self, input, fd_out):
        fu_out = self.FU(input, fd_out)
        output = self.TFCM(fu_out)
        output = self.ASA(output)
        return output


class MASK(nn.Module):
    def __init__(self, in_channels, mask_size):
        super(MASK, self).__init__()
        k_f = mask_size[1]
        self.unfold = torch.nn.Unfold(kernel_size=(k_f, 1),padding=((k_f-1)//2, 0))
        self.mask = nn.Conv2d(in_channels=in_channels, out_channels=k_f + 2, kernel_size=(1,1))
        self.k_f = k_f
        
    def forward(self, input, spec):
        # spec: (B, 1, F, T) complex
        input = input.permute(0,1,3,2) # (B, C, F, T)
        mask = self.mask(input) # (B, k_f+2, F, T)
        
        # (B, 1, F, T)
        mag = torch.abs(spec)
        pha = torch.angle(spec)
        # (B, 1 * sub_size, F*T) -> (B, sub_size, F, T)
        mag_unfold = self.unfold(mag).reshape(mag.shape[0], -1, mag.shape[2], mag.shape[3])

        # (B, sub_size, F, T)
        mask_s1 = mask[:,:self.k_f].sigmoid()
        # (B, 1, F, T)
        mag = torch.sum(mag_unfold * mask_s1, dim=1, keepdim=True)
        mask_s2_real = mask[:,self.k_f: self.k_f+1]
        mask_s2_imag = mask[:,self.k_f+1: self.k_f+2]
        mag_mask = torch.sqrt(torch.clamp(mask_s2_real**2 + mask_s2_imag**2, 1e-10)).tanh()
        pha_mask = torch.atan2(mask_s2_imag+1e-10, mask_s2_real+1e-10)
        real = torch.mean(mag * mag_mask * torch.cos(pha+pha_mask), dim=1)
        imag = torch.mean(mag * mag_mask * torch.sin(pha+pha_mask), dim=1)
        return torch.stack([real, imag],dim=-1)


class PLDNet(nn.Module):
    def __init__(self):
        super(PLDNet, self).__init__()
        self.PhaseEncoder = PhaseEncoder(in_channels=3, out_channels=4)
        
        self.encoder1 = Encoder(4, 16, kernel_size=(1,7), stride=(1,4), padding=(0,3), groups=2)
        self.encoder2 = Encoder(16, 24, kernel_size=(1,7), stride=(1,4), padding=(0,3), groups=2)
        self.encoder3 = Encoder(24, 40, kernel_size=(1,7), stride=(1,4), padding=(0,3), groups=2)
        
        self.bottleneck1 = nn.Sequential(
            TFCM(40, layers=6),
            TFCM(40, layers=6),
            GCAFA(40))
        self.bottleneck2 = nn.Sequential(
            TFCM(40, layers=6),
            TFCM(40, layers=6),
            GCAFA(40))
        
        self.decoder1 = Decoder(40, 24, kernel_size=(1,7), stride=(1,4), padding=(0,3), output_padding=(0,0), groups=2)
        self.decoder2 = Decoder(24, 16, kernel_size=(1,7), stride=(1,4), padding=(0,3) ,output_padding=(0,0), groups=2)
        self.decoder3 = Decoder(16, 4, kernel_size=(1,7), stride=(1,4), padding=(0,3), output_padding=(0,0), groups=2)
        self.mask = MASK(4, mask_size=(1,3))

    def forward(self, input):
        input = torch.complex(input[..., 0], input[..., -1])
        output = self.PhaseEncoder(input)
        
        fd_out1, output = self.encoder1(output)
        fd_out2, output = self.encoder2(output)
        fd_out3, output = self.encoder3(output)
        
        output = self.bottleneck1(output)
        output = self.bottleneck2(output)
        
        output = self.decoder1(output, fd_out3)
        output = self.decoder2(output, fd_out2)
        output = self.decoder3(output, fd_out1)
        output = self.mask(output, input[:, 0:1])
        return output
        

if __name__ == '__main__':
    from thop import profile, clever_format
    
    model = PLDNet()
    model.eval()
    
    T = 63
    x = torch.randn(1, 3, 257, T, 2)
    y = model(x)
    print(y.shape)
    
    flops, params = profile(model, [x], verbose=False)
    flops, params = clever_format([flops, params], "%.2f")
    print(f"total MACs: {flops}")
    print(f"total params: {params}")
    
    print("causality test")
    x0 = torch.randn(1, 3, 257, T, 2)
    x1 = torch.randn(1, 3, 257, T, 2)
    x2 = torch.randn(1, 3, 257, T, 2)
    x1 = torch.cat([x0, x1], dim=-2)
    x2 = torch.cat([x0, x2], dim=-2)
    y1 = model(x1)
    y2 = model(x2)
    print(torch.sum((y1[..., 0:T, :] - y2[..., 0:T, :]) ** 2))
    print(torch.sum((y1[..., T+1, :] - y2[..., T+1, :]) ** 2))
