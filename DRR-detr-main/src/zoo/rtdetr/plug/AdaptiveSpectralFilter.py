import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple


class StarReLU(nn.Module):
    """StarReLU: s * relu(x) ** 2 + b"""
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)

    # 插值重采样
    new_weight = F.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)

    # L2 归一化权重（防止频域爆炸）
    new_weight = F.normalize(new_weight, dim=-1)
    return new_weight


class AdaptiveSpectralFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, weight_resize=True, **kwargs):
        super().__init__()
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize

        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)

        self.complex_weights = nn.Parameter(
            torch.randn(16, 9, num_filters, 2, dtype=torch.float32) * 0.02)

        self.alpha_param = nn.Parameter(torch.tensor(0.5))
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, _ = x.shape
        filter_size = W // 2 + 1

        # 路由 MLP 输出 reshape 显式指定维度
        routeing = self.reweight(x.mean(dim=(1, 2)))
        routeing = routeing.view(B, self.num_filters, self.med_channels).softmax(dim=1)

        # 空域分支
        x_spatial = self.pwconv1(x)
        x_spatial = self.act1(x_spatial)

        # 频域分支
        x_freq = x_spatial.to(torch.float32)
        x_freq = torch.fft.rfft2(x_freq, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, H, filter_size)
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            if H != self.complex_weights.shape[0] or filter_size != self.complex_weights.shape[1]:
                raise ValueError(f"Expected input size {(self.complex_weights.shape[0], self.complex_weights.shape[1])}, "
                                 f"but got {(H, filter_size)}. Set weight_resize=True to enable resizing.")
            complex_weights = torch.view_as_complex(self.complex_weights)

        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)

        weight = weight.view(B, H, filter_size, self.med_channels)
        x_freq = x_freq * weight
        x_freq = torch.fft.irfft2(x_freq, s=(H, W), dim=(1, 2), norm='ortho')

        # 空频融合
        alpha = torch.sigmoid(self.alpha_param)
        x_fused = alpha * x_spatial + (1 - alpha) * x_freq

        x_fused = self.act2(x_fused)
        x_out = self.pwconv2(x_fused)
        return x_out


if __name__ == '__main__':
    block = AdaptiveSpectralFilter(32)  # 无需设置 size
    print(block)

    input = torch.rand(3, 32, 48, 48)   # B C H W 任意大小
    input_bhwc = input.permute(0, 2, 3, 1)  # B H W C

    output = block(input_bhwc)
    output = output.permute(0, 3, 1, 2)  # B C H W

    print(input.size())
    print(output.size())  # 输出应与输入一致

    input = torch.rand(3, 32, 108, 108)
    input_bhwc = input.permute(0, 2, 3, 1)
    output = block(input_bhwc).permute(0, 3, 1, 2)

    print(input.size())
    print(output.size())
