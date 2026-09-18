from typing import Optional
import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")



#MSDCA
class MSDCA(nn.Module):
    """优化版上下文锚点注意力模块，增强小目标检测能力"""
    def __init__(self,
                 channels: int,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        self.channels = channels

        # 增强型空间注意力（多尺度）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)  # 全局上下文

        # 通道调整
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                               norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 多尺度方向卷积（并行结构）
        self.h_conv = nn.ModuleList([
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels),
            nn.Conv2d(channels, channels, (1, 5), padding=(0, 2), groups=channels)
        ])
        self.v_conv = nn.ModuleList([
            nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels),
            nn.Conv2d(channels, channels, (5, 1), padding=(2, 0), groups=channels)
        ])

        # 轻量级通道注意力（降低压缩比，保留更多通道信息）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(32, channels//4), 1),  # 压缩比从 8x → 4x
            nn.ReLU(inplace=True),
            nn.Conv2d(max(32, channels//4), channels, 1),
            nn.Sigmoid()
        )

        # 特征融合
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                               norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 动态权重参数
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

        #新改进
        # 在 self.gamma = ... 之后添加
        self.attn_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]))

    def forward(self, x):
        # 增强空间注意力
        spatial_mask = self.spatial_attn(x)
        global_context = self.spatial_pool(x)

        # 10% 阈值软门控：sort-based quantile (ONNX compatible)
        B, C, H, W = spatial_mask.shape
        flat_mask = spatial_mask.float().view(B, -1)
        k = max(int(flat_mask.shape[1] * 0.1), 1)
        sorted_mask, _ = torch.sort(flat_mask, dim=1)
        threshold = sorted_mask[:, k:k+1].view(B, 1, 1, 1)
        gated_mask = (spatial_mask > threshold).float().detach()

        # 多尺度方向感知
        feat = self.conv1(x)

        h_feat = sum(conv(feat) for conv in self.h_conv)
        v_feat = sum(conv(feat) for conv in self.v_conv)
        direction_feat = h_feat + v_feat

        # 通道注意力
        channel_attn = self.se(x)

        # 注意力融合
        fused_feat = self.conv2(direction_feat)

        w = torch.softmax(self.attn_weights, dim=0)
        attn_factor = (
            w[0] * torch.sigmoid(fused_feat) +
            w[1] * channel_attn +
            w[2] * spatial_mask * global_context
        )

        # 输出增强（带通道级缩放 + 空间门控抑制 FP）
        enhanced = x * (1 + self.alpha * attn_factor * gated_mask)
        return self.gamma * enhanced + self.beta


# 测试模块
if __name__ == "__main__":
    caa = MSDCA(32)
    input_tensor = torch.rand(1, 32, 256, 256)
    print(f"输入张量的形状: {input_tensor.shape}")
    output_tensor = caa(input_tensor)
    print(f"输出张量的形状: {output_tensor.shape}")

    input_tensor = torch.rand(1, 32, 34, 34)
    print(f"输入张量的形状: {input_tensor.shape}")
    output_tensor = caa(input_tensor)
    print(f"输出张量的形状: {output_tensor.shape}")

    input_tensor = torch.rand(1, 33, 34, 34)
    print(f"输入张量的形状: {input_tensor.shape}")
    output_tensor = caa(input_tensor)
    print(f"输出张量的形状: {output_tensor.shape}")
