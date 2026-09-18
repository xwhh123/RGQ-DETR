import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from src.core import register

from timm.models.vision_transformer import Block as ViTBlock
from timm.models.swin_transformer import SwinTransformerBlock
from .plug.MSDCA import MSDCA
from .plug.SRFP import SRFP


__all__ = ['HybridEncoder']


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        del self.conv1
        del self.conv2

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)



# transformer
class ComplexityRouter(nn.Module):
    """Learns to predict image complexity from C4 features.

    During training: always runs frozen encoder, router learns a self-supervised
    target derived from ||frozen_output|| / ||c4_input|| (how much the frozen
    branch contributes).

    During inference: router < 0.5 → skip frozen encoder (simple image, save FLOPs).
    """
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: [B, C, H, W] → complexity score [B, 1]"""
        dtype = x.dtype
        pooled = self.gap(x.float()).flatten(1)  # GAP stable in fp32
        return self.mlp(pooled).to(dtype)


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output
        
        
class FiLMChannel_yuan(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C//4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C//4, 2*C, 1)
        )
    def forward(self, c2, ck):
        gamma_beta = self.mlp(c2)           # [B,2C,1,1]
        gamma, beta = gamma_beta.chunk(2,1) # 广播到 ck 的 H×W
        return ck * torch.sigmoid(gamma) + beta


class FiLMChannel(nn.Module):
    def __init__(self, cond_channels, target_channels, reduction=8):
        super().__init__()
        self.cond_channels = cond_channels

        self.multi_scale = nn.ModuleList([
            nn.Conv2d(cond_channels, cond_channels//4, 3, padding=1, dilation=1, groups=cond_channels//4),
            nn.Conv2d(cond_channels, cond_channels//4, 3, padding=3, dilation=3, groups=cond_channels//4),
            nn.Conv2d(cond_channels, cond_channels//4, 3, padding=5, dilation=5, groups=cond_channels//4),
        ])

        # 关键：输入通道是 cat 后的通道数
        fused_channels = cond_channels//4 * 3 + cond_channels  # 192 + 256 = 448
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(fused_channels, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # SE 部分不变（全局池化后通道不影响）
        mid_channels = max(target_channels // reduction, 32)
        self.se_gamma = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_channels, mid_channels, 1),   # ← 改这里！
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, target_channels, 1),
            nn.Sigmoid()
        )
        self.se_beta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_channels, mid_channels, 1),   # ← 改这里！
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, target_channels, 1)
        )

        self.fusion = nn.Conv2d(target_channels, target_channels, 3, padding=1, groups=target_channels)

    def forward(self, c2, ck):
        feats = [conv(c2) for conv in self.multi_scale]
        fused = torch.cat(feats + [c2], dim=1)          # [B, 448, H, W]

        spatial_gate = self.spatial_attn(fused)
        fused = fused * spatial_gate

        gamma = self.se_gamma(fused)
        beta = self.se_beta(fused)

        out = ck * gamma + beta
        out = out + self.fusion(out)
        return out


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 feat_strides=[4, 8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[3],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 use_msdca=True,
                 use_film=True,
                 use_cafe=True,
                 use_srfp=False):
        super().__init__()
        self.use_msdca = use_msdca
        self.use_film = use_film
        self.use_cafe = use_cafe
        self.use_srfp = use_srfp
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        self.msdca = MSDCA(hidden_dim)

        # SRFP: Structurally-Regularized Feature Purification for P2 (with annealed TV)
        self.srfp = SRFP(hidden_dim) if use_srfp else None

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)
        
        
        # 关键修改 3：为 C3 和 C4 分别做轻量 adapter（必须！通道对不齐会炸）
        self.adapter_c3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.adapter_c4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        # 关键修改 4：每个层独立的可学习融合强度（必须独立！）
        self.res_scale_c3 = nn.Parameter(torch.tensor(0.6))
        self.res_scale_c4 = nn.Parameter(torch.tensor(0.9))


        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])
    
        if self.num_encoder_layers > 0:
            # 深拷贝 C5 的 encoder，冻结，只注入 C4
            self.frozen_encoder_for_low = copy.deepcopy(self.encoder[0])
            for p in self.frozen_encoder_for_low.parameters():
                p.requires_grad = False
            self.frozen_encoder_for_low.eval()

            # Complexity-aware router: learns when frozen encoder helps
            self.router = ComplexityRouter(hidden_dim)

        # Deploy mode: always-on frozen encoder for ONNX export (no branching)
        self._deploy = False
        # ── Ablation switches (set via __init__ params) ──

        # C2融合因子
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        
        # 共享 FiLM 权重，仅 per-level alpha 独立（省 0.25M）
        self.c2_to_ck_film = FiLMChannel(cond_channels=hidden_dim, target_channels=hidden_dim)
        # self.alpha = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))  # 对 C3/C4/C5 的加权
        self.alpha = nn.ParameterList([
            torch.ones(1) * 0.5,   # 可学习融合系数
            torch.ones(1) * 0.8,
            torch.ones(1) * 1.0,
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            # for idx in range(len(self.in_channels)):
            #need_idx = set(self.use_encoder_idx) | {2}
            for idx in self.use_encoder_idx:
            #for idx in need_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # SRFP: purify P2 before FPN fusion (annealed TV-L1)
        if self.use_srfp:
            proj_feats[0] = self.srfp(proj_feats[0])

        if self.use_msdca: proj_feats[0] = self.msdca(proj_feats[0])
        
        
        #显存开销太大了。。
        # 我忽然的点子，C2作为query

            #         if self.use_film:
            #     if isinstance(self.c2_to_ck_film, nn.ModuleList):
            #         mod = self.c2_to_ck_film[i-0](proj_feats[0], proj_feats[idx])   # FiLM(c2, ck)
            #     else:
            #         mod = self.c2_to_ck_film(proj_feats[0], proj_feats[idx])        # 共享权重版本
            #     # 残差插值：ck <- ck + alpha*(mod - ck)，等价于 (1-alpha)*ck + alpha*mod
            #     proj_feats[idx] = proj_feats[idx] + self.alpha[i] * (mod - proj_feats[idx])
        if self.use_film:
            for i, idx in enumerate([1, 2, 3]):
                mod = self.c2_to_ck_film(proj_feats[0], proj_feats[idx])  # P2 source (rich detail)
                proj_feats[idx] = proj_feats[idx] + self.alpha[i] * (mod - proj_feats[idx])

        # Step 2: Complexity-aware Frozen Encoder (C4 only) [ablation: self.use_cafe]
        if not self.use_cafe:
            self._router_score = None
            self._router_target = None
        # else: skip frozen (simple image)
        if self.num_encoder_layers > 0 and hasattr(self, 'frozen_encoder_for_low') and self.use_cafe:
            c4 = proj_feats[2]
            B, _, H4, W4 = c4.shape

            # Router predicts complexity from C4 features
            router_score = self.router(c4)  # [B, 1]

            if self.training:
                adapted_c4 = self.adapter_c4(c4)
                src_c4 = adapted_c4.flatten(2).permute(0, 2, 1)
                pos_c4 = self.build_2d_sincos_position_embedding(
                    W4, H4, self.hidden_dim, self.pe_temperature).to(src_c4.device)

                # Frozen encoder on C4
                with torch.no_grad():
                    mem_c4 = self.frozen_encoder_for_low(src_c4, pos_embed=pos_c4)
                    mem_c4 = mem_c4.permute(0, 2, 1).reshape(B, self.hidden_dim, H4, W4)

                # Fix 1: Cross-Scale Feature Alignment
                # Live C5 encoder also processes C4 → cosine with frozen output
                with torch.no_grad():
                    mem_c4_live = self.encoder[0](src_c4, pos_embed=pos_c4)
                    mem_c4_live = mem_c4_live.permute(0, 2, 1).reshape(
                        B, self.hidden_dim, H4, W4)
                self._distill_loss = (1.0 - F.cosine_similarity(
                    mem_c4.flatten(1).float(),
                    mem_c4_live.flatten(1).float(), dim=-1)).mean()

                # Fix 2: Momentum Evolving Teacher (EMA θ_T ← m·θ_T + (1-m)·θ_E)
                with torch.no_grad():
                    for p_t, p_e in zip(self.frozen_encoder_for_low.parameters(),
                                        self.encoder[0].parameters()):
                        p_t.data.mul_(0.9999).add_(p_e.data, alpha=0.0001)

                # Router self-supervised target
                with torch.no_grad():
                    contrib = (mem_c4.float().norm(p=2, dim=[1, 2, 3]) /
                               (c4.float().norm(p=2, dim=[1, 2, 3]) + 1e-6))
                    self._router_target = (contrib / contrib.max().clamp(min=1e-6))
                self._router_score = router_score.squeeze(-1)
                self._used_frozen = None
                proj_feats[2] = c4 + (self.res_scale_c4 * mem_c4).to(c4.dtype)
            elif self._deploy:
                # Deploy mode: always-on frozen encoder (ONNX export)
                adapted_c4 = self.adapter_c4(c4)
                src_c4 = adapted_c4.flatten(2).permute(0, 2, 1)
                pos_c4 = self.build_2d_sincos_position_embedding(
                    W4, H4, self.hidden_dim, self.pe_temperature).to(src_c4.device)
                with torch.no_grad():
                    mem_c4 = self.frozen_encoder_for_low(src_c4, pos_embed=pos_c4)
                    mem_c4 = mem_c4.permute(0, 2, 1).reshape(B, self.hidden_dim, H4, W4)
                proj_feats[2] = c4 + (self.res_scale_c4 * mem_c4).to(c4.dtype)
                self._used_frozen = None
                self._router_score = None
                self._router_target = None
            else:
                # Inference: adaptive routing (per-sample gate)
                use_frozen = (router_score > 0.5).squeeze(-1)  # [B] bool
                self._used_frozen = use_frozen
                self._router_score = router_score.squeeze(-1)
                self._router_target = None

                adapted_c4 = self.adapter_c4(c4)
                src_c4 = adapted_c4.flatten(2).permute(0, 2, 1)
                pos_c4 = self.build_2d_sincos_position_embedding(
                    W4, H4, self.hidden_dim, self.pe_temperature).to(src_c4.device)
                mem_c4_all = torch.zeros_like(c4)

                for b in range(B):
                    if use_frozen[b]:
                        with torch.no_grad():
                            mf = self.frozen_encoder_for_low(
                                src_c4[b:b+1], pos_embed=pos_c4)
                            mf = mf.permute(0, 2, 1).reshape(1, self.hidden_dim, H4, W4)
                        mem_c4_all[b:b+1] = mf
                proj_feats[2] = c4 + (self.res_scale_c4 * mem_c4_all).to(c4.dtype)


        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        
        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
        #for idx in range(len(self.in_channels) - 1, 1, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
        #for idx in range(len(inner_outs) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        # Drop C2 (stride-4) if present — decoder expects C3/C4/C5 only
        if len(outs) > 3:
            outs = outs[1:]

        return outs
