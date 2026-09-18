"""by lyuwenyu
"""

import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob


from src.core import register


__all__ = ['RTDETRTransformer']



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class MSDeformableAttention(nn.Module):
    """Shape-Aware Dual-Branch Deformable Attention (SADA).

    Contour branch: samples along bbox edges for shape discrimination.
    Texture branch: samples inside bbox for leaf/color features.
    Learned gate fuses the two branches adaptively per query.
    """
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, use_sada=True):
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points
        self.use_sada = use_sada

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # --- standard deformable attn ---
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # --- Shape-Aware Dual-Branch ---
        # 可学习逐 head scale（sigmoid 门控），初始化为 0.1 让 SADA 逐步激活
        # 注意：bias 的 dim=2 是 num_heads，所以 scale 维度必须匹配 num_heads
        self.sada_scale = nn.Parameter(torch.full((num_heads,), 0.1))
        self.shape_offset_bias = nn.Sequential(
            nn.Linear(4, embed_dim // 4), nn.ReLU(),
            nn.Linear(embed_dim // 4, self.total_points * 2))
        self.texture_offset_bias = nn.Sequential(
            nn.Linear(4, embed_dim // 4), nn.ReLU(),
            nn.Linear(embed_dim // 4, self.total_points * 2))
        self.branch_gate = nn.Sequential(
            nn.Linear(embed_dim + 4, embed_dim // 4), nn.ReLU(),
            nn.Linear(embed_dim // 4, 1))

        self.ms_deformable_attn_core = deformable_attention_core_func
        self._reset_parameters()


    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)
        init.xavier_uniform_(self.value_proj.weight); init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight); init.constant_(self.output_proj.bias, 0)
        # Dual-branch: init small so standard attn dominates early
        for m in [self.shape_offset_bias, self.texture_offset_bias]:
            init.zeros_(m[-1].weight); init.zeros_(m[-1].bias)
        init.zeros_(self.branch_gate[-1].weight); init.zeros_(self.branch_gate[-1].bias)

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None,
                bbox_pred=None):
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        # ── Shape-Aware Dual-Branch: fp32, learnable MLP ──
        if bbox_pred is not None and self.use_sada:
            _bbox = bbox_pred.detach()
            _shape = torch.cat([_bbox[..., 2:],
                (_bbox[..., 2] / (_bbox[..., 3] + 1e-6)).unsqueeze(-1),
                (_bbox[..., 2] * _bbox[..., 3]).unsqueeze(-1)], dim=-1)
            _shape = torch.nan_to_num(_shape, nan=0.0)

            # Force fp32 for SADA branch (AMP float16 overflows softmax gradients)
            with torch.autocast(device_type='cuda', enabled=False):
                shape_b = self.shape_offset_bias(_shape.float()).reshape_as(sampling_offsets)
                tex_b = self.texture_offset_bias(_shape.float()).reshape_as(sampling_offsets)
                gate_in = torch.cat([query.float(), _shape.float()], dim=-1)
                gate = self.branch_gate(gate_in).sigmoid().view(bs, Len_q, 1, 1, 1, 1)
                bias = gate * shape_b + (1 - gate) * tex_b
                scale = self.sada_scale.sigmoid().view(1, 1, -1, 1, 1, 1)
                bias = bias.clamp(-0.1, 0.1) * scale

                sampling_offsets = (sampling_offsets.float() + bias.float()).to(sampling_offsets.dtype)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError("Last dim of reference_points must be 2 or 4, but get {} instead."
                             .format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)
        if output.isnan().any():
            import builtins as _b
            _b.print("[SADA-NaN] deform attn OUTPUT has NaN!", force=True)
        output = self.output_proj(output)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4, use_sada=True,):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, use_sada=use_sada)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # self._reset_parameters()

    # def _reset_parameters(self):
    #     linear_init_(self.linear1)
    #     linear_init_(self.linear2)
    #     xavier_uniform_(self.linear1.weight)
    #     xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                bbox_pred=None):   # ← shape-aware attn
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention (with shape-aware dual-branch)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask,
            bbox_pred=bbox_pred)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt.clamp(min=-65504, max=65504))

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                return_hs=False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_hs = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            # Shape-aware: pass current bbox pred to guide deformable sampling
            _bbox = ref_points_detach if i == 0 else ref_points

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed,
                           bbox_pred=_bbox)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training or return_hs:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
                if return_hs:
                    dec_hs.append(output)

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training or return_hs else inter_ref_bbox

        if return_hs:
            return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), torch.stack(dec_hs)
        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


@register
class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True, use_sada=True):

        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points, use_sada=use_sada)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0: 
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,)
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # P1: shared IoU pred head (all layers/aux/dn reuse, 385 params)
        self.iou_head = nn.Linear(hidden_dim, 1)

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        
        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)


    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits


    def forward(self, feats, targets=None):

        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        
        # prepare denoising training (skip if no targets)
        if self.training and self.num_denoising > 0 and targets is not None:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder (return_hs=True to get hidden states for shared IoU head)
        out_bboxes, out_logits, dec_hs = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            return_hs=True)
        out_ious = torch.stack([self.iou_head(hs) for hs in dec_hs])

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_ious, out_ious = torch.split(out_ious, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_ious': out_ious[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1], out_ious[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes, dn_out_ious)
                out['dn_meta'] = dn_meta

        return out

    def forward_semi(self, feats, input_query_label, input_query_bbox, attn_mask, dn_meta):
        """Semi-supervised forward pass with external queries.

        Query structure: [consistency_queries | dn_queries | regular_queries]

        Args:
            feats: encoder output feature maps list of [B,C,H,W]
            input_query_label: [bs, pad_size_1+pad_size_2, hidden_dim] combined content queries
            input_query_bbox: [bs, pad_size_1+pad_size_2, 4] combined reference points
            attn_mask: [total_queries, total_queries] attention mask
            dn_meta: dict with 'pad_size_1', 'pad_size_2', etc.

        Returns:
            out dict with:
                'pred_logits', 'pred_boxes': regular query outputs (last layer)
                'aux_outputs': intermediate regular query outputs
                'dn_outputs': denoising query outputs list
                'dn_meta': denoising meta info
                'hs': all decoder hidden states [num_layers, bs, total_q, hidden_dim]
        """
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        bs = memory.shape[0]

        # generate anchors and encoder output for regular queries
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors.to(memory.device)
            valid_mask = self.valid_mask.to(memory.device)

        memory_out = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory_out)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(
            enc_outputs_class.max(-1).values, self.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(
                1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(
                1, 1, enc_outputs_class.shape[-1]))

        # extract region features for regular queries
        if self.learnt_init_query:
            regular_target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            regular_target = output_memory.gather(
                dim=1, index=topk_ind.unsqueeze(-1).repeat(
                    1, 1, output_memory.shape[-1]))
            regular_target = regular_target.detach()

        # combine: [external_queries, regular_queries]
        target = torch.cat([input_query_label, regular_target], dim=1)
        init_ref_points_unact = torch.cat(
            [input_query_bbox, reference_points_unact.detach()], dim=1)

        # decoder forward with hidden states (return_hs for shared IoU head)
        out_bboxes, out_logits, hs = self.decoder(
            target, init_ref_points_unact, memory,
            spatial_shapes, level_start_index,
            self.dec_bbox_head, self.dec_score_head,
            self.query_pos_head, attn_mask=attn_mask, return_hs=True)
        out_ious = torch.stack([self.iou_head(h) for h in hs])

        # split outputs by query type
        pad_size_1 = dn_meta['pad_size_1']
        pad_size_2 = dn_meta['pad_size_2']

        # out_bboxes/out_logits/out_ious: [num_layers, bs, total_queries, ...]
        regular_outputs_coord = out_bboxes[:, :, pad_size_1 + pad_size_2:, :]
        regular_outputs_class = out_logits[:, :, pad_size_1 + pad_size_2:, :]
        regular_outputs_ious  = out_ious[:,  :, pad_size_1 + pad_size_2:, :]
        dn_outputs_coord = out_bboxes[:, :, pad_size_1:pad_size_1 + pad_size_2, :]
        dn_outputs_class = out_logits[:, :, pad_size_1:pad_size_1 + pad_size_2, :]
        dn_outputs_ious  = out_ious[:,  :, pad_size_1:pad_size_1 + pad_size_2, :]

        # build output dict
        out = {
            'pred_logits': regular_outputs_class[-1],
            'pred_boxes':  regular_outputs_coord[-1],
            'pred_ious':   regular_outputs_ious[-1],
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                regular_outputs_class[:-1], regular_outputs_coord[:-1], regular_outputs_ious[:-1])
            out['aux_outputs'].extend(
                self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

        out['dn_outputs'] = self._set_aux_loss(dn_outputs_class, dn_outputs_coord, dn_outputs_ious)
        out['dn_meta'] = dn_meta
        out['hs'] = hs

        # Kill NaN at source — before it enters loss computation graph
        for k in list(out.keys()):
            if isinstance(out[k], torch.Tensor):
                out[k] = torch.nan_to_num(out[k], nan=0.0)
            elif isinstance(out[k], list):
                out[k] = [{kk: torch.nan_to_num(vv, nan=0.0)
                           for kk, vv in d.items()} for d in out[k]]

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_iou=None):
        if outputs_iou is not None:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_ious': c}
                    for a, b, c in zip(outputs_class, outputs_coord, outputs_iou)]
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
