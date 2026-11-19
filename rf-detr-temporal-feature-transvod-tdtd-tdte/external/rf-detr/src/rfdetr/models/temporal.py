# ------------------------------------------------------------------------
# RF-DETR temporal extensions for video object detection.
# Copyright (c) 2026.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Temporal query fusion modules inspired by TransVOD++.

RF-DETR is an image detector: every decoder query describes one candidate
object in the current image.  TransVOD++ adds temporal reasoning mostly at the
query level, after a spatial detector has produced query embeddings for the key
frame and neighbouring reference frames.

This file keeps that idea, but avoids copying TransVOD++'s full Sparse-RCNN/QRF
stack.  The module below is deliberately small and compatible with RF-DETR's
existing decoder outputs:

    current frame queries -> MultiHeadAttention -> reference frame queries

The result is still a normal ``hs`` tensor, so RF-DETR's classification head,
box head, criterion, and post-processing remain unchanged.
"""

from __future__ import annotations

'''
этот файл содержит сам temporal fusion module.
он берет query текущего кадра и через attention смотрит на query соседних
кадров, чтобы сохранить полезный контекст: где машина была раньше или рядом
во времени, даже если на текущем кадре она маленькая, перекрыта или смазана.
'''

import copy
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from rfdetr.models.ops.modules import MSDeformAttn
from rfdetr.models.transformer import TransformerDecoder


class TemporalQueryFusionLayer(nn.Module):
    """Fuse current-frame object queries with reference-frame queries.

    Args:
        d_model: Query feature dimension.
        nhead: Number of attention heads.
        dim_feedforward: Hidden size of the feed-forward block.
        dropout: Dropout probability.
        activation: Feed-forward activation name.

    Shape:
        ``query``: ``[B, N, C]``
        ``ref_query``: ``[B, T, N, C]`` or ``[B, T*N, C]``
        output: ``[B, N, C]``
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)  # current query смотрит на query соседних кадров
        self.dropout1 = nn.Dropout(dropout)  # dropout после attention
        self.norm1 = nn.LayerNorm(d_model)  # стабилизирует residual после attention

        self.linear1 = nn.Linear(d_model, dim_feedforward)  # первая часть ffn, как в transformer слое
        self.dropout = nn.Dropout(dropout)  # dropout внутри ffn
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # возвращаем размерность обратно в d_model
        self.dropout2 = nn.Dropout(dropout)  # dropout перед вторым residual
        self.norm2 = nn.LayerNorm(d_model)  # нормализация после ffn
        self.activation = _get_activation_fn(activation)  # выбираем relu/gelu/glu

    def forward(
        self,
        query: Tensor,
        ref_query: Tensor,
        query_pos: Optional[Tensor] = None,
        ref_query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        if ref_query.dim() == 4:
            batch_size, num_ref_frames, num_queries, channels = ref_query.shape  # ref задан как [b, t, n, c]
            ref_query = ref_query.reshape(batch_size, num_ref_frames * num_queries, channels)  # attention ждет одну ось tokens
            if ref_query_pos is not None:
                ref_query_pos = ref_query_pos.reshape(batch_size, num_ref_frames * num_queries, channels)  # позиционные признаки выравниваем так же

        q = query if query_pos is None else query + query_pos  # q - текущий кадр
        k = ref_query if ref_query_pos is None else ref_query + ref_query_pos  # k - соседние кадры
        v = ref_query  # v несет сами признаки соседних кадров

        attended = self.cross_attn(q, k, v, need_weights=False)[0]  # текущие query получают информацию из reference query
        query = self.norm1(query + self.dropout1(attended))  # residual сохраняет исходный текущий кадр

        ffn = self.linear2(self.dropout(self.activation(self.linear1(query))))  # ffn дообрабатывает смешанные признаки
        query = self.norm2(query + self.dropout2(ffn))  # второй residual как в transformer блоке
        return query


class TemporalQueryFusion(nn.Module):
    """Apply temporal query fusion to RF-DETR decoder hidden states.

    RF-DETR returns hidden states as ``[L, B*(T+1), N, C]`` when a clip is
    flattened into the batch dimension.  The first frame in every clip is the
    key/current frame, matching TransVOD++ datasets where the target annotation
    belongs to the first frame.  Reference frames provide context only.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        queries_per_group: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("TemporalQueryFusion requires num_layers >= 1.")
        self.queries_per_group = queries_per_group  # нужно для group-detr: каждая группа query обрабатывается отдельно
        self.layers = nn.ModuleList(
            [
                TemporalQueryFusionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hs: Tensor, batch_size: int, num_frames: int, key_frame_index: int = 0) -> Tensor:
        if hs.dim() != 4:
            raise ValueError(f"Expected hs [L, B*(T+1), N, C], got {tuple(hs.shape)}.")
        if num_frames < 2:
            return hs  # нет соседних кадров, значит нечего fusion-ить
        if hs.shape[1] != batch_size * num_frames:
            raise ValueError(
                "TemporalQueryFusion got inconsistent batch/frame dimensions: "
                f"hs batch={hs.shape[1]}, batch_size={batch_size}, num_frames={num_frames}."
            )
        if key_frame_index < 0 or key_frame_index >= num_frames:
            raise ValueError(f"key_frame_index must be in [0, {num_frames}), got {key_frame_index}.")

        num_decoder_layers, _, num_queries, channels = hs.shape  # l - decoder layers, n - queries
        hs_by_frame = hs.view(num_decoder_layers, batch_size, num_frames, num_queries, channels)  # восстанавливаем ось кадров

        # RF-DETR trains with Group DETR: the decoder query dimension is
        # num_queries * group_detr (for example 300 * 13).  Attending all groups
        # to all reference groups would create a huge dense matrix and does not
        # match the independent-group training design.  Fuse each query group
        # independently, then restore the original query layout for the heads.
        queries_per_group = self.queries_per_group or num_queries  # если групп нет, вся ось query одна группа
        if num_queries % queries_per_group != 0:
            queries_per_group = num_queries  # fallback, если размерность не делится на группы
        num_query_groups = num_queries // queries_per_group  # сколько group-detr групп получилось

        fused_layers = []  # здесь будут query key кадра после fusion для каждого decoder layer
        for layer_idx in range(num_decoder_layers):
            current = hs_by_frame[layer_idx, :, key_frame_index]  # query текущего кадра
            refs = torch.cat(
                [
                    hs_by_frame[layer_idx, :, :key_frame_index],
                    hs_by_frame[layer_idx, :, key_frame_index + 1 :],
                ],
                dim=1,
            )  # query reference кадров

            if num_query_groups > 1:
                current = current.view(
                    batch_size,
                    num_query_groups,
                    queries_per_group,
                    channels,
                ).flatten(0, 1)  # каждая group-detr группа становится отдельным batch элементом
                refs = (
                    refs.view(
                        batch_size,
                        num_frames - 1,
                        num_query_groups,
                        queries_per_group,
                        channels,
                    )
                    .permute(0, 2, 1, 3, 4)  # ставим group рядом с batch
                    .reshape(
                        batch_size * num_query_groups,
                        num_frames - 1,
                        queries_per_group,
                        channels,
                    )  # reference query той же группы попадут в attention вместе
                )

            for fusion_layer in self.layers:
                current = fusion_layer(current, refs)  # применяем один или несколько temporal attention слоев

            if num_query_groups > 1:
                current = current.view(
                    batch_size,
                    num_query_groups,
                    queries_per_group,
                    channels,
                ).reshape(batch_size, num_queries, channels)  # возвращаем исходную ось query
            fused_layers.append(current)  # сохраняем key-frame query этого decoder layer

        return torch.stack(fused_layers, dim=0)  # итог снова [l, b, n, c], как ожидают головы rf-detr


class TemporalDetectionDecoder(nn.Module):
    """TransVOD-style TDTD stage over key-frame memory.

    The module reuses RF-DETR's decoder layer type, but runs it only after
    temporal query fusion. It receives enriched key-frame queries, key-frame
    reference points, and encoder memory from the key frame. Reference frames
    have already contributed through TQE, so this stage keeps inference output
    semantics unchanged: one prediction set for the key frame.
    """

    def __init__(
        self,
        base_decoder: TransformerDecoder,
        num_layers: int,
        num_memory_feature_levels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("TemporalDetectionDecoder requires num_layers >= 1.")
        decoder_layer = copy.deepcopy(base_decoder.layers[-1])
        if num_memory_feature_levels is not None:
            decoder_layer.cross_attn = MSDeformAttn(
                base_decoder.d_model,
                n_levels=num_memory_feature_levels,
                n_heads=decoder_layer.cross_attn.n_heads,
                n_points=decoder_layer.cross_attn.n_points,
            )
        decoder_norm = copy.deepcopy(base_decoder.norm)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers,
            decoder_norm,
            return_intermediate=True,
            d_model=base_decoder.d_model,
            lite_refpoint_refine=base_decoder.lite_refpoint_refine,
            bbox_reparam=base_decoder.bbox_reparam,
        )
        # Keep the stage lightweight: RF-DETR's shared bbox head is applied
        # after TDTD, while reference points are carried through this decoder.
        self.decoder.bbox_embed = None

    def forward(
        self,
        query: Tensor,
        refpoints: Tensor,
        memory: Tensor,
        *,
        memory_key_padding_mask: Optional[Tensor],
        pos: Optional[Tensor],
        level_start_index: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Optional[Tensor],
        spatial_shapes_hw: list[tuple[int, int]] | None,
    ) -> tuple[Tensor, Tensor]:
        if valid_ratios is None:
            valid_ratios = memory.new_ones((memory.shape[0], spatial_shapes.shape[0], 2))
        hs, references = self.decoder(
            query,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            refpoints_unsigmoid=refpoints,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            spatial_shapes_hw=spatial_shapes_hw,
        )
        if references.shape[0] == 1 and hs.shape[0] > 1:
            references = references.expand(hs.shape[0], -1, -1, -1).contiguous()
        return hs, references


class TemporalDeformableEncoderLayer(nn.Module):
    """TransVOD-style TDTE layer adapted to RF-DETR multi-level memory.

    Operation order mirrors TransVOD's ``TemporalDeformableTransformerEncoderLayer``:
    key-frame memory self-attention, deformable cross-attention into reference
    frame memory, then FFN.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        num_ref_feature_levels: int,
        nhead: int,
        n_points: int,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformAttn(
            d_model,
            n_levels=num_ref_feature_levels,
            n_heads=nhead,
            n_points=n_points,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, memory: Tensor) -> Tensor:
        memory2 = self.linear2(self.dropout3(self.activation(self.linear1(memory))))
        memory = memory + self.dropout4(memory2)
        return self.norm3(memory)

    def forward(
        self,
        key_memory: Tensor,
        key_pos: Optional[Tensor],
        reference_points: Tensor,
        ref_memory: Tensor,
        ref_spatial_shapes: Tensor,
        ref_level_start_index: Tensor,
        ref_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = k = self.with_pos_embed(key_memory, key_pos)
        memory2 = self.self_attn(q, k, key_memory, need_weights=False)[0]
        key_memory = self.norm1(key_memory + self.dropout1(memory2))

        memory2 = self.cross_attn(
            self.with_pos_embed(key_memory, key_pos),
            reference_points,
            ref_memory,
            ref_spatial_shapes,
            ref_level_start_index,
            ref_padding_mask,
        )
        key_memory = self.norm2(key_memory + self.dropout2(memory2))
        return self.forward_ffn(key_memory)


class TemporalDeformableEncoder(nn.Module):
    """Apply TDTE layers to key-frame encoder memory using reference memories."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        num_feature_levels: int,
        num_ref_frames: int,
        nhead: int,
        n_points: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("TemporalDeformableEncoder requires num_layers >= 1.")
        self.layers = nn.ModuleList(
            [
                TemporalDeformableEncoderLayer(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    num_ref_feature_levels=num_feature_levels * num_ref_frames,
                    nhead=nhead,
                    n_points=n_points,
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def get_reference_points(
        spatial_shapes: Tensor,
        source_valid_ratios: Tensor,
        target_valid_ratios: Tensor,
        device: torch.device,
    ) -> Tensor:
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (source_valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (source_valid_ratios[:, None, level, 0] * width)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, dim=1)
        return reference_points[:, :, None] * target_valid_ratios[:, None]

    def forward(
        self,
        key_memory: Tensor,
        key_pos: Optional[Tensor],
        key_spatial_shapes: Tensor,
        key_valid_ratios: Tensor,
        ref_memory: Tensor,
        ref_pos: Optional[Tensor],
        ref_spatial_shapes: Tensor,
        ref_level_start_index: Tensor,
        ref_valid_ratios: Tensor,
        ref_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = key_memory
        ref_memory_with_pos = ref_memory if ref_pos is None else ref_memory + ref_pos
        reference_points = self.get_reference_points(
            key_spatial_shapes,
            key_valid_ratios,
            ref_valid_ratios,
            output.device,
        )
        for layer in self.layers:
            output = layer(
                output,
                key_pos,
                reference_points,
                ref_memory_with_pos,
                ref_spatial_shapes,
                ref_level_start_index,
                ref_padding_mask,
            )
        return output


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
