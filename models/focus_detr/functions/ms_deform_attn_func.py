# Copyright 2023 Huawei Technologies Co., Ltd
#
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
#
from __future__ import absolute_import, division, print_function

import copy
###########
import pdb

import mindspore.numpy as ms_np
import mindspore.ops as ops
import MultiScaleDeformableAttention as MSDA
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(
        ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
    ):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step
        )
        ctx.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


def ms_deform_attn(value, value_spatial_shapes, sampling_locations, attention_weights):
    ####
    # print(f"-555--value_spatial_shapes:{value_spatial_shapes}")
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    split_points = [H_ * W_ for H_, W_ in value_spatial_shapes]
    # print(f"-111---split_points:{split_points}")
    split_points = split_points[:-1]
    split_points_size = len(split_points)
    for i in range(1, split_points_size):
        split_points[i] = split_points[i - 1] + split_points[i]
    split_points = [int(one.asnumpy()) for one in split_points]
    # print(f"-222---split_points:{split_points}")
    ###
    # print(f"-666--value_spatial_shapes:{value_spatial_shapes}")
    # print(f"---value.shape:{value.shape}")
    value_list = ms_np.split(value, split_points, axis=1)
    ###
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    # new_value_spatial_shapes=copy.deepcopy(value_spatial_shapes)
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        # value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        n1, n2, n3, n4 = value_list[lid_].shape
        value_l_ = value_list[lid_].reshape(n1, n2, n3 * n4)
        value_l_ = ops.transpose(value_l_, (0, 2, 1))
        value_l_ = value_l_.reshape(N_ * M_, D_, int(H_.asnumpy()), int(W_.asnumpy()))
        sampling_grid_l_ = sampling_grids[:, :, :, lid_]
        sampling_grid_l_ = ops.transpose(sampling_grid_l_, (0, 2, 1, 3, 4))
        #############################################
        n1, n2, n3, n4, n5 = sampling_grid_l_.shape
        sampling_grid_l_ = sampling_grid_l_.reshape(n1 * n2, n3, n4, n5)
        sampling_value_l_ = ops.grid_sample(
            value_l_, sampling_grid_l_, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = ops.transpose(attention_weights, (0, 2, 1, 3, 4)).reshape(N_ * M_, 1, Lq_, L_ * P_)
    sampling_value_list = ms_np.stack(sampling_value_list, axis=-2)
    n1, n2, n3, n4, n5 = sampling_value_list.shape
    sampling_value_list = sampling_value_list.reshape(n1, n2, n3, n5 * n4)
    output = (sampling_value_list * attention_weights).sum(-1).view((N_, M_ * D_, Lq_))
    output = ops.transpose(output, (0, 2, 1))
    return output


def ms_deform_attn_dec(value, value_spatial_shapes, sampling_locations, attention_weights):
    ####
    # print(f"-555--value_spatial_shapes:{value_spatial_shapes}")
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    split_points = [H_ * W_ for H_, W_ in value_spatial_shapes]
    # print(f"-111---split_points:{split_points}")
    split_points = split_points[:-1]
    split_points_size = len(split_points)
    for i in range(1, split_points_size):
        split_points[i] = split_points[i - 1] + split_points[i]
    split_points = [int(one.asnumpy()) for one in split_points]
    # print(f"-222---split_points:{split_points}")
    ###
    # print(f"-666--value_spatial_shapes:{value_spatial_shapes}")
    # print(f"---value.shape:{value.shape}")
    value_list = ms_np.split(value, split_points, axis=1)
    ###
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        # value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        n1, n2, n3, n4 = value_list[lid_].shape
        value_l_ = value_list[lid_].reshape(n1, n2, n3 * n4)
        value_l_ = ops.transpose(value_l_, (0, 2, 1))
        value_l_ = value_l_.reshape(N_ * M_, D_, int(H_.asnumpy()), int(W_.asnumpy()))
        sampling_grid_l_ = sampling_grids[:, :, :, lid_]
        sampling_grid_l_ = ops.transpose(sampling_grid_l_, (0, 2, 1, 3, 4))
        #############################################
        n1, n2, n3, n4, n5 = sampling_grid_l_.shape
        sampling_grid_l_ = sampling_grid_l_.reshape(n1 * n2, n3, n4, n5)
        # temp_value_l_=copy.deepcopy(value_l_)
        # temp_sampling_grid_l_=copy.deepcopy(sampling_grid_l_)
        # sampling_value_l_ =  ops.grid_sample(temp_value_l_, temp_sampling_grid_l_, interpolation_mode='bilinear',padding_mode='zeros', align_corners=False)
        sampling_value_l_ = ops.grid_sample(
            value_l_, sampling_grid_l_, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False
        )
        # temp=copy.deepcopy(sampling_value_l_)
        sampling_value_list.append(sampling_value_l_)  #
    attention_weights = ops.transpose(attention_weights, (0, 2, 1, 3, 4)).reshape(N_ * M_, 1, Lq_, L_ * P_)
    # print(f"-777--value_spatial_shapes:{value_spatial_shapes}")
    ###################
    # N_*M_, D_, Lq_, P_
    # print(f"----len sampling_value_list : {len(sampling_value_list)}")
    sampling_value_list = ms_np.stack(sampling_value_list, axis=-2)
    # return sampling_value_list
    n1, n2, n3, n4, n5 = sampling_value_list.shape
    sampling_value_list = sampling_value_list.reshape(n1, n2, n3, n5 * n4)
    # print(f"---sampling_value_list-----output:{sampling_value_list}")
    # print(f"---attention_weights-----output:{attention_weights}")
    output = (sampling_value_list * attention_weights).sum(-1).view((N_, M_ * D_, Lq_))
    # return sampling_value_list
    output = ops.transpose(output, (0, 2, 1))
    # output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    # print(f"---ms_deform_attn-----output:{output}")
    print(f"-999--value_spatial_shapes:{value_spatial_shapes}")  #
    return output
