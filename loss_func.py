from loss import get_bce, get_bits, get_metrics, get_mse, rgb2yuv
import torch
from scipy.spatial import cKDTree
import MinkowskiEngine as ME
import numpy as np

def geo_loss (input, out_set):
    bce, bce1, bce2, bce_hat, bce_list = 0, 0, 0, 0, []
    # bce
    for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
        out_cls = ME.SparseTensor(features=out_cls.F[:, :1],
                                coordinate_map_key=out_cls.coordinate_map_key,
                                coordinate_manager=out_cls.coordinate_manager,
                                device=out_cls.device)
        ground_truth = ME.SparseTensor(features=ground_truth.F[:, :1],
                                coordinate_map_key=ground_truth.coordinate_map_key,
                                coordinate_manager=ground_truth.coordinate_manager,
                                device=ground_truth.device)
        curr_bce = get_bce(out_cls, ground_truth)
        bce += curr_bce
    # bce1
    for out_cls, ground_truth in zip(out_set['out_cls_list3'], out_set['ground_truth_list3']):
        out_cls = ME.SparseTensor(features=out_cls.F[:, :1],
                                coordinate_map_key=out_cls.coordinate_map_key,
                                coordinate_manager=out_cls.coordinate_manager,
                                device=out_cls.device)
        ground_truth = ME.SparseTensor(features=ground_truth.F[:, :1],
                                       coordinate_map_key=ground_truth.coordinate_map_key,
                                       coordinate_manager=ground_truth.coordinate_manager,
                                       device=ground_truth.device)
        curr_bce = get_bce(out_cls, ground_truth)
        bce1 += curr_bce
    # bce2
    for out_cls, ground_truth in zip(out_set['out_cls_list2'], out_set['ground_truth_list2']):
        out_cls = ME.SparseTensor(features=out_cls.F[:, :1],
                                coordinate_map_key=out_cls.coordinate_map_key,
                                coordinate_manager=out_cls.coordinate_manager,
                                device=out_cls.device)
        ground_truth = ME.SparseTensor(features=ground_truth.F[:, :1],
                                       coordinate_map_key=ground_truth.coordinate_map_key,
                                       coordinate_manager=ground_truth.coordinate_manager,
                                       device=ground_truth.device)
        curr_bce = get_bce(out_cls, ground_truth)
        bce2 += curr_bce
    # bce_hat
    bce_hat = get_bce(out_set['out'], input)

    bce_list = [bce, bce1, bce2, bce_hat]

    bce_list1 = torch.stack([bce, bce1, bce2, bce_hat], dim=0).tolist()

    return bce_list, bce_list1

def yuv_mse(ori_pts, dist_pts):
    ori_geo = ori_pts.C[:, 1:].cpu().numpy()
    ori_col = ori_pts.F
    dist_geo = dist_pts.C[:, 1:].cpu().numpy()
    dist_col = dist_pts.F

    bwd_tree = cKDTree(ori_geo, balanced_tree=False)
    _, bwd_idx = bwd_tree.query(dist_geo)
    bwd_colors = ori_col[bwd_idx]

    # dist_colors, ori_colors = rgb2yuv(bwd_colors.cpu(), ori_col.cpu())
    dist_colors = dist_col.cpu()
    ori_colors = bwd_colors.cpu()
    y_mse = torch.nn.functional.mse_loss(ori_colors[0], dist_colors[0])
    u_mse = torch.nn.functional.mse_loss(ori_colors[1], dist_colors[1])
    v_mse = torch.nn.functional.mse_loss(ori_colors[2], dist_colors[2])
    mse = torch.stack([y_mse, u_mse, v_mse], dim=0)
    return mse

def attri_loss (input, out_set):
    mse, mse1, mse2, mse_hat, mse_list = 0, 0, 0, 0, []

    out = out_set['out1']
    GT = out_set['ground_truth_list'][2]
    # out = ME.SparseTensor(features=out.F[:, 1:] * 255,
    out = ME.SparseTensor(features=out.F[:, 1:],
                        coordinate_map_key=out.coordinate_map_key,
                        coordinate_manager=out.coordinate_manager,
                        device=out.device)
    # GT = ME.SparseTensor(features=GT.F[:, 1:] * 255,
    GT = ME.SparseTensor(features=GT.F[:, 1:],
                        coordinate_map_key=GT.coordinate_map_key,
                        coordinate_manager=GT.coordinate_manager,
                        device=GT.device)
    mse = sum(yuv_mse(out, GT))

    out1 = out_set['out3']
    GT1 = out_set['ground_truth_list'][2]
    # out1 = ME.SparseTensor(features=out1.F[:, 1:] * 255,
    out1 = ME.SparseTensor(features=out1.F[:, 1:],
                        coordinate_map_key=out1.coordinate_map_key,
                        coordinate_manager=out1.coordinate_manager,
                        device=out1.device)
    # GT1 = ME.SparseTensor(features=GT1.F[:, 1:] * 255,
    GT1 = ME.SparseTensor(features=GT1.F[:, 1:],
                        coordinate_map_key=GT1.coordinate_map_key,
                        coordinate_manager=GT1.coordinate_manager,
                        device=GT1.device)
    mse1 = sum(yuv_mse(out1, GT1))

    out2 = out_set['out2']
    GT2 = out_set['ground_truth_list2'][2]
    # out2 = ME.SparseTensor(features=out2.F[:, 1:] * 255,
    out2 = ME.SparseTensor(features=out2.F[:, 1:],
                        coordinate_map_key=out2.coordinate_map_key,
                        coordinate_manager=out2.coordinate_manager,
                        device=out2.device)
    # GT2 = ME.SparseTensor(features=GT2.F[:, 1:] * 255,
    GT2 = ME.SparseTensor(features=GT2.F[:, 1:],
                        coordinate_map_key=GT2.coordinate_map_key,
                        coordinate_manager=GT2.coordinate_manager,
                        device=GT2.device)
    mse2 = sum(yuv_mse(out2, GT2))

    out3 = out_set['out']
    GT3 = input
    # out3 = ME.SparseTensor(features=out3.F[:, 1:] * 255,
    out3 = ME.SparseTensor(features=out3.F[:, 1:],
                        coordinate_map_key=out3.coordinate_map_key,
                        coordinate_manager=out3.coordinate_manager,
                        device=out3.device)
    # GT3 = ME.SparseTensor(features=GT3.F[:, 1:] * 255,
    GT3 = ME.SparseTensor(features=GT3.F[:, 1:],
                        coordinate_map_key=GT3.coordinate_map_key,
                        coordinate_manager=GT3.coordinate_manager,
                        device=GT3.device)
    mse_hat = sum(yuv_mse(out3, GT3))

    mse_list = [mse, mse1, mse2, mse_hat]

    mse_list1 = torch.stack([mse, mse1, mse2, mse_hat], dim=0).tolist()

    return mse_list, mse_list1