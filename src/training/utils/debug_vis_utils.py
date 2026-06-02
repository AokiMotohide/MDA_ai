import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def debug_vis_output_utils_onlydepth(output, batch, batch_idx, saved_path, complete=False):
    img_std = torch.tensor([0.229, 0.224, 0.225], device=batch['img'].device)
    img_mean = torch.tensor([0.485, 0.456, 0.406], device=batch['img'].device)
    nowimgs = batch['img'].permute(0, 1, 3, 4, 2) * img_std + img_mean

    os.makedirs(saved_path, exist_ok=True)

    gt_depth = batch['depthmap']
    valid_mask = batch['valid_mask']

    if 'depth_inference' in output:
        pred_depth = output['depth_inference'].detach()
    elif isinstance(output['depth'], list):
        pred_depth = torch.stack(output['depth'], dim=-1).detach()
        mog_weights = torch.exp(output['mog_weight'].detach())
        pred_depth = (pred_depth * mog_weights).sum(dim=-1)
    else:
        pred_depth = output['depth'].detach()

    B = nowimgs.shape[0]
    for b in range(B):
        seq_len = nowimgs[b].shape[0]

        interval = 1 if complete else 8
        for i in range(0, seq_len, interval):
            nowimg_i = nowimgs[b][i].cpu().numpy()
            nowimg_i = (nowimg_i * 255).astype(np.uint8)

            pred_depth_i = pred_depth[b][i].cpu().numpy()

            gt_depth_i = gt_depth[b][i].cpu().numpy()
            valid_mask_i = valid_mask[b][i].cpu().numpy()
            valid_mask_i_img = (valid_mask[b][i].float() * 255).unsqueeze(-1).expand(-1, -1, 3).cpu().numpy()
            valid_mask_i_img = valid_mask_i_img.astype(np.uint8)

            colors = depth_vis_walign(gt_depth_i, pred_depth_i.copy(), pred_depth_i, valid_mask_i)
            colors = [nowimg_i, colors[0], colors[1], valid_mask_i_img, colors[3]]

            nowimg = np.concatenate(colors, axis=1)
            cv2.imwrite(f'{saved_path}/debug_vis_{batch_idx}_{b}_{i}_output.png',
                        cv2.cvtColor(nowimg, cv2.COLOR_RGB2BGR))

    return


def debug_vis_output_utils_separate_depth(output, batch, batch_idx, saved_path=None, complete=False, return_images=False, max_images=None, global_rank=0):
    img_std = torch.tensor([0.229, 0.224, 0.225], device=batch['img'].device)
    img_mean = torch.tensor([0.485, 0.456, 0.406], device=batch['img'].device)
    nowimgs = batch['img'].permute(0, 1, 3, 4, 2) * img_std + img_mean

    if saved_path is not None:
        os.makedirs(saved_path, exist_ok=True)

    vis_images = []

    if 'depthmap' in batch:
        gt_depth = batch['depthmap']
        valid_mask = batch['valid_mask']
    else:
        gt_depth = output['depth'][0].detach()
        valid_mask = torch.ones_like(gt_depth).detach()

    if 'depth_inference' in output:
        pred_depth = output['depth_inference'].detach()
    elif isinstance(output['depth'], list):
        pred_depth = torch.stack(output['depth'], dim=-1).detach()
        mog_weights = torch.exp(output['mog_weight'].detach())
        pred_depth = (pred_depth * mog_weights).sum(dim=-1)
    else:
        pred_depth = output['depth'].detach()

    pred_depth_list = output['depth']
    if not isinstance(pred_depth_list, list):
        pred_depth_list = [pred_depth_list]
    pred_depth_list = [pred_depth] + pred_depth_list
    if 'depth_pretrained' in output:
        pred_depth_pretrained = output['depth_pretrained'].detach()
        pred_depth_list = pred_depth_list + [pred_depth_pretrained]

    extra_depthmap = None
    if 'depthmap_sampling' in batch:
        extra_depthmap = batch['depthmap_sampling']
    elif 'extra_depthmap' in batch:
        extra_depthmap = batch['extra_depthmap']

    if extra_depthmap is not None:
        extra_depthmap_list = []
        gt_depth_list = []
        for i in range(len(extra_depthmap)):
            if gt_depth[i].max() - gt_depth[i].min() < 1e-6:
                extra_depthmap_list.append(pred_depth_list[0][i])
                gt_depth_list.append(pred_depth_list[-1][i])
            else:
                extra_depthmap_list.append(extra_depthmap[i])
                gt_depth_list.append(gt_depth[i])
        extra_depthmap = torch.stack(extra_depthmap_list, dim=0)
        pred_depth_list = [extra_depthmap] + pred_depth_list
        gt_depth = torch.stack(gt_depth_list, dim=0)

    depth_diff = pred_depth - output['depth'][0]
    depth_diff = torch.abs(depth_diff) * valid_mask
    depth_diff = depth_diff.sum() / (valid_mask.sum() + 1e-8)

    N = len(output['depth'])
    if 'indices' in output:
        best_indices = output['indices']

    H, W = nowimgs.shape[2:4]
    B, S = nowimgs.shape[0:2]
    for b in range(min(B, 4)):
        seq_len = nowimgs[b].shape[0]

        interval = 1 if complete else 8
        for i in range(0, seq_len, interval):
            nowimg_i = nowimgs[b][i].cpu().numpy()
            nowimg_i = (nowimg_i * 255).astype(np.uint8)

            gt_depth_i = gt_depth[b][i].cpu().numpy()
            valid_mask_i = valid_mask[b][i].cpu().numpy()
            valid_mask_i_img = (valid_mask[b][i].float() * 255).unsqueeze(-1).expand(-1, -1, 3).cpu().numpy()
            valid_mask_i_img = valid_mask_i_img.astype(np.uint8)

            pred_depth_list_i = [
                pred_depth_list[j][b][i].cpu().numpy() for j in range(len(pred_depth_list))
            ]

            gt_color, color_list, diff_list = depth_vis_walign2(gt_depth_i, pred_depth_list_i, valid_mask_i, ('extra_depthmap' in batch))
            pred_depth_color = color_list[1] if ('extra_depthmap' in batch) else color_list[0]

            if 'alpha_img' in batch:
                diff_list = diff_list[1:]

                alpha_img_i = batch['alpha_img'][b][i].unsqueeze(-1).expand(-1, -1, 3).cpu().numpy()
                alpha_img_i = (alpha_img_i * 255).astype(np.uint8)
                diff_list = [alpha_img_i] + diff_list

            zeros = np.zeros_like(gt_color)
            colors_line_1 = [nowimg_i, gt_color] + color_list

            if 'mean_std_weight' in output:
                mean_std_weight_i = output['mean_std_weight'][b][i].reshape(H, W, 1).expand(-1, -1, 3).cpu().numpy()
                mean_std_weight_i = (mean_std_weight_i * 255).astype(np.uint8)
                colors_line_2 = [nowimg_i, mean_std_weight_i] + diff_list
            else:
                colors_line_2 = [nowimg_i, zeros] + diff_list

            colors_line_1 = np.concatenate(colors_line_1, axis=1)
            colors_line_2 = np.concatenate(colors_line_2, axis=1)
            nowimg = np.concatenate([colors_line_1, colors_line_2], axis=0)

            if 'indices' in output:
                tmp_color = []
                for j in range(N):
                    best_indices_i = (best_indices[b][i] == j).float().unsqueeze(-1).cpu().numpy()
                    colored1 = (best_indices_i * 255).astype(np.uint8)
                    tmp_color.append(colored1)
                tmp_color = np.concatenate(tmp_color, axis=1)

                nowimg_i_cat = np.concatenate([nowimg_i] * N, axis=1)
                tmp_color = (tmp_color * 0.5 + nowimg_i_cat * 0.5).astype(np.uint8)

                if 'tangent_weight' in output:
                    tangent_weight_i = output['tangent_weight'][b][i].unsqueeze(-1).expand(-1, -1, 3).cpu().numpy()
                    tangent_weight_i = (tangent_weight_i * 255).astype(np.uint8)
                else:
                    tangent_weight_i = zeros

                colors_line_3 = [nowimg_i, tangent_weight_i, zeros, tmp_color]
                if 'extra_depthmap' in batch:
                    colors_line_3 = [nowimg_i, tangent_weight_i, zeros, zeros, tmp_color]
                colors_line_3 = np.concatenate(colors_line_3, axis=1)
                if colors_line_3.shape[1] != nowimg.shape[1]:
                    colors_line_3 = cv2.resize(colors_line_3, (nowimg.shape[1], colors_line_3.shape[0]), interpolation=cv2.INTER_NEAREST)

                nowimg = np.concatenate([nowimg, colors_line_3], axis=0)

            if ('seg_logits' in output) and (output['seg_logits'] is not None):
                seg_pred = output['seg_logits'][b][i][0].unsqueeze(-1).expand(-1, -1, 3).long().cpu().numpy()
                if 'glass_mask' in batch:
                    seg_gt = (batch['glass_mask'][b][i] > 0.1).unsqueeze(-1).expand(-1, -1, 3).long().cpu().numpy()
                else:
                    seg_gt = np.zeros_like(seg_pred)

                seg_gt_vis = (seg_gt.astype(np.float32) * 255).astype(np.uint8)
                seg_pred_vis = (seg_pred.astype(np.float32) * 255).astype(np.uint8)

                colors_line_4 = [nowimg_i, seg_gt_vis, zeros, seg_pred_vis]
                num_extra = len(color_list) - 2
                for _ in range(num_extra):
                    colors_line_4.append(zeros)

                colors_line_4 = np.concatenate(colors_line_4, axis=1)
                if colors_line_4.shape[1] != nowimg.shape[1]:
                    colors_line_4 = cv2.resize(colors_line_4, (nowimg.shape[1], colors_line_4.shape[0]), interpolation=cv2.INTER_NEAREST)

                nowimg = np.concatenate([nowimg, colors_line_4], axis=0)

            if ('mog_weight_raw' in output) and (output['mog_weight_raw'] is not None):
                mog_weight_list = output['mog_weight_raw'].detach()
                mog_weight_i = mog_weight_list[b][i]

                mog_weight_i = mog_weight_i.clamp(0, 1).cpu().numpy()
                mog_weight_heatmaps = []
                for j in range(mog_weight_i.shape[-1]):
                    weight_map = mog_weight_i[..., j]
                    weight_color = matplotlib.colormaps['turbo'](weight_map)[..., :3]
                    weight_color = (weight_color * 255).astype(np.uint8)
                    mog_weight_heatmaps.append(weight_color)

                if 'glass_mask' in batch:
                    seg_gt = (batch['glass_mask'][b][i] > 0.1).unsqueeze(-1).expand(-1, -1, 3).long().cpu().numpy()
                else:
                    seg_gt = np.zeros_like(weight_color)
                seg_gt_vis = (seg_gt.astype(np.float32) * 255).astype(np.uint8)

                if len(mog_weight_heatmaps) > 0:
                    colors_line_mog = [nowimg_i, seg_gt_vis, zeros, zeros] + mog_weight_heatmaps
                    colors_line_mog = np.concatenate(colors_line_mog, axis=1)
                    if colors_line_mog.shape[1] != nowimg.shape[1]:
                        colors_line_mog = cv2.resize(colors_line_mog, (nowimg.shape[1], colors_line_mog.shape[0]), interpolation=cv2.INTER_NEAREST)
                    nowimg = np.concatenate([nowimg, colors_line_mog], axis=0)

            if 'mog_weight_full' in output:
                if 'sky_mask' in batch:
                    sky_mask = batch['sky_mask'][b][i].unsqueeze(-1).expand(-1, -1, 3).cpu().numpy()
                    sky_mask_vis = (sky_mask * 255).astype(np.uint8)
                else:
                    sky_mask_vis = zeros
                sky_mask_pred = torch.exp(output['mog_weight_full'][b][i][..., -1]).detach().cpu().numpy()
                sky_mask_pred = matplotlib.colormaps['turbo'](sky_mask_pred)[..., :3]
                sky_mask_pred = (sky_mask_pred * 255).astype(np.uint8)

                sky_mask_binary = output['sky_mask'][b][i].unsqueeze(-1).detach().cpu().numpy()
                colors_line_4 = [nowimg_i, sky_mask_vis, zeros, sky_mask_pred, pred_depth_color * (1 - sky_mask_binary)] + [zeros] * (len(color_list) - 3)
                colors_line_4 = np.concatenate(colors_line_4, axis=1)
                if colors_line_4.shape[1] != nowimg.shape[1]:
                    colors_line_4 = cv2.resize(colors_line_4, (nowimg.shape[1], colors_line_4.shape[0]), interpolation=cv2.INTER_NEAREST)
                nowimg = np.concatenate([nowimg, colors_line_4], axis=0)

            if 'per_pixel_conf_loss' in output:
                conf_loss_seq = output['per_pixel_conf_loss'][b].detach().cpu().numpy()
                valid_mask_seq = valid_mask[b].cpu().numpy()
                seq_valid_pixels = conf_loss_seq[valid_mask_seq > 0]
                if seq_valid_pixels.size > 0:
                    seq_top_thresh = float(np.quantile(seq_valid_pixels, 0.98))
                else:
                    seq_top_thresh = float(conf_loss_seq.max())

                conf_loss_i = conf_loss_seq[i]
                valid_pixels = conf_loss_i[valid_mask_i > 0]
                if valid_pixels.size > 0:
                    vmin = float(np.quantile(valid_pixels, 0.05))
                    vmax = float(np.quantile(valid_pixels, 0.95))
                else:
                    vmin, vmax = float(conf_loss_i.min()), float(conf_loss_i.max())
                conf_loss_norm = np.clip((conf_loss_i - vmin) / (vmax - vmin + 1e-8), 0, 1)
                conf_loss_color = matplotlib.colormaps['turbo'](conf_loss_norm)[..., :3]
                conf_loss_color = (conf_loss_color * 255).astype(np.uint8)

                top_mask = conf_loss_i > seq_top_thresh
                conf_loss_color[top_mask] = 0

                colors_line_conf = [nowimg_i, conf_loss_color] + [zeros] * len(color_list)
                colors_line_conf = np.concatenate(colors_line_conf, axis=1)
                if colors_line_conf.shape[1] != nowimg.shape[1]:
                    colors_line_conf = cv2.resize(colors_line_conf, (nowimg.shape[1], colors_line_conf.shape[0]), interpolation=cv2.INTER_NEAREST)
                nowimg = np.concatenate([nowimg, colors_line_conf], axis=0)

            try:
                dataset_name = batch['dataset'][b][i]
            except Exception:
                dataset_name = ''
            if return_images:
                vis_images.append((f'debug_vis_{batch_idx}_{b}_{i}_{dataset_name}_output_{global_rank}.png', nowimg.copy()))
            if saved_path is not None:
                cv2.imwrite(f'{saved_path}/debug_vis_{batch_idx}_{b}_{i}_{dataset_name}_output_{global_rank}.png',
                            cv2.cvtColor(nowimg, cv2.COLOR_RGB2BGR))
                print(f'saved debug_vis_{batch_idx}_{b}_{i}_{dataset_name}_output_{global_rank}.png')

            if max_images is not None and return_images and len(vis_images) >= max_images:
                return vis_images[:max_images]

    if return_images:
        return vis_images if max_images is None else vis_images[:max_images]

    return


def depth_vis_walign(depthmap1, depthmap2, depthmap3, mask, cmap='Spectral'):
    mask1 = np.where(depthmap1 > 1e-5, True, False)
    mask2 = np.where(depthmap2 > 1e-5, True, False)
    mask3 = np.where(depthmap3 > 1e-5, True, False)

    mask = (mask > 0)

    mask0 = (mask1 & mask2 & mask3 & mask)
    depthmap1_nan = np.where(mask0, depthmap1, np.nan)
    disp1_nan = 1 / (depthmap1_nan + 1e-6)

    min_disp1, max_disp1 = np.nanquantile(disp1_nan, 0.001), np.nanquantile(disp1_nan, 0.99)

    disp1 = 1 / (depthmap1_nan + 1e-6)
    disp2 = 1 / (depthmap2 + 1e-6)
    disp3 = 1 / (depthmap3 + 1e-6)

    disp1 = (disp1 - min_disp1) / (max_disp1 - min_disp1)
    disp2 = (disp2 - min_disp1) / (max_disp1 - min_disp1)
    disp3 = (disp3 - min_disp1) / (max_disp1 - min_disp1)

    diff1 = np.nan_to_num(np.abs(disp1 - disp2))
    diff2 = np.nan_to_num(np.abs(disp1 - disp3))
    diff1 = diff1 * mask
    diff2 = diff2 * mask

    diff1_max = diff1.max()
    diff1 = diff1 / diff1_max
    diff2 = diff2 / diff1_max

    colored1 = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp1)[..., :3], 0)
    colored2 = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp2)[..., :3], 0)
    colored3 = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp3)[..., :3], 0)

    diff1 = np.nan_to_num(matplotlib.colormaps[cmap](diff1)[..., :3], 0)
    diff2 = np.nan_to_num(matplotlib.colormaps[cmap](diff2)[..., :3], 0)

    colored1 = np.ascontiguousarray((colored1.clip(0, 1) * 255).astype(np.uint8))
    colored2 = np.ascontiguousarray((colored2.clip(0, 1) * 255).astype(np.uint8))
    colored3 = np.ascontiguousarray((colored3.clip(0, 1) * 255).astype(np.uint8))

    diff1 = np.ascontiguousarray((diff1.clip(0, 1) * 255).astype(np.uint8))
    diff2 = np.ascontiguousarray((diff2.clip(0, 1) * 255).astype(np.uint8))

    colored1 = cv2.cvtColor(colored1, cv2.COLOR_RGB2BGR)
    colored2 = cv2.cvtColor(colored2, cv2.COLOR_RGB2BGR)
    colored3 = cv2.cvtColor(colored3, cv2.COLOR_RGB2BGR)

    diff1 = cv2.cvtColor(diff1, cv2.COLOR_RGB2BGR)
    diff2 = cv2.cvtColor(diff2, cv2.COLOR_RGB2BGR)

    return colored1, colored2, colored3, diff1, diff2


def depth_vis_walign2(depthmap1, depth_list, mask, extra_depthmap=False, cmap='Spectral'):
    mask1 = np.where(depthmap1 > 1e-5, True, False)

    mask = (mask > 0)

    mask0 = (mask1 & mask)
    depthmap1_nan = np.where(mask0, depthmap1, np.nan)
    disp1_nan = 1 / (depthmap1_nan + 1e-6)

    min_disp1, max_disp1 = np.nanquantile(disp1_nan, 0.001), np.nanquantile(disp1_nan, 0.99)

    disp1 = 1 / (depthmap1_nan + 1e-6)
    disp1 = (disp1 - min_disp1) / (max_disp1 - min_disp1)
    colored1 = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp1)[..., :3], 0)
    colored1 = np.ascontiguousarray((colored1.clip(0, 1) * 255).astype(np.uint8))
    colored1 = cv2.cvtColor(colored1, cv2.COLOR_RGB2BGR)

    diff_i_max_list = []
    for i, depth in enumerate(depth_list):
        if extra_depthmap and i == 0:
            continue
        disp_i = 1 / (depth + 1e-6)
        disp_i = (disp_i - min_disp1) / (max_disp1 - min_disp1)

        diff_i = np.nan_to_num(np.abs(disp1 - disp_i))
        diff_i = diff_i * mask
        diff_i_max_list.append(diff_i.max())

    diff_max = max(diff_i_max_list)

    colored_list = []
    diff_list = []
    for i, depth in enumerate(depth_list):
        disp_i = 1 / (depth + 1e-6)
        disp_i = (disp_i - min_disp1) / (max_disp1 - min_disp1)

        diff_i = np.nan_to_num(np.abs(disp1 - disp_i))
        diff_i = diff_i * mask
        diff_i = diff_i / diff_max

        colored_i = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp_i)[..., :3], 0)
        diff_i = np.nan_to_num(matplotlib.colormaps[cmap](diff_i)[..., :3], 0)

        colored_i = np.ascontiguousarray((colored_i.clip(0, 1) * 255).astype(np.uint8))
        colored_i = cv2.cvtColor(colored_i, cv2.COLOR_RGB2BGR)
        diff_i = np.ascontiguousarray((diff_i.clip(0, 1) * 255).astype(np.uint8))
        diff_i = cv2.cvtColor(diff_i, cv2.COLOR_RGB2BGR)

        colored_list.append(colored_i)
        diff_list.append(diff_i)

    return colored1, colored_list, diff_list
