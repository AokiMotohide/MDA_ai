import math
import torch
import random
from depth_anything_3.utils.geometry import affine_inverse

def prepare_inputs(views, device, re_concat=True, new_gs_mask=False, rand_shuffle=False):
    views_new = {}
    if re_concat:
        
        if rand_shuffle:
            if random.random() < 0.25:
                views_1 = views[0]
                views_extra = views[1:]
                random.shuffle(views_extra)
                views = [views_1] + views_extra
            
        for k, v in views[0].items():
            all_vs = [view[k] for view in views]
            if isinstance(v, torch.Tensor):
                views_new[k] = torch.stack(all_vs, dim=1)
                views_new[k] = views_new[k].to(device, non_blocking=True)
            elif isinstance(v, list):
                views_new[k] = list(map(list, zip(*all_vs)))
            else:
                views_new[k] = all_vs
    else:
        for k, v in views.items():
            if isinstance(v, torch.Tensor):
                views_new[k] = v.to(device, non_blocking=True)
            else:
                # print(k)
                views_new[k] = v
            
    views_new["camera_extrinsics"] = affine_inverse(views_new["camera_pose"])
    views_new["camera_extrinsics"] = views_new["camera_extrinsics"].to(device, non_blocking=True)
    if 'camera_pose_Vreadout' in views_new:
        views_new["camera_extrinsics_Vreadout"] = affine_inverse(views_new["camera_pose_Vreadout"])
        views_new["camera_extrinsics_Vreadout"] = views_new["camera_extrinsics_Vreadout"].to(device, non_blocking=True)
 
    return views_new

