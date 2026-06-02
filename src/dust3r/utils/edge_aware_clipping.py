import cv2
import numpy as np




# def two_step_downsample(depth, target_h, target_w, edge_threshold=0.2, erode_size=3):
#     # Step 1: nearest-neighbor resize (handles float factor natively)
#     nearest = cv2.resize(depth, (target_w, target_h),
#                          interpolation=cv2.INTER_NEAREST)

#     # Step 2: min-pool at full res, then resize with nearest
#     # Use INTER_AREA to approximate "min over region"
#     # But INTER_AREA averages, so instead: erode then resize
#     kernel = np.ones((erode_size, erode_size), np.uint8)
#     eroded = cv2.erode(depth, kernel)  # local min
#     min_resized = cv2.resize(eroded, (target_w, target_h),
#                              interpolation=cv2.INTER_NEAREST)

#     # Step 3: hybrid blend
#     # diff = cv2.resize(
#     #     cv2.absdiff(
#     #         cv2.dilate(depth, kernel),
#     #         cv2.erode(depth, kernel)
#     #     ), (target_w, target_h), interpolation=cv2.INTER_NEAREST
#     # )
#     diff = np.abs(min_resized.astype(np.float32) - nearest.astype(np.float32))
#     has_edge = (diff > (np.median(depth[depth > 0]) * edge_threshold)) & (nearest > 0) & (min_resized > 0)
#     return has_edge, np.where(has_edge, min_resized, nearest)


def two_step_downsample(depth, target_h, target_w, edge_threshold=0.1, erode_size=3):
    # Step 1: nearest-neighbor resize (handles float factor natively)
    nearest = cv2.resize(depth, (target_w, target_h),
                         interpolation=cv2.INTER_NEAREST)

    # Step 2: min-pool at full res, then resize with nearest
    safe_depth = depth.copy().astype(np.float32)
    depth_max = depth.max()
    safe_depth[safe_depth == 0] = depth.max() + 1
    
    interpolated_depth = cv2.resize(safe_depth, (target_w, target_h),
                                    interpolation=cv2.INTER_LINEAR)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    eroded = cv2.erode(safe_depth, kernel)  # local min
    min_resized = cv2.resize(eroded, (target_w, target_h),
                             interpolation=cv2.INTER_NEAREST)

    # Step 3: hybrid blend
    diff = np.abs(interpolated_depth.astype(np.float32) - nearest.astype(np.float32))
    has_edge = (diff > (np.median(depth[depth > 0]) * edge_threshold)) & (nearest > 0) & (min_resized > 0) & \
        (interpolated_depth > 0) & (interpolated_depth <= depth_max) & (min_resized <= depth_max)
    return has_edge, np.where(has_edge, min_resized, nearest)



