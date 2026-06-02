import numpy as np
from PIL import Image


def resize_nearest_w_bicubic_sampling(img: np.ndarray, new_size: tuple, multiplier=2) -> np.ndarray:
    """
    Resizes an image using Bicubic interpolation in a single aggregation step.
    
    - UPSCALING: Aggregates the 4x4 (16) surrounding pixels.
    - DOWNSCALING: Aggregates a larger area to match PIL's antialiasing.
    - Matches PIL.Image.BICUBIC (a = -0.5).
    """
    out_w, out_h = new_size
    
    orig_ndim = img.ndim
    if orig_ndim == 2:
        img = img[:, :, None]
    
    in_h, in_w = img.shape[:2]
    if (in_w, in_h) == (out_w, out_h):
        if orig_ndim == 2:
            img = img[:, :, 0]
        return img.copy(), None

    src = img.astype(np.float32)

    def bicubic_kernel(x, a=-0.5):
        # Optimized vectorized Bicubic implementation
        abs_x = np.abs(x)
        
        # Two regions: |x| < 1 and 1 <= |x| < 2
        # We process them using masks
        mask1 = abs_x < 1.0
        mask2 = (abs_x >= 1.0) & (abs_x < 2.0)
        
        # Region 1 formula: (a+2)|x|^3 - (a+3)|x|^2 + 1
        val1 = (a + 2) * (abs_x ** 3) - (a + 3) * (abs_x ** 2) + 1
        
        # Region 2 formula: a|x|^3 - 5a|x|^2 + 8a|x| - 4a
        val2 = a * (abs_x ** 3) - 5 * a * (abs_x ** 2) + 8 * a * abs_x - 4 * a
        
        # Combine (values outside |x| >= 2 are 0.0)
        return np.where(mask1, val1, np.where(mask2, val2, 0.0))

    def get_weights_and_indices(in_len, out_len):
        scale = out_len / in_len
        
        # Determine filter support
        # Upscaling: radius = 2.0 (Standard Bicubic)
        # Downscaling: radius = 2.0 / scale (Antialiasing)
        if scale < 1.0:
            support = 2.0 / scale
            filter_scale = scale
        else:
            support = 2.0
            filter_scale = 1.0
            
        # Output centers
        x = np.arange(out_len, dtype=np.float32)
        center = (x + 0.5) / scale - 0.5
        
        # Window of input pixels
        left = np.floor(center - support).astype(np.int32)
        right = np.ceil(center + support).astype(np.int32)
        max_k = int(np.max(right - left) + 1)
        
        # Create Index Grid: (out_len, max_k)
        k_offsets = np.arange(max_k, dtype=np.int32)
        indices = left[:, None] + k_offsets[None, :]
        
        # Calculate Weights
        dist = (center[:, None] - indices) * filter_scale
        weights = bicubic_kernel(dist)
        
        # Normalize weights
        w_sum = np.sum(weights, axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1.0
        weights /= w_sum
        
        # Clamp indices
        indices = np.clip(indices, 0, in_len - 1)
        
        return indices, weights

    # 1. Precompute 1D Weights and Indices
    x_idx, x_w = get_weights_and_indices(in_w, out_w)
    y_idx, y_w = get_weights_and_indices(in_h, out_h)
    
    # 2. Broadcast for 2D Gathering
    y_idx_b = y_idx[:, None, :, None]  # (out_h, 1, ky, 1)
    x_idx_b = x_idx[None, :, None, :]  # (1, out_w, 1, kx)
    
    # 3. Gather Pixels
    gathered_pixels = src[y_idx_b, x_idx_b]
    
    # 4. Compute 2D Weights (Outer Product)
    w_2d = y_w[:, None, :, None] * x_w[None, :, None, :]
    
    gathered_pixels = gathered_pixels.reshape(out_h, out_w, -1)
    probs = w_2d.reshape(out_h, out_w, -1)
    probs = np.clip(probs, 1e-10, 1)
    
    # probs = probs / probs.sum(axis=-1, keepdims=True)
    # print('probs', probs)
    
    probs = np.log(probs) * multiplier
    probs = probs - probs.max(axis=-1, keepdims=True)
    p_sharp = np.exp(probs)
    probs = p_sharp / (p_sharp.sum(axis=-1, keepdims=True) + 1e-12)
    
    # idx = np.argmax(probs, axis=-1)  # (H, W)
    # output_max = np.take_along_axis(gathered_pixels, idx[..., None], axis=-1)[..., 0]  # (H, W)
    
    output = numpy_sampling(gathered_pixels, probs)
    
    # close_ratio = np.sum( np.abs(output - output_max) < 1e-5) / (out_h * out_w)
    # print('close_ratio', close_ratio)
    # diff = np.abs(output - output_max).mean() / close_ratio
    # print('diff', diff)
    
    # return output, output_max
    return output, None



def numpy_sampling(gathered_pixels, probs, eps=1e-12):
    """
    gathered_pixels: (H, W, K)  values to sample from (K=36)
    probs:           (H, W, K)  nonnegative weights/probabilities
    returns:         (H, W)     one sampled value per pixel
    """
    gathered_pixels = np.asarray(gathered_pixels)
    probs = np.asarray(probs)

    if gathered_pixels.shape != probs.shape:
        raise ValueError(f"shape mismatch: {gathered_pixels.shape} vs {probs.shape}")
    if gathered_pixels.ndim != 3:
        raise ValueError("expected 3D arrays (H, W, K)")

    H, W, K = probs.shape

    # sanitize & normalize probs along K
    p = np.clip(probs, 0.0, None)
    s = p.sum(axis=-1, keepdims=True)
    # if a row sums to 0, fall back to uniform
    p = np.where(s > eps, p / s, 1.0 / K)

    # inverse-CDF sampling
    cdf = np.cumsum(p, axis=-1)                       # (H, W, K)
    u = np.random.rand(H, W, 1)                       # (H, W, 1) in [0,1)
    idx = (u <= cdf).argmax(axis=-1)                  # (H, W) first True along K

    # gather values
    out = np.take_along_axis(gathered_pixels, idx[..., None], axis=-1)[..., 0]
    return out
