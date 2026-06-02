import math

import torch
import torch.nn as nn


def calc_loglikelihood_l1(pred_depth, pred_conf, mog_weight, sampled_depths, gamma=1.0, alpha=0.2):
    """Negative log-likelihood of candidate depths under a Laplace mixture.

    Each expert is a Laplace component: location pred_depth[i], scale set by
    pred_conf[i].  For every candidate depth this returns -log of the mixture
    density, summed over experts with logsumexp.

    Shapes:
        pred_depth, pred_conf: [N, L, H, W]
        mog_weight:            [L, H, W, N]   (log mixing weights)
        sampled_depths:        [M, L, H, W]   (M candidate depths)
    Returns:
        nll:                   [M, L, H, W]
    """
    N = pred_depth.shape[0]
    device = pred_depth.device
    log_two = torch.log(torch.tensor(2.0, device=device))
    log_alpha = torch.log(torch.tensor(alpha, device=device))

    component_loglik = []
    for i in range(N):
        depth_i = pred_depth[i].unsqueeze(0)            # [1, L, H, W]
        conf_i = pred_conf[i].unsqueeze(0)              # [1, L, H, W]
        weight_i = mog_weight[..., i]                   # [L, H, W]

        residual = torch.abs(depth_i - sampled_depths)  # [M, L, H, W]
        nll_i = gamma * residual * conf_i - alpha * torch.log(conf_i)
        nll_i = nll_i / alpha + log_two + log_alpha
        component_loglik.append(weight_i - nll_i)        # [M, L, H, W]

    loglik = torch.stack(component_loglik, dim=-1)        # [M, L, H, W, N]
    return -torch.logsumexp(loglik, dim=-1)               # [M, L, H, W]


def calc_loglikelihood_l2(pred_depth, pred_conf, mog_weight, sampled_depths, gamma=1.0, alpha=0.2):
    """Negative log-likelihood of candidate depths under a Gaussian mixture.

    Same structure as calc_loglikelihood_l1, but each expert is Gaussian: the
    residual is squared and the density carries the 0.5 factor and the 2*pi
    normalization constant.

    Shapes match calc_loglikelihood_l1.
    """
    N = pred_depth.shape[0]
    device = pred_depth.device
    log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=device))
    log_inv_alpha = torch.log(torch.tensor(1.0 / alpha, device=device))

    component_loglik = []
    for i in range(N):
        depth_i = pred_depth[i].unsqueeze(0)            # [1, L, H, W]
        conf_i = pred_conf[i].unsqueeze(0)              # [1, L, H, W]
        weight_i = mog_weight[..., i]                   # [L, H, W]

        residual = torch.square(depth_i - sampled_depths)  # [M, L, H, W]
        nll_i = gamma * residual * conf_i - alpha * torch.log(conf_i)
        nll_i = nll_i / alpha + log_two_pi - log_inv_alpha
        component_loglik.append(weight_i - 0.5 * nll_i)    # [M, L, H, W]

    loglik = torch.stack(component_loglik, dim=-1)          # [M, L, H, W, N]
    return -torch.logsumexp(loglik, dim=-1)                 # [M, L, H, W]


def _prepare_inputs(preds, apply_log):
    """Stack the per-expert prediction lists into batched tensors (batch 0 only).

    Args:
        preds: dict with
            "depth"      - list of N tensors, each [B, L, H, W]
            "depth_conf" - list of N tensors, each [B, L, H, W]
            "mog_weight" - tensor [B, L, H, W, N] (log mixing weights)
        apply_log: if True, replace each depth d with log(d + 0.1) before
            stacking.  Used by the 'logl2' loss type.

    Returns:
        pred_depth: [N, L, H, W]
        pred_conf:  [N, L, H, W]
        mog_weight: [L, H, W, N]
    """
    depth_list = preds["depth"]
    conf_list = preds["depth_conf"]
    mog_weight = preds["mog_weight"].detach()[0]            # [L, H, W, N]

    if apply_log:
        depth_list = [torch.log(d + 0.1) for d in depth_list]

    pred_depth = torch.stack(depth_list, dim=1).detach()[0]  # [N, L, H, W]
    pred_conf = torch.stack(conf_list, dim=1).detach()[0]    # [N, L, H, W]
    return pred_depth, pred_conf, mog_weight


def _per_pixel_min_max(pred_depth):
    """Per-pixel min and max depth across the N experts.

    pred_depth: [N, L, H, W].  Returns two [1, L, H, W] tensors used to map
    depths into a normalized [0, 1] range and back.
    """
    depth_max = pred_depth.max(dim=0, keepdim=True)[0]
    depth_min = pred_depth.min(dim=0, keepdim=True)[0]
    return depth_min, depth_max


def _sample_candidate_depths(pred_depth, pred_conf):
    """Build the candidate depths to score: expert means plus pairwise midpoints.

    The midpoint between experts i and j is weighted by their scales
    b = 5 / conf, so it sits closer to the more confident expert.

    The midpoint count is capped at "just over 8": the inner loop breaks once
    more than 8 midpoints exist.  This matches the original behavior; the outer
    loop can still add one more midpoint per remaining expert.

    Args:
        pred_depth, pred_conf: [N, L, H, W], with N >= 2.
    Returns:
        candidates: [M, L, H, W], M >= N.
    """
    scale = 5.0 / pred_conf                       # b, the Laplace/Gaussian scale
    means = pred_depth.clone()

    midpoints = []
    N = pred_depth.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            b_i, b_j = scale[i], scale[j]
            midpoint = (b_i * pred_depth[j] + b_j * pred_depth[i]) / (b_i + b_j)
            midpoints.append(midpoint)
            if len(midpoints) > 8:
                break

    midpoints = torch.stack(midpoints, dim=0)
    return torch.cat([means, midpoints], dim=0)


class _RunningBest:
    """Tracks, per pixel, the candidate depth with the lowest NLL seen so far.

    All tensors have shape [L, H, W].  Call update() once per candidate set; at
    every pixel it keeps the candidate whose NLL is smallest across all calls.
    """

    def __init__(self, reference):
        # reference: a [L, H, W] tensor; supplies shape, dtype, and device.
        self.nll = torch.full_like(reference, 1e5)
        self.depth = torch.zeros_like(reference)
        self.indices = torch.zeros_like(reference)

    def update(self, nll, candidates):
        # nll, candidates: [M, L, H, W], M = number of candidates.
        assert nll.shape == candidates.shape, "nll and candidates must align"
        nll_now, index_now = nll.min(dim=0)                              # [L, H, W]
        depth_now = candidates.gather(0, index_now.unsqueeze(0)).squeeze(0)

        improved = nll_now < self.nll
        self.nll = torch.where(improved, nll_now, self.nll)
        self.depth = torch.where(improved, depth_now, self.depth)
        self.indices = torch.where(improved, index_now, self.indices)


def find_gmm_mode_gpu(preds, lr=1, steps=1, alpha=0.2, loss_type="l1",
                      saved_dir='logs/visualize_laplace_mixture'):
    """Find the per-pixel mode of the depth mixture for one chunk of frames.

    loss_type is matched by substring, so values may be combined (e.g.
    'logl2').  It selects the behavior:
        contains 'noopt' - score the expert means once, no optimization.
        contains 'onlyp' - pick the expert with the largest mixing weight,
                           ignoring how well its depth fits.
        otherwise        - score expert means plus pairwise midpoints, then
                           refine the candidates with LBFGS for `steps` steps.
        contains 'l2'    - use the Gaussian likelihood; otherwise Laplace.
        log-depth: the optimized path transforms when loss_type == 'logl2'
                   exactly; the 'noopt'/'onlyp' paths transform whenever
                   'logl2' is a substring.  Both conditions are kept as in the
                   original code.

    Args:
        preds: dict with "depth"/"depth_conf" (lists of N [B, L, H, W] tensors)
            and "mog_weight" ([B, L, H, W, N], log mixing weights).
        lr, steps: LBFGS learning rate and step count (optimized path only).
        alpha: scale parameter shared by the likelihood terms.
        saved_dir: unused; kept for call compatibility.

    Returns:
        best_depth: [1, L, H, W]  mode depth per pixel.
        best_index: [1, L, H, W]  index of the winning candidate per pixel.
    """
    is_noopt = 'noopt' in loss_type
    is_onlyp = 'onlyp' in loss_type
    optimize = not (is_noopt or is_onlyp)

    apply_log = ('logl2' in loss_type)

    pred_depth, pred_conf, mog_weight = _prepare_inputs(preds, apply_log)
    depth_min, depth_max = _per_pixel_min_max(pred_depth)

    # Candidate depths.  The optimized path adds pairwise midpoints when there
    # is more than one expert; the trivial paths score the means only.
    if optimize and pred_depth.shape[0] > 1:
        candidates = _sample_candidate_depths(pred_depth, pred_conf)
    else:
        candidates = pred_depth.clone()

    # Search in a normalized [0, 1] range for numerical stability.
    candidates_normed = (candidates - depth_min) / (depth_max - depth_min + 1e-8)
    candidates_normed = nn.Parameter(candidates_normed)

    def to_depth(normed):
        return normed * (depth_max - depth_min) + depth_min

    calc_loglikelihood = calc_loglikelihood_l2 if 'l2' in loss_type else calc_loglikelihood_l1

    def compute_nll(depths):
        if is_onlyp:
            # Score by mixing weight only: NLL = -log_weight per expert.
            return -mog_weight.permute(3, 0, 1, 2)
        # alpha defaults to 0.2 in every call path, so passing it here matches
        # the original 'noopt' path that relied on the default.
        return calc_loglikelihood(pred_depth, pred_conf, mog_weight, depths, alpha=alpha)

    best = _RunningBest(candidates_normed[0].detach())

    if optimize:
        optimizer = torch.optim.LBFGS([candidates_normed], lr=lr,
                                      line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            depths = to_depth(candidates_normed)
            nll = compute_nll(depths)
            with torch.no_grad():
                best.update(nll, depths)
            loss = nll.mean()
            loss.backward()
            return loss

        for _ in range(steps):
            optimizer.step(closure)
    else:
        with torch.no_grad():
            depths = to_depth(candidates_normed)
            best.update(compute_nll(depths), depths)

    best_depth = best.depth
    if apply_log:
        best_depth = torch.exp(best_depth) - 0.1

    return best_depth.unsqueeze(0), best.indices.unsqueeze(0)


def find_gmm_mode_gpu_chunk(preds, lr=1, steps=1, alpha=0.2, loss_type="l1",
                            saved_dir='logs/visualize_laplace_mixture', chunk_size=8):
    """Run find_gmm_mode_gpu over the frames in fixed-size chunks.

    Splits the L frames into chunks of `chunk_size` to bound memory, runs the
    mode finder on each, and concatenates the results along the frame axis.

    Args:
        preds: dict with "depth"/"depth_conf" (lists of N [B, L, H, W] tensors)
            and "mog_weight" ([B, L, H, W, N]).
    Returns:
        depths:  [1, L, H, W]
        indices: [1, L, H, W]
    """
    depth_list = preds["depth"]
    conf_list = preds["depth_conf"]
    mog_weight = preds["mog_weight"].detach()        # [B, L, H, W, N]

    L = depth_list[0].shape[1]
    depths_per_chunk = []
    indices_per_chunk = []
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        preds_chunk = {
            "depth": [d[:, start:end] for d in depth_list],
            "depth_conf": [c[:, start:end] for c in conf_list],
            "mog_weight": mog_weight[:, start:end],
        }
        depths_i, indices_i = find_gmm_mode_gpu(
            preds_chunk, lr=lr, steps=steps, alpha=alpha,
            loss_type=loss_type, saved_dir=saved_dir)
        depths_per_chunk.append(depths_i)
        indices_per_chunk.append(indices_i)

    depths = torch.cat(depths_per_chunk, dim=1)
    indices = torch.cat(indices_per_chunk, dim=1)
    return depths, indices


def find_gmm_mode_gpu_multilayer(preds, lr=1, steps=1, alpha=0.2, loss_type="l1",
                                 saved_dir='logs/visualize_laplace_mixture', chunk_size=8):
    """Chunked mode finder for models with a transparent-surface depth layer.

    In addition to the primary per-pixel mode depth, this returns a second
    depth layer for transparent/glass regions and a mask marking where it is
    valid.  It reuses find_gmm_mode_gpu_chunk for the primary depth, then reads
    the extra layer from the predictions.

    The model is trained so that:
        preds["depth"][0]   -> foreground / surface depth
        preds["depth"][-1]  -> background depth seen through transparent objects
        mog_weight_raw      -> per-expert sigmoid weights; all near 1 at
                               transparent pixels, sum near 1 at opaque pixels.

    Args:
        preds: dict with
            "depth"          - list of N tensors [B, L, H, W]
            "depth_conf"     - list of N tensors [B, L, H, W]
            "mog_weight"     - log mixing weights [B, L, H, W, N]
            "mog_weight_raw" - sigmoid weights     [B, L, H, W, N]
        chunk_size: process this many frames at a time (memory budget).

    Returns:
        primary_depth    : [1, L, H, W]  primary depth (highest-density component)
        indices          : [1, L, H, W]  selected component index per pixel
        extra_depth      : [1, L, H, W]  last-expert depth (through-glass / background)
        transparent_mask : [1, L, H, W]  bool mask, True where transparent objects detected
    """
    assert 'noopt' not in loss_type and 'onlyp' not in loss_type, \
        "find_gmm_mode_gpu_multilayer supports only the optimized loss types"

    primary_depth, indices = find_gmm_mode_gpu_chunk(
        preds, lr=lr, steps=steps, alpha=alpha,
        loss_type=loss_type, saved_dir=saved_dir, chunk_size=chunk_size)

    N = len(preds["depth"])
    mog_weight_raw = preds["mog_weight_raw"].detach()           # [B, L, H, W, N], sigmoid

    # Background / through-glass depth: the last expert, batch index 0.
    extra_depth = preds["depth"][-1].detach()[0].unsqueeze(0)   # [1, L, H, W]

    # Transparent pixels: every expert is active at once, so the sigmoid
    # weights all sit near 1 and their sum exceeds N/2.
    mog_weight_sum = mog_weight_raw[0].sum(dim=-1)              # [L, H, W]
    transparent_mask = (mog_weight_sum > N / 2).unsqueeze(0)    # [1, L, H, W]

    return primary_depth, indices, extra_depth, transparent_mask
