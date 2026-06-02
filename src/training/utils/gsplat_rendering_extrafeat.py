import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

from gsplat.cuda._wrapper import (
    RollingShutterType,
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    fully_fused_projection,
    fully_fused_projection_2dgs,
    fully_fused_projection_with_ut,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    rasterize_to_pixels_2dgs,
    rasterize_to_pixels_eval3d,
    spherical_harmonics,
)
from gsplat.distributed import (
    all_gather_int32,
    all_gather_tensor_list,
    all_to_all_int32,
    all_to_all_tensor_list,
)
from gsplat.utils import depth_to_normal, get_projection_matrix
from depth_anything_3.utils.geometry import affine_inverse


def _compute_view_dirs_packed(
    means: Tensor,  # [..., N, 3]
    campos: Tensor,  # [..., C, 3]
    batch_ids: Tensor,  # [nnz]
    camera_ids: Tensor,  # [nnz]
    gaussian_ids: Tensor,  # [nnz]
    indptr: Tensor,  # [B*C+1]
    B: int,
    C: int,
) -> Tensor:
    """Compute view directions for packed Gaussian-camera pairs.

    This function computes the view directions (means - campos) for each
    Gaussian-camera pair in the packed format. It automatically selects between
    a simple vectorized approach or an optimized loop-based approach based on
    the data size and whether campos requires gradients.

    Args:
        means: The 3D centers of the Gaussians. [..., N, 3]
        campos: Camera positions in world coordinates [..., C, 3]
        batch_ids: The batch indices of the projected Gaussians. Int32 tensor of shape [nnz].
        camera_ids: The camera indices of the projected Gaussians. Int32 tensor of shape [nnz].
        gaussian_ids: The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        indptr: CSR-style index pointer into gaussian_ids for batch-camera pairs. Int32 tensor of shape [B*C+1].
        B: Number of batches
        C: Number of cameras

    Returns:
        dirs: View directions [nnz, 3]
    """
    N = means.shape[-2]
    nnz = batch_ids.shape[0]
    device = means.device
    means_flat = means.view(B, N, 3)
    campos_flat = campos.view(B, C, 3)

    if B * C == 1:
        # Single batch-camera pair. No indexed lookup for campos is needed.
        dirs = means_flat[0, gaussian_ids] - campos_flat[0, 0]  # [nnz, 3]
    else:
        avg_means_per_camera = nnz / (B * C)
        split_batch_camera_ops = (
            avg_means_per_camera > 10000
            and campos_flat.is_cuda
            and campos_flat.requires_grad
        )

        if not split_batch_camera_ops:
            # Simple vectorized indexing for campos.
            dirs = (
                means_flat[batch_ids, gaussian_ids] - campos_flat[batch_ids, camera_ids]
            )  # [nnz, 3]
        else:
            # For large N with pose optimization: split into B*C separate operations
            # to avoid many-to-one indexing of campos in backward pass. This speeds up the
            # backwards pass and is more impactful when GPU occupancy is high.
            dirs = torch.zeros((nnz, 3), dtype=means_flat.dtype, device=device)
            indptr_cpu = indptr.cpu()
            for b_idx in range(B):
                for c_idx in range(C):
                    bc_idx = b_idx * C + c_idx
                    start_idx = indptr_cpu[bc_idx].item()
                    end_idx = indptr_cpu[bc_idx + 1].item()
                    if start_idx == end_idx:
                        continue

                    # Get the gaussian indices for this batch-camera pair and compute dirs
                    gids = gaussian_ids[start_idx:end_idx]
                    dirs[start_idx:end_idx] = (
                        means_flat[b_idx, gids] - campos_flat[b_idx, c_idx]
                    )

    return dirs


def rasterization_extrafeat(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
    extra_feat_color_sh: Tensor,  # [..., (C,) N, D]
    extra_feat_color_value: Tensor,  # [..., (C,) N, C']
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
    segmented: bool = False,
    covars: Optional[Tensor] = None,
    with_ut: bool = False,
    with_eval3d: bool = False,
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of 3D Gaussians with extra features."""
    meta = {}

    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    device = means.device
    assert means.shape == batch_dims + (N, 3), means.shape
    if covars is None:
        assert quats.shape == batch_dims + (N, 4), quats.shape
        assert scales.shape == batch_dims + (N, 3), scales.shape
    else:
        assert covars.shape == batch_dims + (N, 3, 3), covars.shape
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    def reshape_view(C: int, world_view: torch.Tensor, N_world: list) -> torch.Tensor:
        view_list = list(
            map(
                lambda x: x.split(int(x.shape[0] / C), dim=0),
                world_view.split([C * N_i for N_i in N_world], dim=0),
            )
        )
        return torch.stack([torch.cat(l, dim=0) for l in zip(*view_list)], dim=0)

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (
            colors.dim() == num_batch_dims + 2
            and colors.shape[:-1] == batch_dims + (N,)
        ) or (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
        if distributed:
            assert (
                colors.dim() == num_batch_dims + 2
            ), "Distributed mode only supports per-Gaussian colors."
    else:
        # treat colors as SH coefficients, should be in shape [..., N, K, 3] or [..., C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-2] == batch_dims + (N,)
            and colors.shape[-1] == 3
        ) or (
            colors.dim() == num_batch_dims + 4
            and colors.shape[:-2] == batch_dims + (C, N)
            and colors.shape[-1] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape
        if distributed:
            assert (
                colors.dim() == num_batch_dims + 3
            ), "Distributed mode only supports per-Gaussian colors."

    if extra_feat_color_value is not None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (
            extra_feat_color_value.dim() == num_batch_dims + 2
            and extra_feat_color_value.shape[:-1] == batch_dims + (N,)
        ) or (
            extra_feat_color_value.dim() == num_batch_dims + 3
            and extra_feat_color_value.shape[:-1] == batch_dims + (C, N)
        ), extra_feat_color_value.shape
        if distributed:
            assert (
                extra_feat_color_value.dim() == num_batch_dims + 2
            ), "Distributed mode only supports per-Gaussian colors."
    
    if extra_feat_color_sh is not None:
        # treat colors as SH coefficients, should be in shape [..., N, K, 3] or [..., C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            extra_feat_color_sh.dim() == num_batch_dims + 3
            and extra_feat_color_sh.shape[:-2] == batch_dims + (N,)
            and extra_feat_color_sh.shape[-1] == 3
        ) or (
            extra_feat_color_sh.dim() == num_batch_dims + 4
            and extra_feat_color_sh.shape[:-2] == batch_dims + (C, N)
            and extra_feat_color_sh.shape[-1] == 3
        ), extra_feat_color_sh.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape
        if distributed:
            assert (
                extra_feat_color_sh.dim() == num_batch_dims + 3
            ), "Distributed mode only supports per-Gaussian colors."
    
    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."

    if (
        radial_coeffs is not None
        or tangential_coeffs is not None
        or thin_prism_coeffs is not None
        or ftheta_coeffs is not None
        or rolling_shutter != RollingShutterType.GLOBAL
    ):
        assert (
            with_ut
        ), "Distortion and rolling shutter are only supported with `with_ut=True`."

    if rolling_shutter != RollingShutterType.GLOBAL:
        assert (
            viewmats_rs is not None
        ), "Rolling shutter requires to provide viewmats_rs."
    else:
        assert (
            viewmats_rs is None
        ), "viewmats_rs should be None for global rolling shutter."

    if with_ut or with_eval3d:
        assert (quats is not None) and (
            scales is not None
        ), "UT and eval3d requires to provide quats and scales."
        assert packed is False, "Packed mode is not supported with UT."
        assert sparse_grad is False, "Sparse grad is not supported with UT."

    # Implement the multi-GPU strategy proposed in
    # `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`.
    #
    # If in distributed mode, we distribute the projection computation over Gaussians
    # and the rasterize computation over cameras. So first we gather the cameras
    # from all ranks for projection.
    if distributed:
        raise NotImplementedError("Distributed mode is not supported yet.")
        assert batch_dims == (), "Distributed mode does not support batch dimensions"
        world_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Gather the number of Gaussians in each rank.
        N_world = all_gather_int32(world_size, N, device=device)

        # Enforce that the number of cameras is the same across all ranks.
        C_world = [C] * world_size
        viewmats, Ks = all_gather_tensor_list(world_size, [viewmats, Ks])
        if viewmats_rs is not None:
            (viewmats_rs,) = all_gather_tensor_list(world_size, [viewmats_rs])

        # Silently change C from local #Cameras to global #Cameras.
        C = len(viewmats)

    if with_ut:
        proj_results = fully_fused_projection_with_ut(
            means,
            quats,
            scales,
            opacities,  # use opacities to compute a tigher bound for radii.
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            calc_compensations=(rasterize_mode == "antialiased"),
            camera_model=camera_model,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
            rolling_shutter=rolling_shutter,
            viewmats_rs=viewmats_rs,
        )

    else:
        # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
        proj_results = fully_fused_projection(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            packed=packed,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            sparse_grad=sparse_grad,
            calc_compensations=(rasterize_mode == "antialiased"),
            camera_model=camera_model,
            opacities=opacities,  # use opacities to compute a tigher bound for radii.
        )

    if packed:
        raise NotImplementedError("Packed mode is not supported yet.")
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            indptr,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities.view(B, N)[batch_ids, gaussian_ids]  # [nnz]
        image_ids = batch_ids * C + camera_ids
    else:
        # The results are with shape [..., C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = torch.broadcast_to(
            opacities[..., None, :], batch_dims + (C, N)
        )  # [..., C, N]
        indptr, batch_ids, camera_ids, gaussian_ids = None, None, None, None
        image_ids = None

    if compensations is not None:
        opacities = opacities * compensations

    meta.update(
        {
            # global batch and camera ids
            "batch_ids": batch_ids,
            "camera_ids": camera_ids,
            # local gaussian_ids
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
        }
    )

    # Turn colors into [..., C, N, D] or [..., nnz, D] to pass into rasterize_to_pixels()
    if sh_degree is None:
        # Colors are post-activation values, with shape [..., N, D] or [..., C, N, D]
        if packed:
            if colors.dim() == num_batch_dims + 2:
                # Turn [..., N, D] into [nnz, D]
                colors = colors.view(B, N, -1)[batch_ids, gaussian_ids]
            else:
                # Turn [..., C, N, D] into [nnz, D]
                colors = colors.view(B, C, N, -1)[batch_ids, camera_ids, gaussian_ids]
        else:
            if colors.dim() == num_batch_dims + 2:
                # Turn [..., N, D] into [..., C, N, D]
                colors = torch.broadcast_to(
                    colors[..., None, :, :], batch_dims + (C, N, -1)
                )
            else:
                # colors is already [..., C, N, D]
                pass
    else:
        # Colors are SH coefficients, with shape [..., N, K, 3] or [..., C, N, K, 3]
        # campos = torch.inverse(viewmats)[..., :3, 3]  # [..., C, 3]
        campos = affine_inverse(viewmats)[..., :3, 3]
        if viewmats_rs is not None:
            # campos_rs = torch.inverse(viewmats_rs)[..., :3, 3]
            campos_rs = affine_inverse(viewmats_rs)[..., :3, 3]
            campos = 0.5 * (campos + campos_rs)  # [..., C, 3]
        if packed:
            dirs = _compute_view_dirs_packed(
                means,
                campos,
                batch_ids,
                camera_ids,
                gaussian_ids,
                indptr,
                B,
                C,
            )  # [nnz, 3]

            masks = (radii > 0).all(dim=-1)  # [nnz]
            if colors.dim() == num_batch_dims + 3:
                # Turn [..., N, K, 3] into [nnz, 3]
                shs = colors.view(B, N, -1, 3)[batch_ids, gaussian_ids]  # [nnz, K, 3]
            else:
                # Turn [..., C, N, K, 3] into [nnz, 3]
                shs = colors.view(B, C, N, -1, 3)[
                    batch_ids, camera_ids, gaussian_ids
                ]  # [nnz, K, 3]
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [nnz, 3]
        else:
            dirs = means[..., None, :, :] - campos[..., None, :]  # [..., C, N, 3]
            masks = (radii > 0).all(dim=-1)  # [..., C, N]
            if colors.dim() == num_batch_dims + 3:
                # Turn [..., N, K, 3] into [..., C, N, K, 3]
                shs = torch.broadcast_to(
                    colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
                )
            else:
                # colors is already [..., C, N, K, 3]
                shs = colors
            colors = spherical_harmonics(
                sh_degree, dirs, shs, masks=masks
            )  # [..., C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)
    
    if extra_feat_color_value is not None:
        if extra_feat_color_value.dim() == num_batch_dims + 2:
            # Turn [..., N, D] into [..., C, N, D]
            extra_feat_color_value = torch.broadcast_to(
                extra_feat_color_value[..., None, :, :], batch_dims + (C, N, -1)
            )
    
    if extra_feat_color_sh is not None:
        # Colors are SH coefficients, with shape [..., N, K, 3] or [..., C, N, K, 3]
        # campos = torch.inverse(viewmats)[..., :3, 3]  # [..., C, 3]
        campos = affine_inverse(viewmats)[..., :3, 3]
        if viewmats_rs is not None:
            # campos_rs = torch.inverse(viewmats_rs)[..., :3, 3]
            campos_rs = affine_inverse(viewmats_rs)[..., :3, 3]
            campos = 0.5 * (campos + campos_rs)  # [..., C, 3]
        
        dirs = means[..., None, :, :] - campos[..., None, :]  # [..., C, N, 3]
        masks = (radii > 0).all(dim=-1)  # [..., C, N]
        if extra_feat_color_sh.dim() == num_batch_dims + 3:
            # Turn [..., N, K, 3] into [..., C, N, K, 3]
            shs = torch.broadcast_to(
                extra_feat_color_sh[..., None, :, :, :], batch_dims + (C, N, -1, 3)
            )
        else:
            # colors is already [..., C, N, K, 3]
            shs = extra_feat_color_sh
        extra_feat_color_sh = spherical_harmonics(
            sh_degree, dirs, shs, masks=masks
        )  # [..., C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        extra_feat_color_sh = torch.clamp_min(extra_feat_color_sh + 0.5, 0.0)
        

    # If in distributed mode, we need to scatter the GSs to the destination ranks, based
    # on which cameras they are visible to, which we already figured out in the projection
    # stage.
    if distributed:
        if packed:
            # count how many elements need to be sent to each rank
            cnts = torch.bincount(camera_ids, minlength=C)  # all cameras
            cnts = cnts.split(C_world, dim=0)
            cnts = [cuts.sum() for cuts in cnts]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            collected_splits = all_to_all_int32(world_size, cnts, device=device)
            (radii,) = all_to_all_tensor_list(
                world_size, [radii], cnts, output_splits=collected_splits
            )
            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [means2d, depths, conics, opacities, colors],
                cnts,
                output_splits=collected_splits,
            )

            # before sending the data, we should turn the camera_ids from global to local.
            # i.e. the camera_ids produced by the projection stage are over all cameras world-wide,
            # so we need to turn them into camera_ids that are local to each rank.
            offsets = torch.tensor(
                [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            camera_ids = camera_ids - offsets

            # and turn gaussian ids from local to global.
            offsets = torch.tensor(
                [0] + N_world[:-1],
                device=gaussian_ids.device,
                dtype=gaussian_ids.dtype,
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            gaussian_ids = gaussian_ids + offsets

            # all to all communication across all ranks.
            (camera_ids, gaussian_ids) = all_to_all_tensor_list(
                world_size,
                [camera_ids, gaussian_ids],
                cnts,
                output_splits=collected_splits,
            )

            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

        else:
            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            (radii,) = all_to_all_tensor_list(
                world_size,
                [radii.flatten(0, 1)],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            radii = reshape_view(C, radii, N_world)

            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [
                    means2d.flatten(0, 1),
                    depths.flatten(0, 1),
                    conics.flatten(0, 1),
                    opacities.flatten(0, 1),
                    colors.flatten(0, 1),
                ],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            means2d = reshape_view(C, means2d, N_world)
            depths = reshape_view(C, depths, N_world)
            conics = reshape_view(C, conics, N_world)
            opacities = reshape_view(C, opacities, N_world)
            colors = reshape_view(C, colors, N_world)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(batch_dims + (C, 1), device=backgrounds.device),
                ],
                dim=-1,
            )
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(batch_dims + (C, 1), device=backgrounds.device)
    else:  # RGB
        pass
    
    if extra_feat_color_value is not None:
        extra_feat_color_value_channel = extra_feat_color_value.shape[-1]
        colors = torch.cat((colors, extra_feat_color_value), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(batch_dims + (C, extra_feat_color_value_channel), device=backgrounds.device),
                ],
                dim=-1,
            )

    
    if extra_feat_color_sh is not None:
        extra_feat_color_sh_channel = extra_feat_color_sh.shape[-1]
        colors = torch.cat((colors, extra_feat_color_sh), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(batch_dims + (C, extra_feat_color_sh_channel), device=backgrounds.device),
                ],
                dim=-1,
            )

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        segmented=segmented,
        packed=packed,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_batches": B,
            "n_cameras": C,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            if with_eval3d:
                render_colors_, render_alphas_ = rasterize_to_pixels_eval3d(
                    means,
                    quats,
                    scales,
                    colors_chunk,
                    opacities,
                    viewmats,
                    Ks,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                    camera_model=camera_model,
                    radial_coeffs=radial_coeffs,
                    tangential_coeffs=tangential_coeffs,
                    thin_prism_coeffs=thin_prism_coeffs,
                    ftheta_coeffs=ftheta_coeffs,
                    rolling_shutter=rolling_shutter,
                    viewmats_rs=viewmats_rs,
                )
            else:
                render_colors_, render_alphas_ = rasterize_to_pixels(
                    means2d,
                    conics,
                    colors_chunk,
                    opacities,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                    packed=packed,
                    absgrad=absgrad,
                )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        if with_eval3d:
            render_colors, render_alphas = rasterize_to_pixels_eval3d(
                means,
                quats,
                scales,
                colors,
                opacities,
                viewmats,
                Ks,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds,
                camera_model=camera_model,
                radial_coeffs=radial_coeffs,
                tangential_coeffs=tangential_coeffs,
                thin_prism_coeffs=thin_prism_coeffs,
                ftheta_coeffs=ftheta_coeffs,
                rolling_shutter=rolling_shutter,
                viewmats_rs=viewmats_rs,
            )
        else:
            render_colors, render_alphas = rasterize_to_pixels(
                means2d,
                conics,
                colors,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds,
                packed=packed,
                absgrad=absgrad,
            )
    if extra_feat_color_sh is not None:
        render_color_sh = render_colors[..., -extra_feat_color_sh_channel:]
        render_colors = render_colors[..., :-extra_feat_color_sh_channel]
    if extra_feat_color_value is not None:
        render_color_value = render_colors[..., -extra_feat_color_value_channel:]
        render_colors = render_colors[..., :-extra_feat_color_value_channel]
        
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )

    return render_colors, render_alphas, meta, render_color_value, render_color_sh



###### 2DGS ######
def rasterization_2dgs_extrafeat(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
    extra_feat_color_value: Tensor,  # [..., (C,) N, C']
    extra_feat_color_sh: Tensor,  # [..., (C,) N, D]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = False,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: Literal["expected", "median"] = "expected",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """Rasterize a set of 2D Gaussians (N) to a batch of image planes (C).
    """

    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    device = means.device
    channels = colors.shape[-1]

    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode
    if distloss:
        assert render_mode in [
            "D",
            "ED",
            "RGB+D",
            "RGB+ED",
        ], f"distloss requires depth rendering, render_mode should be D, ED, RGB+D, RGB+ED, but got {render_mode}"

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (
            colors.dim() == num_batch_dims + 2
            and colors.shape[:-1] == batch_dims + (N,)
        ) or (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients, should be in shape [..., N, K, 3] or [..., C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-2] == batch_dims + (N,)
            and colors.shape[-1] == 3
        ) or (
            colors.dim() == num_batch_dims + 4
            and colors.shape[:-2] == batch_dims + (C, N)
            and colors.shape[-1] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    if extra_feat_color_value is not None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (
            extra_feat_color_value.dim() == num_batch_dims + 2
            and extra_feat_color_value.shape[:-1] == batch_dims + (N,)
        ) or (
            extra_feat_color_value.dim() == num_batch_dims + 3
            and extra_feat_color_value.shape[:-1] == batch_dims + (C, N)
        ), extra_feat_color_value.shape

    if extra_feat_color_sh is not None:
        # treat colors as SH coefficients, should be in shape [..., N, K, 3] or [..., C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            extra_feat_color_sh.dim() == num_batch_dims + 3
            and extra_feat_color_sh.shape[:-2] == batch_dims + (N,)
            # and extra_feat_color_sh.shape[-1] == 3
        ) or (
            extra_feat_color_sh.dim() == num_batch_dims + 4
            and extra_feat_color_sh.shape[:-2] == batch_dims + (C, N)
            # and extra_feat_color_sh.shape[-1] == 3
        ), extra_feat_color_sh.shape
        assert (sh_degree + 1) ** 2 <= extra_feat_color_sh.shape[-2], extra_feat_color_sh.shape

    
    # Compute Ray-Splat intersection transformation.
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        packed,
        sparse_grad,
    )

    if packed:
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = proj_results
        opacities = opacities.view(B, N)[batch_ids, gaussian_ids]
        image_ids = batch_ids * C + camera_ids
    else:
        radii, means2d, depths, ray_transforms, normals = proj_results
        opacities = torch.broadcast_to(
            opacities[..., None, :], batch_dims + (C, N)
        )  # [..., C, N]
        camera_ids, gaussian_ids = None, None
        image_ids = None

    densify = torch.zeros_like(
        means2d, dtype=means.dtype, requires_grad=True, device="cuda"
    )
    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # TODO: SH also suport N-D.
    if sh_degree is not None:  # SH coefficients
        # camtoworlds = torch.inverse(viewmats)
        camtoworlds = affine_inverse(viewmats)
        # print('camtoworlds', camtoworlds.shape, camtoworlds.min(), camtoworlds.max())
        # print('camtoworlds2', camtoworlds2.shape, camtoworlds2.min(), camtoworlds2.max())
        # print('diff', (camtoworlds - camtoworlds2).abs().max())
        
        if packed:
            dirs = means[..., gaussian_ids, :] - camtoworlds[..., camera_ids, :3, 3]
        else:
            dirs = means[..., None, :, :] - camtoworlds[..., None, :3, 3]

        # print('colors', colors.shape)
        if colors.dim() == num_batch_dims + 3:
            # Turn [..., N, K, 3] into [..., C, N, K, 3]
            shs = torch.broadcast_to(
                colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
            )  # [..., C, N, K, 3]
        else:
            # colors is already [..., C, N, K, 3]
            shs = colors
        
        assert not torch.isnan(shs).any(), "shs is nan"
        assert not torch.isinf(shs).any(), "shs is inf"
        assert not torch.isnan(dirs).any(), "dirs is nan"
        assert not torch.isinf(dirs).any(), "dirs is inf"
        # print((radii > 0).all(dim=-1).shape, dirs.shape, dirs)
        mask = ((radii > 0).all(dim=-1) & (dirs.norm(dim=-1) > 1e-10))
        colors = spherical_harmonics(
            # sh_degree, dirs, shs, masks=(radii > 0).all(dim=-1)
            sh_degree, dirs, shs, masks=mask
        )  # [nnz, D] or [..., C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)


    if extra_feat_color_sh is not None:
        # Colors are SH coefficients, with shape [..., N, K, 3] or [..., C, N, K, 3]
        # camtoworlds = torch.inverse(viewmats)
        camtoworlds = affine_inverse(viewmats)
        if packed:
            dirs = means[..., gaussian_ids, :] - camtoworlds[..., camera_ids, :3, 3]
        else:
            dirs = means[..., None, :, :] - camtoworlds[..., None, :3, 3]
        
        # print('extra_feat_color_sh', extra_feat_color_sh.shape)
        extra_feat_color_sh_channel = extra_feat_color_sh.shape[-1]
        if extra_feat_color_sh.dim() == num_batch_dims + 3:
            # Turn [..., N, K, 3] into [..., C, N, K, 3]
            shs = torch.broadcast_to(
                extra_feat_color_sh[..., None, :, :, :], batch_dims + (C, N, -1, 3)
            )  # [..., C, N, K, 3]
        else:
            # colors is already [..., C, N, K, 3]
            shs = extra_feat_color_sh
        extra_feat_color_sh = spherical_harmonics(
            sh_degree, dirs, shs, masks=(radii > 0).all(dim=-1)
        )  # [nnz, D] or [..., C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        # print('extra_feat_color_sh', extra_feat_color_sh.shape)
        extra_feat_color_sh = extra_feat_color_sh[..., :extra_feat_color_sh_channel]
        extra_feat_color_sh = torch.clamp_min(extra_feat_color_sh + 0.5, 0.0)

    if extra_feat_color_value is not None:
        extra_feat_color_value_channel = extra_feat_color_value.shape[-1]
        # print('extra_feat_color_value_channel', extra_feat_color_value_channel)
        colors = torch.cat((colors, extra_feat_color_value), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(batch_dims + (C, extra_feat_color_value_channel), device=backgrounds.device),
                ],
                dim=-1,
            )

    if extra_feat_color_sh is not None:
        # print('extra_feat_color_sh_channel', extra_feat_color_sh_channel)
        colors = torch.cat((colors, extra_feat_color_sh), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(batch_dims + (C, extra_feat_color_sh_channel), device=backgrounds.device),
                ],
                dim=-1,
            )


    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)

        if backgrounds is not None:
            backgrounds = torch.cat(
                (backgrounds, torch.zeros_like(backgrounds[..., :1])), dim=-1
            )
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
    else:  # RGB
        pass

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=packed,
        absgrad=absgrad,
        distloss=distloss,
    )
    render_normals_from_depth = None
    
    render_color_value, render_color_sh = None, None
    if extra_feat_color_sh is not None:
        render_color_sh = render_colors[..., -extra_feat_color_sh_channel-1:-1]
        # render_colors = render_colors[..., :-extra_feat_color_sh_channel]
        render_colors = torch.cat(
            [render_colors[..., :-extra_feat_color_sh_channel-1], 
             render_colors[..., -1:]], dim=-1
        )
    if extra_feat_color_value is not None:
        render_color_value = render_colors[..., -extra_feat_color_value_channel-1:-1]
        render_colors = torch.cat(
            [render_colors[..., :-extra_feat_color_value_channel-1], 
             render_colors[..., -1:]], dim=-1
        )
    
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
    if render_mode in ["RGB+ED", "RGB+D"]:
        # render_depths = render_colors[..., -1:]
        if depth_mode == "expected":
            depth_for_normal = render_colors[..., -1:]
        elif depth_mode == "median":
            depth_for_normal = render_median

        render_normals_from_depth = depth_to_normal(
            depth_for_normal, torch.linalg.inv(viewmats), Ks
        )

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "opacities": opacities,
        "normals": normals,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        "render_distort": render_distort,
        "gradient_2dgs": densify,  # This holds the gradient used for densification for 2dgs
    }

    render_normals = torch.einsum(
        "...ij,...hwj->...hwi", torch.linalg.inv(viewmats)[..., :3, :3], render_normals
    )

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        meta,
        render_color_value, render_color_sh
    )



class DeferredBP_rasterization_2dgs_extrafeat(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        means, quats, scales, opacities, colors,
        extra_feat_color_value, extra_feat_color_sh,
        viewmats, Ks, backgrounds,
        width, height, 
        render_mode, packed, sh_degree
    ):
        ctx.save_for_backward(
            means, quats, scales, opacities, colors,
            extra_feat_color_value, extra_feat_color_sh,
            viewmats, Ks, backgrounds
        )

        ctx.W = width
        ctx.H = height
        ctx.packed = packed
        ctx.sh_degree = sh_degree

        with torch.no_grad():
            render_colors, render_alphas, render_normals, render_normals_from_depth, render_distort, render_median, meta, render_color_value, render_color_sh = rasterization_2dgs_extrafeat(
                means, quats, scales, opacities, colors,
                viewmats, Ks,
                backgrounds=backgrounds, 
                width=width,
                height=height,
                render_mode=render_mode,
                packed=packed,
                sh_degree=sh_degree,
                extra_feat_color_value=extra_feat_color_value,
                extra_feat_color_sh=extra_feat_color_sh,
            )
        
        render_colors = render_colors.requires_grad_()
        render_alphas = render_alphas.requires_grad_()
        render_normals = render_normals.requires_grad_()
        render_normals_from_depth = render_normals_from_depth.requires_grad_()
        render_distort = render_distort.requires_grad_()
        render_median = render_median.requires_grad_()

        if render_color_value is not None:
            render_color_value = render_color_value.requires_grad_()
        if render_color_sh is not None:
            render_color_sh = render_color_sh.requires_grad_()

        return render_colors, render_alphas, render_normals, render_normals_from_depth, render_distort, render_median, meta, render_color_value, render_color_sh

    @staticmethod
    def backward(
        ctx,
        grad_render_colors,
        grad_render_alphas,
        grad_render_normals,
        grad_render_normals_from_depth,
        grad_render_distort,
        grad_render_median,
        grad_meta,
        grad_render_color_value,
        grad_render_color_sh,
    ):
        (
            means, quats, scales, opacities, colors,
            extra_feat_color_value, extra_feat_color_sh,
            viewmats, Ks, backgrounds
        ) = ctx.saved_tensors

        W = ctx.W
        H = ctx.H
        packed = ctx.packed
        sh_degree = ctx.sh_degree

        # Re-run forward WITH grad tracking.
        # IMPORTANT: if your forward ran under no_grad, this is the standard trick.
        means_ = means.detach().requires_grad_(True)
        quats_ = quats.detach().requires_grad_(True)
        scales_ = scales.detach().requires_grad_(True)
        opacities_ = opacities.detach().requires_grad_(True)
        colors_ = colors.detach().requires_grad_(True)

        if extra_feat_color_value is not None:
            extra_feat_color_value_ = extra_feat_color_value.detach().requires_grad_(True)
        else:
            extra_feat_color_value_ = None

        if extra_feat_color_sh is not None:
            extra_feat_color_sh_ = extra_feat_color_sh.detach().requires_grad_(True)
        else:
            extra_feat_color_sh_ = None

        viewmats_ = viewmats.detach().requires_grad_(True)
        Ks_ = Ks.detach().requires_grad_(True)
        backgrounds_ = backgrounds.detach().requires_grad_(True)

        (
            render_colors,
            render_alphas,
            render_normals,
            render_normals_from_depth,
            render_distort,
            render_median,
            meta,
            render_color_value,
            render_color_sh,
        ) = rasterization_2dgs_extrafeat(
            means_, quats_, scales_, opacities_, colors_,
            viewmats_, Ks_,
            backgrounds=backgrounds_,
            width=W,
            height=H,
            render_mode=ctx.render_mode if hasattr(ctx, "render_mode") else None,  # see note below
            packed=packed,
            sh_degree=sh_degree,
            extra_feat_color_value=extra_feat_color_value_,
            extra_feat_color_sh=extra_feat_color_sh_,
        )

        # Collect outputs + corresponding incoming grads (skip None safely)
        outputs = [
            render_colors,
            render_alphas,
            render_normals,
            render_normals_from_depth,
            render_distort,
            render_median,
        ]
        grad_outputs = [
            grad_render_colors,
            grad_render_alphas,
            grad_render_normals,
            grad_render_normals_from_depth,
            grad_render_distort,
            grad_render_median,
        ]

        if render_color_value is not None:
            outputs.append(render_color_value)
            grad_outputs.append(grad_render_color_value)
        if render_color_sh is not None:
            outputs.append(render_color_sh)
            grad_outputs.append(grad_render_color_sh)

        inputs = [
            means_, quats_, scales_, opacities_, colors_,
            extra_feat_color_value_, extra_feat_color_sh_,
            viewmats_, Ks_, backgrounds_,
        ]

        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        (
            g_means, g_quats, g_scales, g_opacities, g_colors,
            g_extra_feat_color_value, g_extra_feat_color_sh,
            g_viewmats, g_Ks, g_backgrounds
        ) = grads

        # Return one entry per forward() arg, in exact order.
        # Non-tensor args => None.
        return (
            g_means,                 # means
            g_quats,                 # quats
            g_scales,                # scales
            g_opacities,             # opacities
            g_colors,                # colors
            g_extra_feat_color_value,# extra_feat_color_value
            g_extra_feat_color_sh,   # extra_feat_color_sh
            g_viewmats,              # viewmats
            g_Ks,                    # Ks
            g_backgrounds,           # backgrounds
            None,                    # width
            None,                    # height
            None,                    # render_mode
            None,                    # packed
            None,                    # sh_degree
        )
