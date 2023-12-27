import torch
from einops import rearrange, reduce, repeat
from typing import List, Dict, Any, Optional

from models.nerf_model import UDCNeRF

import torch.nn.functional as F


__all__ = ["render_rays", "sample_pdf"]


cal_miou = True 
cal_depth = False 


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    N_importance: int,
    det=False,
    eps=1e-5,
):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, "n1 n2 -> n1 1", "sum")  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(
        torch.stack([below, above], -1), "n1 n2 c -> n1 (n2 c)", c=2
    )
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), "n1 (n2 c) -> n1 n2 c", c=2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0,
    # in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )
    return samples


def inference_model_udc(
    results: Dict[str, Any],
    model: UDCNeRF,
    embeddings: Dict[str, Any],
    typ: str,
    xyz: torch.Tensor,
    instance_mask: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    chunk: int,
    noise_std: float,
    white_back: bool,
    is_eval: bool = False,
    use_zero_as_last_delta: bool = False,
    forward_instance: bool = True,
    embedding_instance: Optional[torch.Tensor] = None,
    frustum_bound_th: float = 0,
    pass_through_mask: Optional[torch.Tensor] = None,
    rays_in_bbox: bool = False,
    use_fused_alpha_for_opacity: bool = False, # false
    use_alpha_for_fuse: bool = False, # false
    remove_bgobj: bool = True, # true
    block_enable: bool = True, 
    block_type: str = 'level0' , # ['level0', 'leval1', 'level2']
    gumbel_tau: float = 1.0, 
    **dummy_kwargs,
):

    embedding_xyz, embedding_dir = embeddings["xyz"], embeddings["dir"]
    N_rays, N_samples_, _ = xyz.shape
    xyz_ = rearrange(xyz, "n1 n2 c -> (n1 n2) c")  # (N_rays*N_samples_, 3)

    # Embed direction
    dir_embedded = embedding_dir(rays_d)  # (N_rays, embed_dir_channels)
    dir_embedded_ = repeat(
        dir_embedded, "n1 1 c -> (n1 n2) c", n2=N_samples_
    )  # (N_rays*N_samples_, embed_dir_channels)

    sigma_chunks = []
    rgb_chunks = []
    unmasked_sigma_chunk = []
    unmasked_rgb_chunk = []

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    # use_voxel_embedding = isinstance(embedding_xyz, EmbeddingVoxel)
    use_voxel_embedding = False
    for i in range(0, B, chunk):
        if use_voxel_embedding:
            xyz_embedded, obj_voxel_embbeded = embedding_xyz(xyz_[i : i + chunk])
        else:
            xyz_embedded = embedding_xyz(xyz_[i : i + chunk])


        if forward_instance:
            inst_output = model.forward_instance(
                {
                    "xyz": xyz_[i:i+chunk],
                    "emb_xyz": xyz_embedded,
                    "emb_dir": dir_embedded_[i : i + chunk],
                    # "N_rays": int(xyz_[i:i+chunk].shape[0] / N_samples_),
                    # "N_samples": N_samples_ # points in the same rays
                }
            )

            rgb_chunks += [inst_output["rgb"]]
            sigma_chunks += [inst_output["sigma"]]

            if typ == 'fine':
                unmasked_sigma_chunk += [inst_output["unmasked_sigma"]] # (P,K,1)
                unmasked_rgb_chunk += [inst_output["unmasked_rgb"]] # (P,K,3)

    
    sigmas = torch.cat(sigma_chunks, 0).view(N_rays, N_samples_)
    rgbs = torch.cat(rgb_chunks, 0).view(N_rays, N_samples_, 3)


    if forward_instance:
        if typ == 'fine':
            unmasked_sigmas = torch.cat(unmasked_sigma_chunk, 0).view(N_rays, N_samples_, -1) # (HW,D,K)
            unmasked_rgbs = torch.cat(unmasked_rgb_chunk, 0).view(N_rays, N_samples_, -1, 3) # (HW,D,K,3)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)

    # (N_rays, 1) the last delta is infinity
    delta_zero = torch.zeros_like(deltas[:, :1])
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])

    # For instance branch: use inf for bg, use zero for fg
    if forward_instance and typ == 'fine':
        deltas_unmasked_ = []
        for k in range(unmasked_sigmas.shape[-1]):
            if use_zero_as_last_delta:
                deltas_unmasked_ += [torch.cat([deltas, delta_zero], -1)]
            else:
                if k == 0: # bg
                    deltas_unmasked_ += [torch.cat([deltas, delta_inf], -1)]
                else:
                    deltas_unmasked_ += [torch.cat([deltas, delta_zero], -1)]

    # For scene branch: use inf as last delta
    if use_zero_as_last_delta:
        deltas = torch.cat([deltas, delta_zero], -1)  # (N_rays, N_samples_)
    else:
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
    
    if forward_instance:
        deltas_instance = deltas
        deltas_merged = deltas
    

    noise = torch.randn_like(sigmas) * noise_std
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)

    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
    )  # [1, 1-a1, 1-a2, ...]
    weights = alphas * torch.cumprod(alphas_shifted[:, :-1], -1)  # (N_rays, N_samples_)

    weights_sum = reduce(
        weights, "n1 n2 -> n1", "sum"
    )  # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    results[f"weights_{typ}"] = weights

    rgb_map = reduce(
        rearrange(weights, "n1 n2 -> n1 n2 1") * rgbs, "n1 n2 c -> n1 c", "sum"
    )

    if white_back:
        rgb_map = rgb_map + 1 - weights_sum.unsqueeze(-1)

    results[f"rgb_{typ}"] = rgb_map

    if cal_depth:
        results[f"depth_{typ}"] = reduce(weights * z_vals, "n1 n2 -> n1", "sum")


    ## matually remove the objects: not used finally
    if remove_bgobj:
        # print('remove_bgobj: ', remove_bgobj)
        ind_x = torch.logical_and(xyz_[:,0] > -0.28, xyz_[:,0] < 0.3) # w
        ind_y = torch.logical_and(xyz_[:,1] > -0.25, xyz_[:,1] < 0.3)
        ind_z = torch.logical_and(xyz_[:,2] > -0.2, xyz_[:,2] < 0.25)
        ind = torch.logical_and(torch.logical_and(ind_x,ind_y),ind_z)

        a, b, c = 0.004, 0.04, 0.016
        ind_box = (a * xyz_[:,0] + b * xyz_[:,1] + c * xyz_[:,2]) < 0 
        ind = torch.logical_and(ind, ind_box)
        
        unmasked_sigmas = unmasked_sigmas.reshape(N_rays*N_samples_, -1, 1)
        unmasked_sigmas[ind,0,:] = -5000000 #
        unmasked_sigmas = unmasked_sigmas.reshape(N_rays, N_samples_, -1)


    if typ == 'fine':

        masks_pre = F.gumbel_softmax(unmasked_sigmas.reshape(N_rays*N_samples_, -1, 1), dim=1, hard=True) # (P,K,1)
        masks_pre = masks_pre.bool()


        if not block_enable:
            unmasked_sigmas_modi = unmasked_sigmas * 1.0
            pass

        elif block_type == 'sigma' and block_enable:
            # copy unmasked_sigmas
            unmasked_sigmas = unmasked_sigmas.reshape(N_rays*N_samples_, -1, 1) # (HWD, K, 1)
            unmasked_sigmas_modi = unmasked_sigmas * 1.0
            # save modified sigmas
            ind_block = torch.logical_not(masks_pre[:,0,0]) # P,K,1 -> P  # get fg index
            results[f"3D_sigmas_modi_{typ}"] = unmasked_sigmas[ind_block, 0, :]
            # modify sigmas
            unmasked_sigmas_modi[ind_block, 0, :] = -5e6
            unmasked_sigmas_modi = unmasked_sigmas_modi.reshape(N_rays, N_samples_, -1)
        else:
            print('Unsuppoerted block level', block_type)

        # Gumbel-Softmax fusion: sigma & rgb
        masks = F.gumbel_softmax(unmasked_sigmas_modi.reshape(N_rays*N_samples_, -1, 1), dim=1, tau=gumbel_tau, hard=True) # (P,K,1)
        masked_sigmas = unmasked_sigmas_modi.reshape(N_rays*N_samples_, -1, 1) * masks # (P,K,1)
        masked_rgbs = unmasked_rgbs.reshape(N_rays*N_samples_, -1, 3) * masks # (P,K,3)
        
        instance_sigma = reduce(masked_sigmas, 'p k c -> p c', 'sum').reshape(N_rays, N_samples_)
        instance_rgb = reduce(masked_rgbs, 'p k c -> p c', 'sum').reshape(N_rays, N_samples_, 3)
        

    if typ == 'fine':
        merged_sigmas = (sigmas + instance_sigma) / 2
        merged_rgbs = (rgbs + instance_rgb) / 2
        noise_merged = torch.randn_like(merged_sigmas) * noise_std
        alphas_merged = 1 - torch.exp(-deltas_merged * torch.relu(merged_sigmas + noise_merged))  # (N_rays, N_samples_)

        alphas_shifted_merged = torch.cat(
            [torch.ones_like(alphas_merged[:, :1]), 1 - alphas_merged + 1e-10], -1
        )  # [1, 1-a1, 1-a2, ...]
        weights_merged = alphas_merged * torch.cumprod(alphas_shifted_merged[:, :-1], -1)  # (N_rays, N_samples_)


        rgb_map_merged = reduce(
            rearrange(weights_merged, "n1 n2 -> n1 n2 1") * merged_rgbs, "n1 n2 c -> n1 c", "sum"
        )

        results[f"rgb_merged_{typ}"] = rgb_map_merged
        results[f"weights_merged_{typ}"] = weights_merged

        if cal_depth:
            results[f"depth_merged_{typ}"] = reduce(weights_merged * z_vals, "n1 n2 -> n1", "sum")

        if cal_miou:
            results[f"merged_masks_{typ}"] = torch.sum(weights_merged.unsqueeze(-1)*masks.reshape(N_rays, N_samples_,-1), 1) # N_rays, K



    if typ == 'fine' and forward_instance:
        noise_instance = torch.randn_like(instance_sigma) * noise_std
        alphas_instance = 1 - torch.exp(
            -deltas_instance * torch.relu(instance_sigma + noise_instance)
        )  # (N_rays, N_samples_)
        alphas_shifted_instance = torch.cat(
            [torch.ones_like(alphas_instance[:, :1]), 1 - alphas_instance + 1e-10],
            -1,
        )  # [1, 1-a1, 1-a2, ...]
        weights_instance = alphas_instance * torch.cumprod(
            alphas_shifted_instance[:, :-1], -1
        )  # (N_rays, N_samples_)

        weights_sum_instance = reduce(weights_instance, "n1 n2 -> n1", "sum")

        # compute instance rgb and depth
        rgb_instance_map = reduce(
            rearrange(weights_instance, "n1 n2 -> n1 n2 1") * instance_rgb,
            "n1 n2 c -> n1 c",
            "sum",
        )
        if white_back:
            rgb_instance_map = rgb_instance_map + 1 - weights_sum_instance.unsqueeze(-1)
        results[f"rgb_instance_{typ}"] = rgb_instance_map
        results[f"weights_instance_{typ}"] = weights_instance

        if cal_depth:
            results[f"depth_instance_{typ}"] = reduce(weights_instance * z_vals, "n1 n2 -> n1", "sum")

        if cal_miou: 
            results[f"instance_masks_{typ}"] = torch.sum(weights_instance.unsqueeze(-1)*masks.reshape(N_rays, N_samples_,-1), 1) # N_rays, K

        
        # rendering for each bg/fgs:
        weights_unmasked_ = []
        weights_sum_unmasked_ = []
        rgb_map_unmasked_ = []
        depth_map_unmasked_ = []
        disp_map_unmasked_ = []
        for k in range(unmasked_sigmas_modi.shape[-1]):
            deltas_unmasked = deltas_unmasked_[k]
            unmasked_sigma = unmasked_sigmas_modi[:,:,k]
            unmasked_rgb = unmasked_rgbs[:,:,k,:]

            noise_unmasked = torch.randn_like(unmasked_sigma) * noise_std
            alphas_unmasked = 1 - torch.exp(
                -deltas_unmasked * torch.relu(unmasked_sigma)
            )  # (N_rays, N_samples_)

            alphas_shifted_unmasked = torch.cat(
            [torch.ones_like(alphas_unmasked[:, :1]), 1 - alphas_unmasked + 1e-10],
            -1,
            )  # [1, 1-a1, 1-a2, ...]

            if not use_fused_alpha_for_opacity:
                weights_unmasked = alphas_unmasked * torch.cumprod(
                    alphas_shifted_unmasked[:, :-1], -1
                )  # (N_rays, N_samples_)
            else:
                weights_unmasked = alphas_unmasked * torch.cumprod(
                    alphas_shifted_instance[:, :-1], -1
                )  # (N_rays, N_samples_)
                

            weights_unmasked_ += [weights_unmasked]
            weights_sum_unmasked = reduce(weights_unmasked, "n1 n2 -> n1", "sum")
            weights_sum_unmasked_ += [weights_sum_unmasked] # 0-1

            # compute unmasked rgb and depth
            rgb_unmasked_map = reduce(
                rearrange(weights_unmasked, "n1 n2 -> n1 n2 1") * unmasked_rgb,
                "n1 n2 c -> n1 c",
                "sum",
            )

            # 1011 add for debug:
            depth_unmasked_map = reduce(weights_unmasked * z_vals, "n1 n2 -> n1", "sum")
            depth_map_unmasked_ += [depth_unmasked_map]

            disp_unmasked_map = 1. / torch.max(1e-10 * torch.ones_like(depth_unmasked_map), depth_unmasked_map / torch.sum(weights_unmasked, -1))
            disp_map_unmasked_ += [disp_unmasked_map]

            if white_back:
                rgb_unmasked_map = rgb_unmasked_map + 1 - weights_sum_unmasked.unsqueeze(-1)
            rgb_map_unmasked_ += [rgb_unmasked_map]
        
        weights_sum_unmasked_ = torch.stack(weights_sum_unmasked_, 0).transpose(0,1) # ?xK
        results[f"opacity_unmasked_{typ}"] = weights_sum_unmasked_

        rgb_map_unmasked_ = torch.stack(rgb_map_unmasked_, 0) # k,hw,3
        rgb_map_unmasked_ = rgb_map_unmasked_.permute([1,2,0]) # hw,3,k
        results[f"rgb_unmasked_{typ}"] = rgb_map_unmasked_

        depth_map_unmasked_ = torch.stack(depth_map_unmasked_, 0) # k,hw
        depth_map_unmasked_ = depth_map_unmasked_.permute([1,0]) # hw,k
        results[f"depth_unmasked_{typ}"] = depth_map_unmasked_

        disp_map_unmasked_ = torch.stack(disp_map_unmasked_, 0) # k,hw
        disp_map_unmasked_ = disp_map_unmasked_.permute([1,0]) # hw,k
        results[f"disp_unmasked_{typ}"] = disp_map_unmasked_

        results[f"z_vals_{typ}"] = z_vals #hw, n_samples





    return

def render_rays_udc(
    models: Dict[str, UDCNeRF],
    embeddings: Dict[str, Any], # dict of xyz & dir
    rays: torch.Tensor, 
    N_samples: int = 64, # 64
    use_disp: bool = False, # false
    perturb: float = 0, # 1
    noise_std: float = 1, # 1
    N_importance: int = 0, # 64
    chunk: int = 1024 * 32, # 32768
    white_back: bool = False, # false
    forward_instance: bool = True, # Edit here
    use_zero_as_last_delta: bool = False, # Edit here
    embedding_instance: Optional[torch.Tensor] = None, # none
    frustum_bound_th: float = 0, # -0.0625
    pass_through_mask: Optional[torch.Tensor] = None, # none
    rays_in_bbox: bool = False, # false
    use_fused_alpha_for_opacity: bool = False, # false
    use_alpha_for_fuse: bool = False, # false
    remove_bgobj: bool = True, # true
    block_enable: bool = False, 
    block_type: str = 'level0' , # ['level0', 'leval1', 'level2']
    gumbel_tau: float = 1.0, # 
    use_which_weight_for_fine: str = 'global',
    **dummy_kwargs,
):
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    rays_o = rearrange(rays_o, "n1 c -> n1 1 c")
    rays_d = rearrange(rays_d, "n1 c -> n1 1 c")

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
    if not use_disp:  # use linear sampling in depth space # not false
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals) # 1
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")

    results = {}

    inference_model_udc(
        results=results,
        model=models["coarse"],
        embeddings=embeddings,
        typ="coarse",
        xyz=xyz_coarse,
        rays_d=rays_d,
        z_vals=z_vals,
        chunk=chunk,
        noise_std=noise_std, # 1
        white_back=white_back, # false
        forward_instance=forward_instance, # 
        use_zero_as_last_delta=use_zero_as_last_delta,
        embedding_instance=embedding_instance,
        frustum_bound_th=frustum_bound_th, # -0.0625
        pass_through_mask=pass_through_mask, # None
        rays_in_bbox=rays_in_bbox, # False
        use_fused_alpha_for_opacity=use_fused_alpha_for_opacity, #false
        remove_bgobj=remove_bgobj, # true
        block_enable=block_enable,
        block_type=block_type, # ['level0', 'leval1', 'level2']
        gumbel_tau=gumbel_tau,
        **dummy_kwargs,
    )

    if N_importance > 0:  # sample points for fine model
        if use_which_weight_for_fine == 'global':
            z_vals_ = sample_pdf(
                z_vals_mid,
                results["weights_coarse"][:, 1:-1].detach(),
                N_importance,
                det=(perturb == 0),
            ) # detach so that grad doesn't propogate to weights_coarse from here
        elif use_which_weight_for_fine == 'local':
            z_vals_ = sample_pdf(
                z_vals_mid,
                results["weights_instance_coarse"][:, 1:-1].detach(),
                N_importance,
                det=(perturb == 0),
            ) # detach so that grad doesn't propogate to weights_coarse from here
        elif use_which_weight_for_fine == 'comp':
            z_vals_ = sample_pdf(
                z_vals_mid,
                results["weights_merged_coarse"][:, 1:-1].detach(),
                N_importance,
                det=(perturb == 0),
            ) # detach so that grad doesn't propogate to weights_coarse from here
        else:
            print('not supported use_which_weight_for_fine')
            exit()

        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
        # combine coarse and fine samples

        xyz_fine = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")

        inference_model_udc(
            results=results,
            model=models["fine"],
            embeddings=embeddings,
            typ="fine",
            xyz=xyz_fine,
            rays_d=rays_d,
            z_vals=z_vals,
            chunk=chunk,
            noise_std=noise_std,
            white_back=white_back,
            forward_instance=forward_instance,
            use_zero_as_last_delta=use_zero_as_last_delta,
            embedding_instance=embedding_instance,
            frustum_bound_th=frustum_bound_th,
            pass_through_mask=pass_through_mask,
            rays_in_bbox=rays_in_bbox,
            use_fused_alpha_for_opacity=use_fused_alpha_for_opacity,
            remove_bgobj=remove_bgobj, # true
            block_enable=block_enable,
            block_type=block_type, # ['level0', 'leval1', 'level2']
            gumbel_tau=gumbel_tau,
            **dummy_kwargs,
        )

    return results




def volume_rendering_multi(
    results: Dict[str, Any],
    typ: str,
    z_vals_list: list,
    rgbs_list: list,
    sigmas_list: list,
    noise_std: float,
    white_back: bool,
    obj_ids_list: list = None,
):
    N_objs = len(z_vals_list)
    z_vals = torch.cat(z_vals_list, 1)  # (N_rays, N_samples*N_objs)
    rgbs = torch.cat(rgbs_list, 1)  # (N_rays, N_samples*N_objs, 3)
    sigmas = torch.cat(sigmas_list, 1)  # (N_rays, N_samples*N_objs)

    z_vals, idx_sorted = torch.sort(z_vals, -1)
    # # TODO(ybbbbt): ugly order three axis
    for i in range(3):
        rgbs[:, :, i] = torch.gather(rgbs[:, :, i], dim=1, index=idx_sorted)
    sigmas = torch.gather(sigmas, dim=1, index=idx_sorted)
    # record object ids for recovering weights of each object after sorting
    if obj_ids_list != None:
        obj_ids = torch.cat(obj_ids_list, -1)
        results[f"obj_ids_{typ}"] = torch.gather(obj_ids, dim=1, index=idx_sorted)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    # delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    delta_inf = torch.zeros_like(
        deltas[:, :1]
    )  # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # compute alpha by the formula (3)
    noise = torch.randn_like(sigmas) * noise_std
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)

    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
    )  # [1, 1-a1, 1-a2, ...]
    weights = alphas * torch.cumprod(alphas_shifted[:, :-1], -1)  # (N_rays, N_samples_)

    weights_sum = reduce(
        weights, "n1 n2 -> n1", "sum"
    )  # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    results[f"weights_{typ}"] = weights
    results[f"opacity_{typ}"] = weights_sum
    results[f"z_vals_{typ}"] = z_vals

    rgb_map = reduce(
        rearrange(weights, "n1 n2 -> n1 n2 1") * rgbs, "n1 n2 c -> n1 c", "sum"
    )
    depth_map = reduce(weights * z_vals, "n1 n2 -> n1", "sum")

    if white_back:
        rgb_map = rgb_map + 1 - weights_sum.unsqueeze(-1)

    results[f"rgb_{typ}"] = rgb_map
    results[f"depth_{typ}"] = depth_map




def inference_from_model_udc(
    model: UDCNeRF,
    embedding_xyz: torch.nn.Module,
    dir_embedded: torch.Tensor,
    xyz: torch.Tensor,
    z_vals: torch.Tensor,
    chunk: int,
    instance_id: int,
    typ: str,
    scene_name: str
):
    # print('inference_from_model for object %d: '%instance_id)
    N_rays = xyz.shape[0]
    N_samples_ = xyz.shape[1]
    xyz_ = rearrange(xyz, "n1 n2 c -> (n1 n2) c")  # (N_rays*N_samples_, 3)

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    rgb_chunks = []
    sigma_chunks = []
    unmasked_sigma_chunk = []
    unmasked_rgb_chunk = []
    
    dir_embedded_ = repeat(dir_embedded, "n1 c -> (n1 n2) c", n2=N_samples_)
    for i in range(0, B, chunk):
        xyz_embedded = embedding_xyz(xyz_[i : i + chunk])
        if True:
            inst_output = model.forward_instance(
                {
                    "xyz": xyz_[i:i+chunk],
                    "emb_xyz": xyz_embedded,
                    "emb_dir": dir_embedded_[i : i + chunk],
                    "N_rays": int(xyz_[i:i+chunk].shape[0] / N_samples_),
                    "N_samples": N_samples_ # points in the same rays
                }
            )

            if typ == 'fine':
                unmasked_sigma_chunk += [inst_output["unmasked_sigma"]] # (P,K,1)
                unmasked_rgb_chunk += [inst_output["unmasked_rgb"]] # (P,K,3)
            else:
                rgb_chunks += [inst_output["rgb"]]
                sigma_chunks += [inst_output["sigma"]]

    if typ == 'fine':
        unmasked_sigmas = torch.cat(unmasked_sigma_chunk, 0).view(N_rays, N_samples_, -1) # (HW,D,K)
        unmasked_rgbs = torch.cat(unmasked_rgb_chunk, 0).view(N_rays, N_samples_, -1, 3) # (HW,D,K,3)
    else:
        sigmas = torch.cat(sigma_chunks, 0).view(N_rays, N_samples_)
        rgbs = torch.cat(rgb_chunks, 0).view(N_rays, N_samples_, 3)

    if scene_name == 'toydesk2':
        mapp = {
            0:0,
            1:4,
            2:3,
            3:5,
            4:2,
            5:1,
        }
    elif scene_name == 'scannet0113':
        mapp = {
            0:0,
            4:1,
            6:2,
        }

    # mapp = {
    #     0:0,
    #     9:1,
    # }

    if typ == 'fine':
        K = len(mapp.keys())
        unmasked_sigmas = unmasked_sigmas.reshape(-1,K)

        if scene_name == 'toydesk2':
            for kk in range(K):
            # for kk in [0,4,6]:
            # for kk in [0,9]:
                if kk == instance_id:
                    continue
                indd = unmasked_sigmas[:,mapp[kk]] > 0 
                # indd = torch.argmax(unmasked_sigmas,1) == mapp[kk]
                unmasked_sigmas[indd,mapp[instance_id]] = -5e-5
        elif scene_name == 'scannet0113':
            # for kk in range(K):
            for kk in [0,4,6]:
            # for kk in [0,9]:
                if kk == instance_id:
                    continue
                # indd = unmasked_sigmas[:,mapp[kk]] > 0 
                indd = torch.argmax(unmasked_sigmas,1) == mapp[kk]
                unmasked_sigmas[indd,mapp[instance_id]] = -5e5
        
        unmasked_sigmas = unmasked_sigmas.reshape(N_rays, N_samples_,K)

        # chosen_sigmas = unmasked_sigmas[:,:,instance_id]
        chosen_sigmas = unmasked_sigmas[:,:,mapp[instance_id]]
        # chosen_rgbs = unmasked_rgbs[:,:,instance_id,:]
        chosen_rgbs = unmasked_rgbs[:,:,mapp[instance_id],:]
        return chosen_rgbs, chosen_sigmas
        

    else:
        return rgbs, sigmas



def render_rays_edit_hack(
    models: Dict[str, UDCNeRF],
    embeddings: Dict[str, Any], # dict of xyz & dir
    rays_list: list, # torch.Tensor, 
    obj_instance_ids: list,
    N_samples: int = 64, # 64
    use_disp: bool = False, # false
    perturb: float = 0, # 1
    noise_std: float = 1, # 1
    N_importance: int = 0, # 64
    chunk: int = 1024 * 32, # 32768
    white_back: bool = False, # false,
    scene_name: str = 'toydesk2',
    **dummy_kwargs,
):
    embedding_xyz, embedding_dir = embeddings["xyz"], embeddings["dir"]

    assert len(rays_list) == len(obj_instance_ids)
    # print('debug in render: ', obj_instance_ids, len(obj_instance_ids))

    z_vals_list = []
    xyz_coarse_list = []
    dir_embedded_list = []
    rays_o_list = []
    rays_d_list = []

    for idx, rays in enumerate(rays_list):
        # Decompose the inputs
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
        near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

        # Embed direction
        dir_embedded = embedding_dir(rays_d)  # (N_rays, embed_dir_channels)

        rays_o = rearrange(rays_o, "n1 c -> n1 1 c")
        rays_d = rearrange(rays_d, "n1 c -> n1 1 c")

        rays_o_list += [rays_o]
        rays_d_list += [rays_d]

        # Sample depth points
        z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
        if not use_disp:  # use linear sampling in depth space
            z_vals = near * (1 - z_steps) + far * z_steps
        else:  # use linear sampling in disparity space
            z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

        z_vals = z_vals.expand(N_rays, N_samples)

        xyz_coarse = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")

        # save for each rays batch
        xyz_coarse_list += [xyz_coarse]
        z_vals_list += [z_vals]
        dir_embedded_list += [dir_embedded]

    # inference for each objects
    rgbs_list = []
    sigmas_list = []
    obj_ids_list = []
    for i in range(len(rays_list)):
        rgbs, sigmas = inference_from_model_udc(
            model=models["coarse"],
            embedding_xyz=embedding_xyz,
            dir_embedded=dir_embedded_list[i],
            xyz=xyz_coarse_list[i],
            z_vals=z_vals_list[i],
            chunk=chunk,
            instance_id=obj_instance_ids[i],
            typ='coarse',
            scene_name=scene_name
        )

        rgbs_list += [rgbs]
        sigmas_list += [sigmas]
        obj_ids_list += [torch.ones_like(sigmas) * i]

    results = {}
    volume_rendering_multi(
        results,
        "coarse",
        z_vals_list,
        rgbs_list,
        sigmas_list,
        noise_std,
        white_back,
        obj_ids_list,
    )

    if N_importance > 0:  # sample points for fine model
        rgbs_list = []
        sigmas_list = []
        z_vals_fine_list = []
        for i in range(len(rays_list)):
            z_vals = z_vals_list[i]
            z_vals_mid = 0.5 * (
                z_vals[:, :-1] + z_vals[:, 1:]
            )  # (N_rays, N_samples-1) interval mid points
            # recover weights according to z_vals from results
            weights_ = results["weights_coarse"][results["obj_ids_coarse"] == i]
            assert weights_.numel() == N_rays * N_samples
            weights_ = rearrange(weights_, "(n1 n2) -> n1 n2", n1=N_rays, n2=N_samples)
            z_vals_ = sample_pdf(
                z_vals_mid, weights_[:, 1:-1].detach(), N_importance, det=(perturb == 0)
            )

            z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]

            # if we have ray mask (e.g. bbox), we clip z values
            rays = rays_list[i]

            if rays.shape[1] == 10:
                bbox_mask_near, bbox_mask_far = rays[:, 8:9], rays[:, 9:10]
                z_val_mask = torch.logical_and(
                    z_vals > bbox_mask_near, z_vals < bbox_mask_far
                )
                z_vals[z_val_mask] = bbox_mask_far.repeat(1, z_vals.shape[1])[
                    z_val_mask
                ]

            # combine coarse and fine samples
            z_vals_fine_list += [z_vals]

            xyz_fine = rays_o_list[i] + rays_d_list[i] * rearrange(
                z_vals, "n1 n2 -> n1 n2 1"
            )

            rgbs, sigmas = inference_from_model_udc(
                model=models["fine"],
                embedding_xyz=embedding_xyz,
                dir_embedded=dir_embedded_list[i],
                xyz=xyz_fine,
                z_vals=z_vals_fine_list[i],
                chunk=chunk,
                instance_id=obj_instance_ids[i],
                typ='fine',
                scene_name=scene_name
            )

            rgbs_list += [rgbs]
            sigmas_list += [sigmas]

        volume_rendering_multi(
            results,
            "fine",
            z_vals_fine_list,
            rgbs_list,
            sigmas_list,
            noise_std,
            white_back,
        )
    return results