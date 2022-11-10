import functools

import numpy as np
import torch
import torch.nn.functional as F

import dist
import covariance
import utils
from utils.schedules import Schedule

LOGGER = utils.log.getLogger(__name__)

def criterion(cfg, 
        preds_dict, 
        flow, 
        iteration=0, 
        beta_mult=1.0, 
        total=10 ** 6,
        extra_flow=tuple(), 
        disp_mask=None):

    masks = preds_dict['sem_seg'] if 'sem_seg' in preds_dict else preds_dict['_masks']

    dist.numeric_report(masks=masks, 
                        flow=flow
                        )

    n = flow.shape[0]

    num_samples = int(cfg.UNSUPVIDSEG.LOSS_NPART)

    loss_str = cfg.UNSUPVIDSEG.LOSS

    temp = get_maybe_schedule(cfg, 'UNSUPVIDSEG.LOSS_TEMP', iteration, total)
    beta = get_maybe_schedule(cfg, 'UNSUPVIDSEG.LOSS_BETA', iteration, total) * beta_mult

    sigma2 = get_maybe_schedule(cfg, 'UNSUPVIDSEG.LOSS_SIGMA2', iteration, total)

    LOGGER.info_once(f"Loss is {loss_str}")

    log_dict = {
        'beta': torch.tensor(beta),
        'temp': torch.tensor(temp),
        'sigma2': torch.tensor(sigma2),
    }

    if loss_str.startswith('ELBO'):
        r = dist.logit_model(
            'id',
            preds_dict
        )
        if isinstance(r, tuple):
            masks_raw, logit_dict = r
        else:
            masks_raw = r
            logit_dict = {}

        m_til_logits = masks_raw.permute(0, 2, 3, 4, 1)  # [B, 1, H, W, K] -- event shape last
        sample_shape = dict(sample_shape=[num_samples])

        q_mask, q_mask_cat, p_mask_cat = dist.mask_dist(m_til_logits,
                                                        temp=temp)

        mask_sample = q_mask.sample(**sample_shape)

        dist.numeric_report(mask_sample=mask_sample)
        assert torch.all(mask_sample.min() >= 0.0), "Mask is < 0.0"
        assert torch.all(mask_sample.max() <= 1.0), "Mask is > 1.0"

        LOGGER.debug_once(f"MASK SAMPLE SHAPE {mask_sample.shape}")

        if disp_mask is not None:
            _disp_mask = disp_mask[None, ..., None]
            LOGGER.debug_once(f"DISP MASK SHAPE {disp_mask.shape} -> {_disp_mask.shape}")
            mask_sample = mask_sample * _disp_mask

        mask_ex = mask_sample.view(n * num_samples, *mask_sample.shape[2:]).permute(0, 4, 1, 2, 3)

        # (B * num_samples) x K x C(1) x H x W

        kl = dist.D.kl_divergence(q_mask_cat, p_mask_cat).sum([-1, -2, -3])
        LOGGER.debug_once(f"KL uniform {kl.shape}")
        kl = kl * beta
        dist.numeric_report(kl=kl)

        log_dict['loss_kl_divergence'] = kl.mean()

        LOGGER.debug_once(f"KL {kl.shape}")
        flows = [flow, *extra_flow]
        assert len(flows) >= 1
        LOGGER.debug_once(f"Evaluating loss for {len(flows)} flows")
        nll = 0.
        for fi, flo in enumerate(flows):
            flow_ex = flo[None].expand(num_samples, *[-1] * len(flo.shape)).reshape(n * num_samples,
                                                                                    *flow.shape[1:])
            nll_flo = neg_log_like(loss_str, flow_ex, mask_ex, cfg, n, iteration, total) / len(flows)
            log_dict[f'loss_flow_nll_{fi}'] = nll_flo.mean(0).mean(0)
            nll = nll + nll_flo

        nll = nll.mean(0)
        LOGGER.debug_once(f"nll loss {nll.shape}")
        log_dict['loss_flow_nll'] = nll.mean(0)
        loss = (nll + kl).mean(0)

    log_dict['loss_total_nll'] = nll.mean(0)
    log_dict['loss_total'] = loss

    if dist.numeric_report(loss_final=loss):
        import ipdb;
        ipdb.set_trace()
    LOGGER.debug_once(f"LOSS {loss.shape} NLL {nll.shape}")
    return loss, log_dict


def neg_log_like(loss_str, flow_ex, mask_ex, cfg, n, iteration, total, samples=None):
    num_samples = samples or int(cfg.UNSUPVIDSEG.LOSS_NPART)

    sigma2 = get_maybe_schedule(cfg, 'UNSUPVIDSEG.LOSS_SIGMA2', iteration, total)

    if loss_str == 'ELBO_AFF_FULL':
        means, cov = covariance.resolve(cfg)
        cov = torch.tensor(cov)
        # Affine NLL
        LOGGER.debug_once(f"Covariance:\n{np.array2string(cov.cpu().numpy(), max_line_width=120, edgeitems=6, floatmode='unique')}")
        log_p_f_ks_ex = dist.affine_gaussian_linear_consitency_log_marginal(mask_ex,
                                                                            flow_ex,
                                                                            cov,
                                                                            sigma2,
                                                                            origin=cfg.UNSUPVIDSEG.LOSS_ORIGIN,
                                                                            detach=cfg.UNSUPVIDSEG.LOSS_GRID_DETACH,
                                                                            means=means if cfg.UNSUPVIDSEG.LOSS_MEANS else None,
                                                                            )
        log_p_f_ks = log_p_f_ks_ex.view(num_samples, n, *log_p_f_ks_ex.shape[1:])
        LOGGER.debug_once(f"Ref eff log p(f|m) {log_p_f_ks_ex.shape} -> {log_p_f_ks.shape}")
        nll = -log_p_f_ks.sum(-1)

    elif loss_str == 'ELBO_TRANS':
        means, cov = covariance.resolve(cfg)
        cov = torch.tensor(cov)
        # Affine NLL
        LOGGER.debug_once(f"Covariance:\n{np.array2string(cov.cpu().numpy(), max_line_width=120, edgeitems=6, floatmode='unique')}")
        var = cov.diag().view(-1)[[2,5]].max().item()
        if cfg.UNSUPVIDSEG.LOSS_MEANS and means is not None:
            means = torch.tensor(means).to(flow_ex.device).to(flow_ex.dtype).view(-1)[[2,5]].view(1,2,1,1)
            flow_ex = flow_ex - means
        log_p_f_ks_ex = dist.translation_gaussian_linear_consitency_log_marginal(mask_ex, flow_ex, var, sigma2)
        log_p_f_ks = log_p_f_ks_ex.view(num_samples, n, *log_p_f_ks_ex.shape[1:])
        LOGGER.debug_once(f"Ref eff log p(f|m) {log_p_f_ks_ex.shape} -> {log_p_f_ks.shape}")
        nll = -log_p_f_ks.sum([-1, -2])
    else:
        raise ValueError(f"Unknown loss {loss_str}")

    LOGGER.debug_once(f"NLL Shape {nll.shape}")
    dist.numeric_report(nll=nll)
    return nll


def get_maybe_schedule(cfg, name, iteration, total):
    # TODO: use one from utils
    sch = Schedule.build(functools.reduce(lambda x, n: getattr(x, n), name.split('.'), cfg))
    LOGGER.debug_once(f"Schedule for {name} {sch}")
    return sch.value(iteration, total)


def warp_fwd(target, bwd_next_frame_flow):
    H, W = bwd_next_frame_flow.shape[-2:]
    sg = dist.cached_grid(H,W, norm=False, device=target.device, dtype=target.dtype) - bwd_next_frame_flow
    sg = sg * 2 / torch.tensor([W-1, H-1], device=target.device, dtype=target.dtype).view(1, 2, 1, 1) -1
    sg = sg.permute(0, 2, 3, 1)
    warped = torch.nn.functional.grid_sample(target, sg, align_corners=False)
    return warped


def warp_bwd(target, fwd_prev_prame_flow):
    H, W = fwd_prev_prame_flow.shape[-2:]
    sg = dist.cached_grid(H,W, norm=False, device=target.device, dtype=target.dtype) + fwd_prev_prame_flow
    sg = sg * 2 / torch.tensor([W-1, H-1], device=target.device, dtype=target.dtype).view(1, 2, 1, 1) -1
    sg = sg.permute(0, 2, 3, 1)
    warped = torch.nn.functional.grid_sample(target, sg, align_corners=False)
    return warped

def get_weight(rgb_prev, rgb_next, k=1):
    """
    Computes disparity weight for each pixel.
    """
    # Compute disparity
    disp = torch.pow(torch.abs(rgb_prev - rgb_next), k).sum(1)
    # Compute normalized disparity weight
    disp = disp - disp.flatten(-2).min(-1, keepdim=True)[0][..., None]
    disp = disp / (disp.flatten(-2).max(-1, keepdim=True)[0].clamp(1e-06)[..., None])
    return 1. - disp


@torch.no_grad()
def photometric_warp_weights(rgb_pair, fwd_flow_pair, bwd_flow_pair, k=1):
    b,t,c,h,w = rgb_pair.shape

    assert b == fwd_flow_pair.shape[0] == bwd_flow_pair.shape[0]
    assert t == fwd_flow_pair.shape[1] == bwd_flow_pair.shape[1] == 2

    rgb_prev = rgb_pair[:, 0]
    rgb_next = rgb_pair[:, 1]
    rgb_prev_warped = warp_fwd(rgb_prev, bwd_flow_pair[:, 1])
    rgb_next_warped = warp_bwd(rgb_next, fwd_flow_pair[:, 0])

    w_next = get_weight(rgb_next, rgb_prev_warped, k)
    w_prev = get_weight(rgb_prev, rgb_next_warped, k)
    LOGGER.debug_once(f"disparity weights n {w_next.shape} p {w_prev.shape}")
    return w_next, w_prev


def warp_pair_loss(logits_pair, fwd_flow_pair, bwd_flow_pair, rgb_pair=None):
    logits_prev = logits_pair[:, 0]
    logits_next = logits_pair[:, 1]
    logits_prev_warped = warp_fwd(logits_prev, bwd_flow_pair[:, 1])
    logits_next_warped = warp_bwd(logits_next, fwd_flow_pair[:, 0])
    if rgb_pair is not None:
        w_next, w_prev = photometric_warp_weights(rgb_pair, fwd_flow_pair, bwd_flow_pair, k=1)
        LOGGER.debug_once(f"disparity weights n {w_next.shape} p {w_prev.shape}")
    else:
        w_next = torch.ones((logits_prev.shape[0], *logits_prev.shape[2:]), device=logits_prev.device, dtype=logits_prev.dtype)
        w_prev = torch.ones((logits_prev.shape[0], *logits_prev.shape[2:]), device=logits_prev.device, dtype=logits_prev.dtype)

    kl_fwd = dist.sym_kl_cat(logits_prev_warped, logits_next)
    kl_bwd = dist.sym_kl_cat(logits_next_warped, logits_prev)

    LOGGER.debug_once(f"equiv kl fwd {kl_fwd.shape} bwd {kl_bwd.shape}")
    return kl_fwd * w_next * 0.5 + kl_bwd * w_prev * 0.5

def warp_loss(logits, fwd_flow, bwd_flow, rgbs):
    B, t, K, H, W = logits.shape
    assert list(fwd_flow.shape) == [B, t, 2, H, W]
    assert list(bwd_flow.shape) == [B, t, 2, H, W]
    loss = []
    for i in range(0, t-1):
        l = warp_pair_loss(
            logits[:, [i, i+1]],
            fwd_flow[:, [i, i+1]],
            bwd_flow[:, [i, i+1]],
            rgbs[:, [i, i+1]]
        ).sum([-1,-2])
        loss.append(l)
    loss = torch.stack(loss, dim=-1).mean(-1)
    dist.numeric_report(equiv_loss=loss)
    return loss


def warp_equiv_loss(cfg, logits, rgbs, fwd_flow, bwd_flow, iteration, total):
    loss = torch.tensor(0., device=logits.device, dtype=logits.dtype)
    mult = get_maybe_schedule(cfg, 'UNSUPVIDSEG.LOSS_EQUIV', iteration, total)
    log_dict = {
        'equiv': torch.tensor(mult)
    }
    if mult > 0:
        LOGGER.debug_once(f'Equiv loss logits {logits.shape} rgbs {rgbs.shape} fwd {fwd_flow.shape} bwd {bwd_flow.shape}')
        if len(logits.shape) == 4:
            t = cfg.INPUT.SAMPLING_FRAME_NUM
            _logits = logits.view(-1, t, *logits.shape[1:])
            LOGGER.debug_once(f'Equiv expanding logits {logits.shape} -> {_logits.shape}')
            logits = _logits
        if len(rgbs.shape) == 4:
            t = cfg.INPUT.SAMPLING_FRAME_NUM
            _rgbs = logits.view(-1, t, *rgbs.shape[1:])
            LOGGER.debug_once(f'Equiv expanding rgbs {rgbs.shape} -> {_rgbs.shape}')
            rgbs = _rgbs
        if len(fwd_flow.shape) == 4:
            t = cfg.INPUT.SAMPLING_FRAME_NUM
            _fwd_flow = fwd_flow.view(-1, t, *fwd_flow.shape[1:])
            LOGGER.debug_once(f'Equiv expanding fwd_flow {fwd_flow.shape} -> {_fwd_flow.shape}')
            fwd_flow = _fwd_flow
        if len(bwd_flow.shape) == 4:
            t = cfg.INPUT.SAMPLING_FRAME_NUM
            _bwd_flow = bwd_flow.view(-1, t, *bwd_flow.shape[1:])
            LOGGER.debug_once(f'Equiv expanding bwd_flow {bwd_flow.shape} -> {_bwd_flow.shape}')
            bwd_flow = _bwd_flow
        loss = warp_loss(logits, fwd_flow, bwd_flow, rgbs).mean(0) * mult
        LOGGER.info_once(f'Equiv loss shape {loss.shape}')
    return loss, log_dict