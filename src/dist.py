import functools
import math

import einops
import torch
import torch.distributions
import torch.distributions as D
import torch.distributions.utils as dist_utils
from tqdm.auto import tqdm

import utils

LOGGER = utils.log.getLogger(__name__)

__defined_kl = False

EPS = 1e-5


def to_simplex(probs, dim=-1):
    return probs / probs.sum(dim, keepdim=True)  # to simplex


def clamp_probs(probs, eps=EPS, dim=-1):
    probs = probs.clamp(eps, 1. - eps)  # Will no longer sum to 1
    return to_simplex(probs, dim=dim)


class ClampedRelaxedOneHotCategorical(D.RelaxedOneHotCategorical):
    """Numerically more stable version of Concrete distribution"""

    def __init__(self, *args, probs=None, logits=None, **kwargs):
        super(ClampedRelaxedOneHotCategorical, self).__init__(*args, logits=norm_clamp_logits_probs(probs, logits),
                                                              **kwargs)

    def rsample(self, sample_shape=torch.Size()):
        """
        If logits are large enought and gumbels also happen to be large, this might saturate exponential and sample will
        lie outside of the support (i.e. =1.). This adds clamping/normalisation to prevent that.
        """
        s = super(ClampedRelaxedOneHotCategorical, self).rsample(sample_shape)
        return clamp_probs(s)

    def log_prob(self, value):
        return super(ClampedRelaxedOneHotCategorical, self).log_prob(clamp_probs(value))


def mask_dist(logits, temp=1.):
    numeric_report(logits=logits)

    indicator = torch.ones(*logits.shape[:-3], dtype=logits.dtype, device=logits.device)
    indicator = indicator[:, :, None, None, None]

    p_or_l = {'logits': logits}
    LOGGER.debug_once(f'Logits {logits.shape} indicator {indicator.shape}')
    q_mask_cat = D.OneHotCategorical(**norm_clamp_logits_probs_kw(**p_or_l))
    p_mask_cat = D.OneHotCategorical(probs=torch.ones_like(logits) / indicator.sum(-1, keepdim=True) * indicator)

    q_mask = ClampedRelaxedOneHotCategorical(torch.tensor(temp, dtype=logits.dtype, device=logits.device),
                                                **p_or_l)
    q_mask.sample = q_mask.rsample
   
    return q_mask, q_mask_cat, p_mask_cat


def grid(h, w, pad=0, device='cpu', dtype=torch.float32, norm=False):
    hr = torch.arange(h + 2 * pad, device=device) - pad
    wr = torch.arange(w + 2 * pad, device=device) - pad
    if norm:
        hr = hr / (h + 2 * pad - 1)
        wr = wr / (w + 2 * pad - 1)
    ig, jg = torch.meshgrid(hr, wr)
    g = torch.stack([jg, ig]).to(dtype)[None]
    return g


@functools.lru_cache(maxsize=1, typed=False)
def cached_grid(h, w, pad=0, device='cpu', dtype=torch.float32, norm=False):
    return grid(h, w, pad, device, dtype, norm)


@functools.lru_cache(2)
def __cached_inv_and_logdet(matrix, device='cpu'):
    if not torch.is_tensor(matrix):
        matrix = torch.tensor(matrix)

    # Inverse in double precision
    matrix = matrix.to(device).double()
    matrix_inv = torch.linalg.inv(matrix)
    return matrix_inv, matrix_inv.logdet()


def __det3x3(t):
    r = t[..., 0, 0] * t[..., 1, 1] * t[..., 2, 2] \
        + t[..., 0, 1] * t[..., 1, 2] * t[..., 2, 0] \
        + t[..., 0, 2] * t[..., 1, 0] * t[..., 2, 1] \
        - t[..., 0, 2] * t[..., 1, 1] * t[..., 2, 0] \
        - t[..., 0, 1] * t[..., 1, 0] * t[..., 2, 2] \
        - t[..., 0, 0] * t[..., 1, 2] * t[..., 2, 1]
    return r


def __inv3x3(t):
    t11 = t[..., 0, 0]
    t12 = t[..., 0, 1]
    t13 = t[..., 0, 2]
    t21 = t[..., 1, 0]
    t22 = t[..., 1, 1]
    t23 = t[..., 1, 2]
    t31 = t[..., 2, 0]
    t32 = t[..., 2, 1]
    t33 = t[..., 2, 2]

    det = __det3x3(t)
    # adjugate
    a11 = (t22 * t33 - t23 * t32) / det
    a12 = -(t12 * t33 - t13 * t32) / det
    a13 = (t12 * t23 - t13 * t22) / det

    a21 = -(t21 * t33 - t21 * t31) / det
    a22 = (t11 * t33 - t13 * t31) / det
    a23 = -(t11 * t23 - t13 * t21) / det

    a31 = (t21 * t32 - t22 * t31) / det
    a32 = -(t11 * t32 - t12 * t31) / det
    a33 = (t11 * t22 - t12 * t21) / det
    return torch.stack([a11, a12, a13, a21, a22, a23, a31, a32, a33], -1).view(*t.shape)


def affine_gaussian_linear_consitency_log_marginal(masks, flows, cov, sigma2,
                                                   gamma=1.0,
                                                   origin=None,
                                                   detach=False,
                                                   means=None,
                                                   double=True,
                                                   no_reduce=False,
                                                   norm=False,
                                                   flow_norm=False):
    LOGGER.debug_once(f'Affine log_prob masks {masks.shape} flows {flows.shape} origin {origin}')
    if gamma != 1.0:
        LOGGER.debug_once(f'Gamma {gamma}')

    gamma0 = gamma

    assert len(flows.shape) == 4
    b, c, H, W = flows.shape
    K = masks.shape[1]
    f = flows.view(b, c, 1, H * W)
    dtype = flows.dtype
    etype = torch.double if double else dtype
    if flow_norm:
        f = f / torch.tensor([W, H], dtype=flows.dtype, device=flows.device).view(1, 2, 1, 1)

    assert list(masks.shape[-2:]) == [H, W]
    assert masks.shape[0] == b
    m = masks.flatten(2)[:, None]  # B x 1 x K x HW

    g = cached_grid(H, W, device=flows.device, dtype=dtype, norm=norm).view(1, 2, 1, H * W)  # 1 x 2 x 1 x H*W
    # x = g[:, :1] / (W - 1)  # 1 x 1 x 1 x HW
    # y = g[:, 1:] / (H - 1)  # 1 x 1 x 1 x HW

    x = g[:, :1] / 1.  # 1 x 1 x 1 x HW
    y = g[:, 1:] / 1.  # 1 x 1 x 1 x HW


    if origin == 'centroid_fix':
        denom = m.sum(-1, keepdim=True) + 1e-6
        cx = (x * m).sum(-1, keepdim=True) / denom  # B x 1 x K x 1
        cy = (y * m).sum(-1, keepdim=True) / denom  # B x 1 x K x 1

        x = x - cx  # B x 1 x K x HW
        y = y - cy  # B x 1 x K x HW
    elif origin == 'frame':
        x = x - 0.5
        y = y - 0.5
    else:
        raise ValueError(f'Unknown origin {origin}')

    if detach:
        x = x.detach()
        y = y.detach()

    # Run loss in double precision
    x = x.to(etype)
    y = y.to(etype)
    m = m.to(etype)
    f = f.to(etype)

    assert cov.shape[-1] == cov.shape[-2]
    _, D = cov.shape

    cov_inv, cov_inv_logdet = __cached_inv_and_logdet(cov, device=flows.device)
    cov_inv_logdet = cov_inv_logdet.to(etype)
    cov_inv = cov_inv.to(etype)

    # The meaning of A/B/C/D and alpha/beta/gamma/delta is swapped compared to the paper.

    A = cov_inv[:3, :3]
    B = cov_inv[:3, 3:]
    C = cov_inv[3:, :3]
    D = cov_inv[3:, 3:]

    xTx = (x * x * m).sum(-1)  # B x 1 x K
    xTy = (x * y * m).sum(-1)  # B x 1 x K
    yTy = (y * y * m).sum(-1)  # B x 1 x K
    xT1 = (x * m).sum(-1)  # B x 1 x K
    yT1 = (y * m).sum(-1)  # B x 1 x K

    # n = 1T1
    n = m.sum(-1)  # B x 1 x K

    GTG = torch.stack([xTx, xTy, xT1, xTy, yTy, yT1, xT1, yT1, n], -1).view(b, 1, K, 3, 3) / sigma2

    # S diag
    A_add_GTG = A[None, None, None] + GTG
    D_add_GTG = D[None, None, None] + GTG

    invA_add_GTG = __inv3x3(A_add_GTG)

    C_invA_add_GTG = torch.einsum('xy,bckyz->bckxz', C, invA_add_GTG)
    invA_add_GTG_B = torch.einsum('bckxy, yz->bckxz', invA_add_GTG, B)
    shur = D_add_GTG - torch.einsum('xy,bckyz,zw->bckxw', B, invA_add_GTG, C)

    detS = __det3x3(A_add_GTG)*__det3x3(shur)

    delta = __inv3x3(shur)  # B x 1 x K x 3 x 3
    beta = -torch.einsum('bckxy,bckyz->bckxz', invA_add_GTG_B, delta)
    gamma = -torch.einsum('bckxy,bckyz->bckxz', delta, C_invA_add_GTG)
    alpha = invA_add_GTG + torch.einsum('bckxy,bckyz,bckzw->bckxw', invA_add_GTG_B, delta, C_invA_add_GTG)

    if means is not None:
        LOGGER.debug_once(f'Using non-zero mean: {means}')
        m1,m2,m3,m4,m5,m6 = means
        mx = (m1 - 1)*x + m2*y + m3  # B x 1 x K x HW
        my = m4*x + (m5 - 1)*y + m6  # B x 1 x K x HW
        mean = torch.cat([mx, my], 1)
        f = f - mean

    fTx = (f * x * m).sum(-1)  # B x C x K
    fTy = (f * y * m).sum(-1)  # B x C x K
    fT1 = (f * m).sum(-1)  # B x C x K

    uTx = fTx[:, :1, :] # B x 1 x K
    vTx = fTx[:, 1:, :] # B x 1 x K
    uTy = fTy[:, :1, :] # B x 1 x K
    vTy = fTy[:, 1:, :]  # B x 1 x K
    uT1 = fT1[:, :1, :]  # B x 1 x K
    vT1 = fT1[:, 1:, :]  # B x 1 x K

    prod_fn = lambda ax, ay, a1, t, bx, by, b1: (ax * (t[..., 0, 0] * bx + t[..., 0, 1] * by + t[..., 0, 2] * b1) +
                                                 ay * (t[..., 1, 0] * bx + t[..., 1, 1] * by + t[..., 1, 2] * b1) +
                                                 a1 * (t[..., 2, 0] * bx + t[..., 2, 1] * by + t[..., 2, 2] * b1))

    FTP_Sinv_PTF = (prod_fn(uTx, uTy, uT1, alpha, uTx, uTy, uT1) +
                    prod_fn(vTx, vTy, vT1, gamma, uTx, uTy, uT1) +
                    prod_fn(uTx, uTy, uT1, beta, vTx, vTy, vT1) +
                    prod_fn(vTx, vTy, vT1, delta, vTx, vTy, vT1))

    LOGGER.debug_once(f"FTP_Sinv_PTF: {FTP_Sinv_PTF.shape}")

    if torch.any(detS <= 0.0):
        LOGGER.critical("*** DetS is negative *** -- Clamping for numerical stability")
        detS = detS.clamp(min=1e-6)
    logdetS = detS.log()

    numeric_report(logdetS=logdetS, FTP_Sinv_PTF=FTP_Sinv_PTF)

    if means is not None or no_reduce:
        FTF = (f * f * m).sum([1, -1])
        ll = -0.5 * ((c * n.squeeze(1) * math.log(2 * math.pi * sigma2)
                     + logdetS.squeeze(1)
                     - cov_inv_logdet.view(1, 1)) * gamma0
                     + FTF / sigma2
                     - FTP_Sinv_PTF.squeeze(1) / sigma2 ** 2)
    else:
        FTF = (f * f).sum([-1, -2, -3])  # B
        ll = -0.5 * ((c * H * W * math.log(2 * math.pi * sigma2)
                     + logdetS.sum(-1).squeeze(1)
                     - K * cov_inv_logdet) * gamma0
                     + FTF / sigma2
                     - FTP_Sinv_PTF.sum(-1).squeeze(1) / sigma2 ** 2)
        ll = ll[..., None]

    return ll.to(dtype)  # B x C


def translation_gaussian_linear_consitency_log_marginal(masks, flows, tau2, sigma2):
    LOGGER.debug_once(f'Trans log_prob masks {masks.shape} flows {flows.shape}')
    assert len(flows.shape) == 4
    B, C, H, W = flows.shape
    K = masks.shape[1]
    masks = masks.view(B, K, 1, H, W)
    f = flows.view(B, C, 1, H * W)
    nu = sigma2 / tau2

    m = masks.flatten(2)[:, None]  # B x 1 x K x HW
    n = m.sum(-1, keepdim=True).expand(-1, C, -1, -1)  # B x C x K x 1

    means = (f * m).sum(-1, keepdim=True) / (n + 1e-6)
    w = 1 - (nu / (n + nu)).sqrt()
    numeric_report(w=w, masks=masks, means=means, flows=flows)

    # This does not appear to zero out the gradient as expected
    # means = torch.where(n > 0, means, torch.tensor(0., dtype=flows.dtype, device=flows.device))

    w_means = w * means

    
    det_term = (n + nu).log() - math.log(nu)


    loc = (w_means * m).sum(-2, keepdim=True)  # B x C x 1 x HW
    ll = D.Normal(loc=loc, scale=sigma2).log_prob(f).sum(-1)  # B x C x 1
    det_term = det_term.sum(-2)  # B x C x 1

    return ll + det_term



def soft_clamp10(x: torch.Tensor):
    """Soft differentiable clamping between [-10, 10]"""
    return 10. * torch.tanh(x.clamp(-20, 20) / 10.)


def norm_clamp_logits_probs(probs=None, logits=None, dim=-1):
    if (probs is None) == (logits is None):
        raise ValueError("Either `probs` or `logits` must be specified, but not both.")
    if probs is not None:
        logits = dist_utils.probs_to_logits(probs)
    if numeric_report(pre_clamp_logits=logits):
        LOGGER.critical(f"*** LOGITS in NORM fuction are funny; Setting ***")
        logits = torch.nan_to_num(logits, nan=0., posinf=11., neginf=-11.)
    logits = logits - logits.logsumexp(dim=dim, keepdim=True)  # Normalise
    logits = soft_clamp10(logits)  # clamp
    return logits


def norm_clamp_logits_probs_kw(*args, **kwargs):
    return {'logits': norm_clamp_logits_probs(*args, **kwargs)}


def numeric_report(**kw_tensors):
    funny = False
    for k, t in kw_tensors.items():
        if t is None:
            continue
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        if torch.isnan(t).any():
            LOGGER.error(f"{k} is nan")
            funny = True
        if torch.isinf(t).any():
            LOGGER.error(f"{k} is inf")
            funny = True
    return funny


def kl_cat(q_probs, p_probs):
    q_probs = clamp_probs(q_probs)
    p_probs = clamp_probs(p_probs)
    return D.kl_divergence(D.OneHotCategorical(q_probs), D.OneHotCategorical(p_probs))

def sym_kl_cat(logits1, logits2, dim=1):
    logits1 = norm_clamp_logits_probs(logits=logits1, dim=dim)
    logits2 = norm_clamp_logits_probs(logits=logits2, dim=dim)
    probs1 = logits1.softmax(dim=dim)
    probs2 = logits2.softmax(dim=dim)

    t1 = probs1 * (logits1 - logits2)
    t1[(probs1 == 0).expand_as(t1)] = float('inf')
    t1[(probs2 == 0).expand_as(t1)] = 0
    t2 = probs2 * (logits2 - logits1)
    t2[(probs2 == 0).expand_as(t2)] = float('inf')
    t2[(probs1 == 0).expand_as(t2)] = 0

    return t1.sum(dim) * 0.5 + t2.sum(dim) * 0.5


def logit_model(model_str, outputs, gamma=1.0):
    LOGGER.debug_once(f'logit model is {model_str}')
    logits = outputs['sem_seg']
    logits = logits.view(*logits.shape[:2], 1, *logits.shape[-2:])
    assert model_str == 'id'
    return id_model(logits)

def id_model(logits):
    is_training = logits.requires_grad and logits.grad_fn is not None
    if is_training:
        return logits, {}
    return logits
