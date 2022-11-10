import torch

import kornia as K

import dist
from utils.log import getLogger

LOGGER = getLogger(__name__)


def get_meshgrid(resolution, device):
    grid_x, grid_y = torch.meshgrid(torch.arange(resolution[0]).float() / resolution[0],
                                    torch.arange(resolution[1]).float() / resolution[1], indexing='ij')
    grid_x = grid_x.to(device)
    grid_y = grid_y.to(device)
    return grid_x, grid_y


def lstq(A, F_u, F_v, lamda=0.01):
    requires_grad = F_u.requires_grad and F_u.grad_fn is not None # Requires grad and has tape attached
    if requires_grad and torch.__version__ >= '1.13.0' or not requires_grad and torch.__version__ >= '1.10.0':
        # 1.11.0 hopefully introduces grads to lstsq.
        theta_x = torch.linalg.lstsq(A, F_u).solution
        theta_y = torch.linalg.lstsq(A, F_v).solution
        return theta_x, theta_y

    # cols = A.shape[2]
    # assert all(cols == torch.linalg.matrix_rank(A))  # something better?
    try:
        Q, R = torch.linalg.qr(A)
        theta_x = torch.bmm(torch.bmm(torch.linalg.inv(R), Q.transpose(1, 2)), F_u)
        theta_y = torch.bmm(torch.bmm(torch.linalg.inv(R), Q.transpose(1, 2)), F_v)
    except:
        LOGGER.exception("Least Squares failed")
        raise
    return theta_x, theta_y

def affine(masks_softmaxed, flow):
    h, w = flow.shape[-2:]
    flow = flow.clamp(-20, 20)
    masks = dist.clamp_probs(masks_softmaxed, eps=1e-8, dim=1)
    grid_x, grid_y = get_meshgrid((h, w), device=flow.device)
    recon = get_gradient_rec_flow(masks, flow, grid_x, grid_y)
    return torch.nan_to_num(recon.clamp(-20, 20), 0.0, -20, 20)

def get_gradient_rec_flow(masks_softmaxed, flow, grid_x, grid_y):
    rec_flow = 0
    for k in range(masks_softmaxed.size(1)):
        mask = masks_softmaxed[:, k].unsqueeze(1)
        _F = flow * mask
        M = mask.flatten(1)
        bs = _F.shape[0]
        x = grid_x.unsqueeze(0).flatten(1) * M
        y = grid_y.unsqueeze(0).flatten(1) * M

        F_u = _F[:, 0].flatten(1).unsqueeze(2)  # B x L x 1
        F_v = _F[:, 1].flatten(1).unsqueeze(2)  # B x L x 1
        A = torch.stack([x, y, torch.ones_like(y) * M], 2)  # B x L x 2

        theta_x, theta_y = lstq(A, F_u, F_v, lamda=.01)

        rec_flow_m = torch.stack([torch.einsum('bln,bnk->blk', A, theta_x).view(bs, *grid_x.shape),
                                    torch.einsum('bln,bnk->blk', A, theta_y).view(bs, *grid_y.shape)], 1)

        rec_flow += rec_flow_m
    return rec_flow
