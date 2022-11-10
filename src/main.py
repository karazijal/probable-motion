import determinism  # noqa

determinism.do_not_delete()  # noqa

import argparse
import os
import sys
import time
from argparse import ArgumentParser
import gc

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import PeriodicCheckpointer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
import config
import losses
import val
from mask2former_trainer_video import setup, Trainer
# from unet import get_unet


# @formatter:on

logger = utils.log.getLogger('unsup_vidseg')


torch.multiprocessing.set_sharing_strategy('file_system')


def train_step(cfg, model, optimizer, scheduler, sample, iteration, total_iter):
    sample_dict = utils.convert.list_of_dicts_2_dict_of_tensors(sample, device=model.device)
    logger.debug_once(f'Train inputs: {[(k, utils.get_shape(v)) for k, v in sample_dict.items()]}')

    preds = model.forward_base(sample, keys=cfg.UNSUPVIDSEG.SAMPLE_KEYS, get_eval=True)

    preds_dict = utils.convert.list_of_dicts_2_dict_of_tensors(preds, device=model.device)
    logger.debug_once(f'Train outputs: {[(k, utils.get_shape(v)) for k,v in preds_dict.items()]}')

    fwd_flow = sample_dict['flow'].clip(-cfg.UNSUPVIDSEG.FLOW_LIM, cfg.UNSUPVIDSEG.FLOW_LIM)
    bwd_flow = None
    extra_flow = []
    disp_mask = None

    flow = None
    if cfg.INPUT.SAMPLING_FRAME_NUM > 1 and cfg.UNSUPVIDSEG.LOSS_DISP_THRESH >= 0:
        bwd_flow = torch.stack([s['flow_2'] for s in sample]).clip(-cfg.UNSUPVIDSEG.FLOW_LIM, cfg.UNSUPVIDSEG.FLOW_LIM).to(model.device)
        _flow = torch.stack([fwd_flow[:, 0], bwd_flow[:, -1]], dim=1)
        logger.debug_once(f'Using flow pair mode: {fwd_flow.shape} {bwd_flow.shape} {_flow.shape}')
        flow = _flow

        rgb_pair = torch.stack([s['rgb'] for s in sample]).to(model.device)
        w_prev, w_next = losses.photometric_warp_weights(rgb_pair, fwd_flow, bwd_flow, k=2)
        disp_mask = torch.stack([w_prev, w_next], dim=1)
        _disp_mask = disp_mask.view(-1, 1, *disp_mask.shape[-2:])
        logger.debug_once(f'Disp weights shape: {disp_mask.shape} -> {_disp_mask.shape}')
        disp_mask = _disp_mask >= cfg.UNSUPVIDSEG.LOSS_DISP_THRESH

    elif cfg.UNSUPVIDSEG.LOSS_MULTI_FLOW:
        logger.debug_once(f'Using MultiFlow mode')
        bwd_flow = torch.stack([s['flow_2'] for s in sample]).clip(-cfg.UNSUPVIDSEG.FLOW_LIM, cfg.UNSUPVIDSEG.FLOW_LIM).to(model.device)
        extra_flow = [bwd_flow]
        flow = fwd_flow
    else:
        logger.debug_once(f'Using single flow mode')
        flow = fwd_flow


    if len(flow.shape) == 5:
        b, t, c, h, w = flow.shape
        flow = flow.view(b * t, c, h, w)

    _extra_flow = []
    for flo in extra_flow:
        if len(flo.shape) == 5:
            b, t, c, h, w = flo.shape
            flo = flo.view(b * t, c, h, w)
        _extra_flow.append(flo)
    extra_flow = _extra_flow

    lflow = flow
    if cfg.UNSUPVIDSEG.LOSS_FLOW_KEY != 'flow':
        lflow = sample_dict[cfg.UNSUPVIDSEG.LOSS_FLOW_KEY].clip(-cfg.UNSUPVIDSEG.FLOW_LIM, cfg.UNSUPVIDSEG.FLOW_LIM)
        if len(lflow.shape) == 5:
            b, t, c, h, w = lflow.shape
            lflow = lflow.view(b * t, c, h, w)

    # TODO: change this to criterion like formulation
    loss, log_dict = losses.criterion(cfg, preds_dict, lflow,
                                        iteration=iteration,
                                        total=total_iter,
                                        extra_flow=extra_flow,
                                        disp_mask=disp_mask)

    if cfg.INPUT.SAMPLING_FRAME_NUM > 1:
        if bwd_flow is None:
            bwd_flow = torch.stack([s['flow_2'] for s in sample]).clamp(-20, 20).to(model.device)
        equiv_loss, eq_log_dict = losses.warp_equiv_loss(cfg, preds_dict['sem_seg'], sample_dict['rgb'], fwd_flow, bwd_flow, iteration, total_iter)
        log_dict.update(eq_log_dict)
        log_dict['loss_equiv'] = equiv_loss.mean()
        del eq_log_dict
        loss = loss + equiv_loss

    train_log_dict = {f'train/{k}': v.item() for k, v in log_dict.items()}
    train_log_dict['train/learning_rate'] = optimizer.param_groups[-1]['lr']

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item(), train_log_dict


def train_vis(cfg, model, sample):
    epreds = model.forward_base(sample, keys=cfg.UNSUPVIDSEG.SAMPLE_KEYS, get_eval=True)
    epreds_dict = utils.convert.list_of_dicts_2_dict_of_tensors(epreds, device=model.device)
    sample_dict = utils.convert.list_of_dicts_2_dict_of_tensors(sample, device=model.device)
    if cfg.INPUT.SAMPLING_FRAME_NUM > 1:
        sample_dict = utils.convert.to_batchxtime(sample_dict)
    masks_softmaxed, pred_masks, true_masks = val.get_masks(cfg, epreds_dict, sample_dict)
    vis = utils.visualisation.Visualiser(cfg)
    vis.add_all(sample_dict, epreds_dict, masks_softmaxed, pred_masks, true_masks)
    imgs = [vis.img_vis()]
    return imgs

def main(args):
    cfg = setup(args)
    logger.info(f"Called as {' '.join(sys.argv)}")
    logger.info(f'Output dir {cfg.OUTPUT_DIR}')

    random_state = utils.random_state.PytorchRNGState(seed=cfg.SEED).to(torch.device(cfg.MODEL.DEVICE))
    random_state.seed_everything()
    utils.log.checkpoint_code(cfg.OUTPUT_DIR)

    writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)

    # initialize modelF
    if cfg.MODEL.META_ARCHITECTURE == 'UNET':
        model, optimizer = get_unet(cfg)
    else:
        model = Trainer.build_model(cfg)
        print(model.sem_seg_head.predictor.whole_seq)
        model.sem_seg_head.predictor.whole_seq = cfg.EVAL_WHOLE_SEQ

        logger.info('Checking backbone trainability')
        if hasattr(model, 'backbone'):
            for n, p in model.backbone.named_parameters():
                if not p.requires_grad:
                    logger.warning(f'{n} is not trainable in backbone')
        else:
            logger.warning('model.backbone not found')

        optimizer = Trainer.build_optimizer(cfg, model)
    scheduler = Trainer.build_lr_scheduler(cfg, optimizer)
    scheduler.max_iters = cfg.SOLVER.MAX_ITER  # Reset if config changed

    logger.info(f'Optimiser is {type(optimizer)}')
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total params {pytorch_total_params} (train) {pytorch_total_train_params}')


    checkpointer = DetectionCheckpointer(model,
                                         save_dir=os.path.join(cfg.OUTPUT_DIR, 'checkpoints'),
                                         random_state=random_state,
                                         optimizer=optimizer,
                                         scheduler=scheduler)
    periodic_checkpointer = PeriodicCheckpointer(checkpointer=checkpointer,
                                                 period=cfg.SOLVER.CHECKPOINT_PERIOD,
                                                 max_iter=cfg.SOLVER.MAX_ITER,
                                                 max_to_keep=None if cfg.FLAGS.KEEP_ALL else 10,
                                                 file_prefix='checkpoint')
    checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume_path is not None)
    iteration = 0 if args.resume_path is None else checkpoint['iteration']

    if cfg.FLAGS.TRAINVAL:
        train_loader, trainval_loader, val_loader = config.loaders(cfg, video=True)
    else:
        train_loader, val_loader = config.loaders(cfg, video=True)
        trainval_loader = None

    if cfg.INPUT.SAMPLING_FRAME_NUM > 1 and not cfg.FLAGS.VIDEO_MODE:
        logger.warning('Sampling frame num is greater than 1, but video mode is not enabled.')


    if len(train_loader.dataset) == 0:
        logger.error("Training dataset: empty")
        sys.exit(0)

    logger.info(
        f'Start of training: dataset {cfg.UNSUPVIDSEG.DATASET},'
        f' train {len(train_loader.dataset)}, val {len(val_loader.dataset)},'
        f' device {model.device}, keys {cfg.UNSUPVIDSEG.SAMPLE_KEYS}, '
        f'multiple flows {cfg.UNSUPVIDSEG.USE_MULT_FLOW}')

    iou_best = 0
    timestart = time.time()

    total_iter = cfg.TOTAL_ITER if cfg.TOTAL_ITER else cfg.SOLVER.MAX_ITER  # early stop


    gc_hist = []

    with torch.autograd.set_detect_anomaly(cfg.DEBUG) and \
         tqdm(initial=iteration, total=total_iter, disable=utils.environment.is_slurm()) as pbar:
        while iteration < total_iter:
            for sample in train_loader:
                sample = [e for s in sample for e in s]
                logger.info_once(f"RGB: {sample[0]['rgb'].shape} {sample[0]['flow'].shape} {sample[0]['sem_seg'].shape}")

                loss, train_log_dict = train_step(cfg, model, optimizer, scheduler, sample, iteration, total_iter)

                pbar.set_postfix(loss=loss)
                pbar.update()

                if (iteration + 1) % cfg.FLAGS.GC_FREQ == 0 or iteration < 100:
                    gc_hist.append(gc.collect())
                    gc_hist = gc_hist[-100:]

                if (iteration + 1) % 1000 == 0 or iteration + 1 in {1, 50}:
                    logger.info(
                        f'Iteration {iteration + 1}. AVG GC {np.nanmean(gc_hist)}. RNG outputs {utils.random_state.get_randstate_magic_numbers(model.device)}')

                if cfg.DEBUG or (iteration + 1) % 100 == 0:
                    logger.info(
                        f'Iteration: {iteration + 1}, time: {time.time() - timestart:.01f}s, loss: {loss:.02f}.')

                    for k, v in train_log_dict.items():
                        if writer:
                            writer.add_scalar(k, v, iteration + 1)

                    if writer:
                        writer.add_scalar('util/train_max_gpu_mem', torch.cuda.max_memory_allocated() / 2.0**20, iteration + 1)
                        torch.cuda.reset_max_memory_allocated()


                iteration += 1
                timestart = time.time()
                if iteration >= total_iter:
                    logger.info("Stopping")
                    checkpointer.save(name='checkpoint_final', iteration=iteration, loss=loss,
                                      iou=iou_best)
                    return iou_best  # Done
                periodic_checkpointer.step(iteration=iteration, loss=loss)
                del train_log_dict

                if (iteration) % cfg.LOG_FREQ == 0 or (iteration) in [1, 50, 500]:
                    logger.info(f"Eval GC: {gc.collect()}")
                    model.eval()
                    torch.cuda.reset_max_memory_allocated()
                    if writer:
                        with torch.no_grad():
                            image_viz = train_vis(cfg, model, sample)
                            writer.add_image('train/images', image_viz[0], iteration)
                            if len(image_viz) > 1:
                                for i in range(1, len(image_viz)):
                                    writer.add_image(f'extras/train_{i}', image_viz[i], iteration)
                            # if cfg.WANDB.ENABLE and (iteration) % 2500 == 0:
                            #     wandb.log({'train/viz': wandb.Image(image_viz.float())}, step=iteration)

                    if trainval_loader is not None:
                        iou = val.run_eval(cfg=cfg,
                                                    val_loader=trainval_loader,
                                                    model=model,
                                                    writer=writer,
                                                    writer_iteration=iteration,
                                                    prefix='trainval',
                                                    video=True)
                        if writer:
                            writer.add_scalar('trainval/IoU', iou, iteration + 1)

                    if iou := val.run_eval(cfg=cfg,
                                                    val_loader=val_loader,
                                                    model=model,
                                                    writer=writer,
                                                    writer_iteration=iteration,
                                                    video=True):
                        
                        if writer:
                            writer.add_scalar('util/eval_max_gpu_mem', torch.cuda.max_memory_allocated() / 2.0**20, iteration)
                    model.train()

def get_argparse_args():
    parser = ArgumentParser()
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--config-file', type=str,
                        default='configs/mask2former/swin/maskformer2_swin_tiny_bs16_160k.yaml')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_argparse_args().parse_args()
    if args.resume_path:
        args.config_file = "/".join(args.resume_path.split('/')[:-2]) + '/config.yaml'
        print(args.config_file)
    main(args)
