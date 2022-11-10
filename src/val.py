import torch
from tqdm import tqdm

import dist
import utils
from clevrtex_eval import CLEVRTEX_Evaluator
from utils import convert, environment, log, visualisation
from utils import postproc

label_colors = visualisation.create_label_colormap()
logger = log.getLogger("unsup_vidseg")


def get_masks(cfg, preds_dict, sample_dict, force_ccpp=False, k=None):
    tspec = {
        "dtype": preds_dict["sem_seg"].dtype,
        "device": preds_dict["sem_seg"].device,
    }

    masks_raw = dist.logit_model("id", preds_dict)
    masks_softmaxed = torch.softmax(masks_raw, dim=1).squeeze(2)

    masks_softmaxed_sel = get_softmasks(cfg, masks_softmaxed, force_ccpp)

    if k is None:
        k = masks_softmaxed_sel.shape[1]
        if masks_softmaxed_sel.shape[0] == 1:
            k = convert.to_5dim_hard_mask(sample_dict["sem_seg"], **tspec).shape[1]
            k = max(masks_softmaxed_sel.shape[1], k)

    true_masks = convert.to_5dim_hard_mask(sample_dict["sem_seg"], k, **tspec)
    pred_masks = convert.to_5dim_hard_mask(masks_softmaxed_sel, k, **tspec)

    return masks_softmaxed, pred_masks, true_masks


def get_softmasks(cfg, masks_softmaxed, force_ccpp=False):

    if cfg.FLAGS.USE_CCPP or force_ccpp:
        masks_softmaxed_pp = postproc.connected_filter(
            masks_softmaxed, filt=0.001
        ).squeeze(2)
        logger.debug_once(f"CCPP {masks_softmaxed.shape} -> {masks_softmaxed_pp.shape}")
    else:
        masks_softmaxed_pp = masks_softmaxed
    return masks_softmaxed_pp


def run_eval(
    cfg, val_loader, model, writer=None, writer_iteration=0, video=False, prefix="eval"
):
    logger.info(
        f"Running {prefix}: {cfg.LOG_ID} Dataset: {cfg.UNSUPVIDSEG.DATASET} #slots: {cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES}"
    )

    with torch.no_grad():
        eval_log_dicts = []
        evaluator = CLEVRTEX_Evaluator(masks_have_background=True)
        pp_evaluator = CLEVRTEX_Evaluator(masks_have_background=True)
        visualizer = visualisation.Visualiser(cfg)

        for idx, sample in enumerate(tqdm(val_loader, disable=environment.is_slurm())):
            t = len(sample[0])
            sample = [e for s in sample for e in s]
            preds = model.forward_base(
                sample, keys=cfg.UNSUPVIDSEG.SAMPLE_KEYS, get_eval=True
            )

            sample_dict = convert.list_of_dicts_2_dict_of_tensors(
                sample, device=model.device
            )
            if video and cfg.EVAL_WHOLE_SEQ:
                sample_dict = convert.to_batchxtime(sample_dict)
            preds_dict = convert.list_of_dicts_2_dict_of_tensors(
                preds, device=model.device
            )
            if video:
                preds_dict = convert.to_batch_and_time(preds_dict, 1)
            logger.info_once(
                f"{prefix.upper()} inputs: {[(k, utils.get_shape(v)) for k, v in sample_dict.items()]}"
            )
            logger.info_once(
                f"{prefix.upper()} outputs: {[(k, utils.get_shape(v)) for k, v in preds_dict.items()]}"
            )

            masks_softmaxed, pred_masks, true_masks = get_masks(
                cfg, preds_dict, sample_dict
            )
            logger.debug_once(
                f'PM {pred_masks.shape} {preds_dict["sem_seg"].shape} TM {true_masks.shape} {sample_dict[cfg.UNSUPVIDSEG.SAMPLE_KEYS[0]].shape}'
            )
            pred_masks, true_masks = evaluator.update(pred_masks, true_masks)

            pp_evaluator.update(
                *get_masks(cfg, preds_dict, sample_dict, force_ccpp=True)[1:]
            )

            eval_log_dict = {}
            eval_log_dict["__batch_size__"] = pred_masks.shape[0]
            eval_log_dicts.append(eval_log_dict)

            if writer is not None:
                logger.debug_once(f"Populating visualizer")
                visualizer.add(
                    sample_dict, preds_dict, masks_softmaxed, pred_masks, true_masks
                )

        mIoU = evaluator.statistic("mIoU")
        mIoU_fg = evaluator.statistic("mIoU_fg")
        ARI = evaluator.statistic("ARI")
        ARI_FG = evaluator.statistic("ARI_FG")

        ACC = evaluator.statistic("acc")
        SDIF = evaluator.statistic("slot_diff")

        if writer:
            eval_log_dict = convert.list_of_dicts_2_dict_of_tensors(
                eval_log_dicts, device="cpu"
            )
            total_samples = eval_log_dict.get("__batch_size__", torch.tensor([0])).sum()

            for k, t in eval_log_dict.items():
                if k == "__batch_size__":
                    continue
                if t.shape[0] != total_samples and val_loader.batch_size == 1:
                    logger.error(
                        f"{prefix} key {k} has shape {t.shape} but should have {total_samples} at dim 0"
                    )
                writer.add_scalar(
                    k,
                    (t * eval_log_dict["__batch_size__"]).sum().detach().cpu().item()
                    / total_samples,
                    writer_iteration,
                )

            writer.add_image(
                f"{prefix}/images", visualizer.img_vis(), writer_iteration
            )  # C H W

            writer.add_scalar(f"{prefix}/mIoU", mIoU, writer_iteration)
            writer.add_scalar(f"{prefix}/mIoU_fg", mIoU_fg, writer_iteration)
            writer.add_scalar(f"{prefix}/ARI", ARI, writer_iteration)
            writer.add_scalar(f"{prefix}/ARI-F", ARI_FG, writer_iteration)
            writer.add_scalar(f"{prefix}/acc", ACC, writer_iteration)
            writer.add_scalar(f"{prefix}/sdif", SDIF, writer_iteration)

            writer.add_scalar(
                f"{prefix}/mIoU_PP", pp_evaluator.statistic("mIoU"), writer_iteration
            )
            writer.add_scalar(
                f"{prefix}/ARI_PP", pp_evaluator.statistic("ARI"), writer_iteration
            )
            writer.add_scalar(
                f"{prefix}/ARI-F_PP", pp_evaluator.statistic("ARI_FG"), writer_iteration
            )

        ret_metric = mIoU
        if "pp" in cfg.FLAGS.METRIC.lower():
            ret_metric = pp_evaluator.statistic("mIoU")
            if "ari" in cfg.FLAGS.METRIC.lower():
                if "f" in cfg.FLAGS.METRIC.lower():
                    logger.info_once(f"returning ARI-F PP")
                    ret_metric = pp_evaluator.statistic("ARI_FG")
                else:
                    logger.info_once(f"returning ARI PP")
                    ret_metric = pp_evaluator.statistic("ARI")
            else:
                logger.info_once(f"returning mIoU PP")
        else:
            if "ari" in cfg.FLAGS.METRIC.lower():
                if "f" in cfg.FLAGS.METRIC.lower():
                    logger.info_once(f"returning ARI-F")
                    ret_metric = ARI_FG
                else:
                    logger.info_once(f"returning ARI")
                    ret_metric = ARI
            else:
                logger.info_once(f"returning mIoU")
        return ret_metric
