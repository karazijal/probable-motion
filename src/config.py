import copy
import glob as gb
import itertools
import os
from functools import partial
from pathlib import Path

import numpy as np
import numpy.random
import torch.utils.data
from detectron2.config import CfgNode as CN

import utils
from datasets import FlowPairCMDetectron, \
    FlowPairMoviDetectron, FlowPairNewDetectron, KITTI_VAL

logger = utils.log.getLogger('unsup_vidseg')


def setup_movi_dataset(cfg, num_frames=1):
    cache_path = None
    resolution = cfg.UNSUPVIDSEG.RESOLUTION  # h,w
    prefix = f'data/{cfg.UNSUPVIDSEG.DATASET.lower()}'

    if utils.environment.is_slurm():
        cache_path = Path(str(os.environ['TMPDIR'])) / str(
            utils.environment.get_slurm_id()) / f'{cfg.UNSUPVIDSEG.DATASET}_cache/'


    if 'smurf' in cfg.UNSUPVIDSEG.DATASET.lower():
        logger.info("Using SMURF dataset for training!")
        dataset_str = cfg.UNSUPVIDSEG.DATASET[:cfg.UNSUPVIDSEG.DATASET.lower().find('smurf')-1]
        prefix = f'data/{dataset_str.lower()}'
        train_dataset = FlowPairNewDetectron([1,-1], resolution,
                                                prefix=f'data/movi_smurf/{dataset_str.lower()}/train',
                                                cache_path=cache_path,
                                                two_flow=num_frames > 1 or cfg.UNSUPVIDSEG.LOSS_MULTI_FLOW,
                                                num_frames=num_frames,
                                                size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                                                res='',
                                                return_original=False,
                                                sequences=None,
                                                flows_prefix='Smurf_Flows_gap'
                                                )
    else:
        train_dataset = FlowPairMoviDetectron(
            'train', None,
            resolution,
            prefix=prefix,
            gt_flow=True,
            cache_path=cache_path,
            num_frames=num_frames,
            two_flow=num_frames > 1 or cfg.UNSUPVIDSEG.LOSS_MULTI_FLOW
        )
    val_dataset = FlowPairMoviDetectron(
        'validation',
        None,
        resolution,
        prefix=prefix,
        gt_flow=True,
        cache_path=cache_path,
        num_frames=None if cfg.EVAL_WHOLE_SEQ else 1
        )
    return train_dataset, val_dataset


def setup_moving_clevrtex(cfg, num_frames=1):
    pairs = [1, 2, -1, -2]

    cache_path = None
    resolution = cfg.UNSUPVIDSEG.RESOLUTION  # h,w

    if cfg.UNSUPVIDSEG.DATASET in ['CM.M', 'CM.M.GT', 'CM.M.F.GT']:

        prefix = 'data/moving_clevrtex'

        if utils.environment.is_slurm():
            cache_path = Path(str(os.environ['TMPDIR'])) / str(
                utils.environment.get_slurm_id()) / 'cache/moving_clevrtex/'


    elif cfg.UNSUPVIDSEG.DATASET in ['CM.R', 'CM.R.GT', 'CM.R.F.GT']:

        prefix = 'data/moving_clevr'

        if utils.environment.is_slurm():
            cache_path = Path(str(os.environ['TMPDIR'])) / str(
                utils.environment.get_slurm_id()) / 'cache/moving_clevr/'

    if cache_path is None:
        logger.warn("Cache path is not set, caching will be disabled.")

    train_dataset = FlowPairCMDetectron(
        split='train',  # This will sample
        pairs=pairs,
        flow_dir=None,
        res='240p',
        resolution=resolution,
        to_rgb=False,
        size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
        first=None,
        prefix=prefix,
        gt_flow=True,
        with_clevr=False,
        single_sample=False,
        two_flow=num_frames > 1 or cfg.UNSUPVIDSEG.LOSS_MULTI_FLOW,
        ccrop=True,
        cache_path=cache_path,
        filter=False,
        num_frames=num_frames,
        no_lims=True,
        first_frame_only=False,
        darken=False
    )
    val_dataset = FlowPairCMDetectron(
        split='val',  # This will process sequentially
        pairs=pairs,  # only first "flow pair" will be used
        flow_dir=None,
        res='240p',
        resolution=resolution,
        to_rgb=False,
        single_sample=False,
        size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
        first=None,
        prefix=prefix,
        gt_flow=True,
        with_clevr=False,
        two_flow=False,
        ccrop=True,
        cache_path=cache_path,
        filter=False,
        num_frames=None if cfg.EVAL_WHOLE_SEQ else 1,
        no_lims=True,
        darken=False
    )

    return train_dataset, val_dataset


def setup_dataset(cfg=None, num_frames=1):
    dataset_str = cfg.UNSUPVIDSEG.DATASET
    if '+' in dataset_str:
        datasets = dataset_str.split('+')
        logger.info(f'Multiple datasets detected: {datasets}')
        train_datasets = []
        val_datasets = []
        val_dataset_index = 0
        for i, ds in enumerate(datasets):
            if ds.startswith('val(') and ds.endswith(')'):
                val_dataset_index = i
                ds = ds[4:-1]
            proxy_cfg = copy.deepcopy(cfg)
            proxy_cfg.merge_from_list(['UNSUPVIDSEG.DATASET', ds]),
            train_ds, val_ds = setup_dataset(proxy_cfg)
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
        logger.info(f'Multiple datasets detected: {datasets}')
        logger.info(f'Validation is still : {datasets[val_dataset_index]}')
        return torch.utils.data.ConcatDataset(train_datasets), val_datasets[val_dataset_index]

    
    if cfg.UNSUPVIDSEG.DATASET in ['KITTI']:
        cache_path = None
        if cache_path is None:
            if utils.environment.is_slurm():
                cache_path = Path(str(os.environ['TMPDIR'])) / str(
                    utils.environment.get_slurm_id()) / f'{cfg.UNSUPVIDSEG.DATASET}_cache/'

        if not isinstance(cache_path, (str, Path)):
            cache_path = None

        if cfg.FLAGS.NO_CACHE:
            cache_path = None

        flows_prefix = 'Flows_gap'

        train_dataset = FlowPairNewDetectron([1, -1], cfg.UNSUPVIDSEG.RESOLUTION,
                                              prefix='data/KITTI',
                                              cache_path=cache_path,
                                              two_flow=num_frames > 1 or cfg.UNSUPVIDSEG.LOSS_MULTI_FLOW,
                                              num_frames=num_frames,
                                              size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                                              res='',
                                              return_original=False,
                                              sequences=None,
                                              rgb_suffix='.png',
                                              flows_prefix=flows_prefix,
                                              pseudo_index=None)

        val_dataset = KITTI_VAL(cfg.UNSUPVIDSEG.RESOLUTION,
                                prefix='data/KITTI_VAL/training',
                                cache_path=cache_path,
                                size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                                return_original=True,)
        return train_dataset, val_dataset

    if cfg.UNSUPVIDSEG.DATASET.startswith('MOVi'):
        return setup_movi_dataset(cfg, num_frames)

    if cfg.UNSUPVIDSEG.DATASET.startswith('CM'):
        return setup_moving_clevrtex(cfg, num_frames)

    raise ValueError(f'Unknown dataset {cfg.UNSUPVIDSEG.DATASET}')


def loaders(cfg, video=False):
    nf = 1
    if video:
        nf = cfg.INPUT.SAMPLING_FRAME_NUM
    train_dataset, val_dataset = setup_dataset(cfg, num_frames=nf)
    logger.info(f"Sourcing data from {val_dataset.data_dir[0]}")
    logger.info(f"training dataset: {train_dataset}")
    logger.info(f"val dataset: {train_dataset}")

    if cfg.FLAGS.DEV_DATA:
        subset = cfg.SOLVER.IMS_PER_BATCH * 3
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(subset)))
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(subset)))

    g = torch.Generator()
    data_generator_seed = int(torch.randint(int(1e6), (1,)).item())
    logger.info(f"Dataloaders generator seed {data_generator_seed}")
    g.manual_seed(data_generator_seed)

    val_loader_size = 1
    if not cfg.EVAL_WHOLE_SEQ and (cfg.UNSUPVIDSEG.DATASET.startswith('CM')
                                   or cfg.UNSUPVIDSEG.DATASET.startswith('MOVi')):
        val_loader_size = max(cfg.SOLVER.IMS_PER_BATCH, 16)
        logger.info(f"Increasing val loader size to {val_loader_size}")
    if cfg.UNSUPVIDSEG.DATASET.startswith('KITTI'):
        val_loader_size = 1  # Enfore singe-sample val for KITTI

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                               collate_fn=lambda x: x,
                                               shuffle=True,
                                               pin_memory=False,
                                               drop_last=True,
                                               persistent_workers=False and cfg.DATALOADER.NUM_WORKERS > 0,
                                               worker_init_fn=utils.random_state.worker_init_function,
                                               generator=g
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             batch_size=val_loader_size,
                                             shuffle=False,
                                             pin_memory=False,
                                             collate_fn=lambda x: x,
                                             drop_last=False,
                                             persistent_workers=False and cfg.DATALOADER.NUM_WORKERS > 0,
                                             worker_init_fn=utils.random_state.worker_init_function,
                                             generator=g)
    if cfg.FLAGS.TRAINVAL:
        rng = np.random.default_rng(seed=42)
        train_dataset_clone = copy.deepcopy(train_dataset)
        train_dataset_clone.dataset.random = False
        trainval_dataset = torch.utils.data.Subset(train_dataset_clone,
                                                   rng.choice(len(train_dataset), len(val_dataset), replace=False))
        trainval_loader = torch.utils.data.DataLoader(trainval_dataset,
                                                      num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                      batch_size=val_loader_size,
                                                      shuffle=False,
                                                      pin_memory=False,
                                                      collate_fn=lambda x: x,
                                                      drop_last=False,
                                                      persistent_workers=False and cfg.DATALOADER.NUM_WORKERS > 0,
                                                      worker_init_fn=utils.random_state.worker_init_function,
                                                      generator=g)
        return train_loader, trainval_loader, val_loader
    return train_loader, val_loader


def add_unsup_vidseg_config(cfg):
    cfg.UNSUPVIDSEG = CN()
    cfg.UNSUPVIDSEG.RESOLUTION = (128, 128)
    cfg.UNSUPVIDSEG.SAMPLE_KEYS = ["rgb"]
    cfg.UNSUPVIDSEG.DATASET = 'CM.M.GT'
    cfg.UNSUPVIDSEG.USE_MULT_FLOW = False


    cfg.UNSUPVIDSEG.FLOW_LIM = 20


    cfg.UNSUPVIDSEG.LOSS = 'ELBO_AFF_FULL'
    cfg.UNSUPVIDSEG.LOSS_ORIGIN = 'centroid_fix'
    cfg.UNSUPVIDSEG.LOSS_FLOW_KEY = 'flow'


    cfg.UNSUPVIDSEG.LOSS_GRID_DETACH = False
    cfg.UNSUPVIDSEG.LOSS_BETA = 'lin(5000,0.1,-0.1)'
    cfg.UNSUPVIDSEG.LOSS_NPART = 3
    cfg.UNSUPVIDSEG.LOSS_TEMP = 'const(1.0)'
    cfg.UNSUPVIDSEG.LOSS_SIGMA2 = 'const(0.5)'


    cfg.UNSUPVIDSEG.LOSS_COV = 'simple'
    cfg.UNSUPVIDSEG.LOSS_MEANS = False


    cfg.UNSUPVIDSEG.LOSS_EQUIV = 'const(0.0)'
    cfg.UNSUPVIDSEG.LOSS_DISP_THRESH = -1.0
    cfg.UNSUPVIDSEG.LOSS_MULTI_FLOW = False


    cfg.FLAGS = CN()
    cfg.FLAGS.METRIC = 'mIOU'
    cfg.FLAGS.KEEP_ALL = False  # Keep all checkoints

    cfg.FLAGS.UNFREEZE_AT = []

    cfg.FLAGS.DEV_DATA = False  # Run with artificially downsampled dataset for fast dev
    cfg.FLAGS.USE_CCPP = True
    
    cfg.FLAGS.GC_FREQ = 10
    cfg.FLAGS.TRAINVAL = False

    cfg.FLAGS.NO_CACHE = False

    cfg.DEBUG = False

    cfg.LOG_ID = 'exp'
    cfg.LOG_FREQ = 1000
    cfg.OUTPUT_BASEDIR = '../outputs'
    cfg.TOTAL_ITER = None
    cfg.CONFIG_FILE = None

    cfg.EVAL_WHOLE_SEQ = False
    

    if os.environ.get('SLURM_JOB_ID', None):
        cfg.LOG_ID = os.environ.get('SLURM_JOB_NAME', cfg.LOG_ID)
        print(f"Setting name {cfg.LOG_ID} based on SLURM job name")
