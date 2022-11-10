import functools
import glob
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import torch

_LOG_DICT = {}


@functools.lru_cache(None)  # always the same :)
def get_datestring_for_the_run():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_key(msg, args, kwargs):
    args_str = ', '.join([str(arg) for arg in args])
    kwargs_str = ', '.join([f'{str(k)}={str(v)}' for k, v in kwargs.items()])
    r = [msg]
    if args_str or kwargs_str:
        r.append(' % (')
    r.append(args_str)
    if args_str:
        r.append(', ')
    r.append(kwargs_str)
    if args_str or kwargs_str:
        r.append(')')
    # MyMessage % (arg1, arg2, kw1=v1m, kw2=v2m)
    return ''.join(r)


def debug_once(msg, *args, logger=None, **kwargs):
    key = _make_key(msg, args, kwargs)

    lvl = logging.DEBUG
    t = datetime.now()
    should_log = True

    if key in _LOG_DICT:
        plvl, pt = _LOG_DICT[key]
        # Do not overwrite
        if plvl > lvl:
            t = pt
            lvl = plvl
        should_log = False

    _LOG_DICT[key] = (lvl, t)
    if should_log:
        logger.debug(msg, *args, **kwargs)


def info_once(msg, *args, logger=None, **kwargs):
    key = _make_key(msg, args, kwargs)

    lvl = logging.INFO
    t = datetime.now()
    should_log = True

    if key in _LOG_DICT:
        plvl, pt = _LOG_DICT[key]
        should_log = plvl <= lvl and t - pt > timedelta(minutes=5)
        lvl = max(lvl, plvl)

    _LOG_DICT[key] = (lvl, t)
    if should_log:
        logger.info(msg, *args, **kwargs)


def getLogger(name):
    if name != 'unsup_vidseg' and not name.startswith('unsup_vidseg.'):
        name = 'unsup_vidseg.' + name
    logger = logging.getLogger(name)
    logger.info_once = functools.partial(info_once, logger=logger)
    logger.debug_once = functools.partial(debug_once, logger=logger)
    return logger


def checkpoint_code(log_path):
    code_path = Path(log_path) / 'code'
    if code_path.exists():
        code_path = code_path.with_name(f'code_{get_datestring_for_the_run()}')
    code_path.mkdir(parents=True, exist_ok=True)
    for file in glob.glob('*.py'):
        shutil.copy(file, code_path)
    shutil.copytree('datasets', code_path / 'datasets', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    # shutil.copytree('losses', code_path / 'losses', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    # shutil.copytree('scops', code_path / 'scops', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    shutil.copytree('utils', code_path / 'utils', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    # shutil.copytree('mask_former', code_path / 'mask_former', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    shutil.copytree('mask2former', code_path / 'mask2former', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    shutil.copytree('mask2former_video', code_path / 'mask2former_video', ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))
    # shutil.copytree('prob', code_path / 'prob',
                    # ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))


@torch.no_grad()
def log_grad_norms(writer, key, *models, p='F', iteration=0):
    getLogger(__name__).debug_once(f'Logging grad norms {len(models)}')
    for i, model in enumerate(models):
        if model is not None:
            for name, param in model.named_parameters():
                k = f'grad_norms/{key}/{i}/{name}'
                if (param.requires_grad and param.grad is not None):
                    g = param.grad
                    writer.add_scalar(k, torch.norm(g, p=p).detach().cpu().item(), global_step=iteration)
    writer.flush()
