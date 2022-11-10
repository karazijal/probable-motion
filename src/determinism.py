import os

_explanations = {
    0: 'No determinism',
    1: 'No CUDA benchmarking',
    2: 'Deterministic operations where available',
    3: 'Only deterministic operations are allowed'
}

lvl = int(os.environ.get('TRY_DETERMISM_LVL', '2'))
if lvl > 0:
    print(f'Attempting to enable deterministic cuDNN and cuBLAS operations to lvl {lvl} ({_explanations[lvl]})')
if lvl >= 2:
    # turn on deterministic operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"  #Need to set before torch gets loaded
    import torch
    if torch.version.__version__ >= '1.12.0':
        torch.use_deterministic_algorithms(True, warn_only=(lvl < 3))
    elif lvl >= 3:
        torch.use_deterministic_algorithms(True)  # This will throw errors if implementations are missing
    else:
        print(f"Torch verions is only {torch.version.__version__}, which will cause errors on lvl {lvl}")
if lvl >= 1:
    import torch
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False


def do_not_delete():
    """This exists to prevent formatters from treating this file as dead code"""
    pass
