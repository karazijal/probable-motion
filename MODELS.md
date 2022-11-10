
## Models
The following pretrained models are available for download:

Dataset | Model details | Link
--------|---------------|-----
MovingClevr | M2F with SACNN | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/clevr.pth)
MovingClevrTex | M2F with SACNN | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/clevrtex.pth)
MOVi A | M2F with SACNN | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/movi_a.pth)
MOVi C | M2F with SACNN | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/movi_c.pth)
MOVi D | M2F with SACNN | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/movi_d.pth)
MOVi E | M2F with SACNN | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/movi_e.pth)
MOVi A | M2F with Swin+WL | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/movi_a_wl.pth)
MOVi C | M2F with Swin+WL | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/movi_c_wl.pth)
MOVi D | M2F with Swin+WL | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/movi_d_wl.pth)
MOVi E | M2F with Swin+WL | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/movi_e_wl.pth)
KITTI | M2F with R18 | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/kitti_r18.pth)
KITTI | M2F with R18+WL | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/kitti_r18_wl.pth)
KITTI | M2F with Swin+WL | [here](https://www.robots.ox.ac.uk/~vgg/research/ppmp/data123/ppmp_checkpoints/kitti_swin_wl.pth)

(SACNN - 6-layer CNN used in [Slot Attention](https://arxiv.org/abs/2006.15055), Swin - [Swin Transformer](https://github.com/SwinTransformer/Transformer-SSL) ([weights for kitti](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_swin_t_300ep_pretrained.pth)), WL - Weighted Loss, R18 - ResNet18)

To load them with an appropriate architecture, consider the following snippet:
```python
from types import SimpleNamespace
from mask2former_trainer_video import setup, Trainer

def load_model_cfg(config_path, weights_path):
    args = SimpleNamespace(config_file=config_path, 
                           opts=[
                            # Any extra arguments, such as number of slots.
                            'MODEL.WEIGHTS', str(weights_path),
                           ], eval_only=True)
    cfg = setup(args)
    model = Trainer.build_model(cfg)
    model.eval()
    return model
```