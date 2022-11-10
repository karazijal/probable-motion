## [Unsupervised Multi-object Segmentation by Predicting Probable Motion Patterns](https://www.robots.ox.ac.uk/~vgg/research/ppmp/)
### [![ProjectPage](https://img.shields.io/badge/-Project%20Page-magenta.svg?style=for-the-badge&color=002146&labelColor=white&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAARCAYAAAA7bUf6AAAABmJLR0QA/wD/AP+gvaeTAAAA5UlEQVQ4jb3TPUqDQRAA0NdoIVrYmUbv4T8GjSCJaHKNHEARb+JRBCFVRFRQCZiAVnY2Ygo7ix2b4MduEJxmYXd4DDOz/FMc4wlfcXamBU7wjHXMYBWjgIuBAWoT91u4+wsA8/jMAS0MKwBooJ9DrlGveFvGC/ZyyDsWI7GPj4APcIRuDoBeJF9gHwvYxSuaJQCp+29oS31p4Rw7uC0BujiT9uIysKsA5jDOAY0oeaXifVvBftyo7npNGvthDhlLJf8GDHCaA+AemxN3S9MApLGOsIZZbEifrxj4iTYepW//EHA2vgHTZjAVN1kZ7gAAAABJRU5ErkJggg==)](https://www.robots.ox.ac.uk/~vgg/research/ppmp/) [![arXiv](https://img.shields.io/badge/2210.12148-b31b1b.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2210.12148) [![In NeurIPS 22](https://img.shields.io/static/v1?label=&message=NeurIPS%202022&labelColor=white&color=blueviolet&style=for-the-badge&logo=data:image/png;base64,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)](https://nips.cc/Conferences/2022/Schedule?showEvent=55353) 
#### _[Laurynas Karazija*](https://karazijal.github.io), [Subhabrata Choudhury*](https://subhabratachoudhury.com/), [Iro Laina](http://campar.in.tum.de/Main/IroLaina), [Christian Rupprecht](https://chrirupp.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/)_
##### [Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/), University of Oxford

### Abstract
<sup> We propose a new approach to learn to segment multiple image objects without manual supervision. The method can extract objects form still images, but uses videos for supervision. While prior works have considered motion for segmentation, a key insight is that, while motion can be used to identify objects, not all objects are necessarily in motion: the absence of motion does not imply the absence of objects. Hence, our model learns to predict image regions that are likely to contain motion patterns characteristic of objects moving rigidly. It does not predict specific motion, which cannot be done unambiguously from a still image, but a distribution of possible motions, which includes the possibility that an object does not move at all. We demonstrate the advantage of this approach over its deterministic counterpart and show state-of-the-art unsupervised object segmentation performance on simulated and real-world benchmarks, surpassing methods that use motion even at test time. As our approach is applicable to variety of network architectures that segment the scenes, we also apply it to existing image reconstruction-based models showing drastic improvement. </sup>


## Getting Started
This repository builds on [Mask2Former](https://github.com/facebookresearch/Mask2Former).


### Requirements

Create and name a conda environment of your choosing, e.g. `ppmp`:
```bash
conda create -n ppmp python=3.9
conda activate ppmp
```
then install the requirements using this one liner:
```bash
conda install -y pytorch=1.12.1 torchvisio=0.13.1 cudatoolkit=11.3 -c pytorch && \
conda install -y kornia jupyter tensorboard timm einops scikit-learn scikit-image openexr-python tqdm -c conda-forge && \
conda install -y gcc_linux-64=7 gxx_linux-64=7 fontconfig && \
yes | pip install cvbase opencv-python filelock && \
yes | python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
cd mask2former/modeling/pixel_decoder/ops && \
sh make.sh
```

### Data Preparation
Datasets should be placed under `data/<dataset_name>`, like `data/movi_a` or `data/moving_clevrtex`.

#### Moving CLEVR/ClevrTex
For [MovingClevrTex](), download and place the tar files under `data/moving_clevrtex/tar`, see instructions [here](./DATASET.md). The dataloader is set up to build an index into tar files and read required information on the fly. 

#### Movi
For [MOVi](https://github.com/google-research/kubric/tree/main/challenges/movi) datasets, the files should be extracted to
`data/<dataset_name>/<train or validation>/<seq name>/` using `<seq name>_rgb_<frame num>.jpg` for rgb, `<seq name>_ano_<frame num>.png` for masks, `<seq name>_fwd_<frame num>.npz` or `<seq name>_bwd_<frame num>.npz` for forward/backward optical flow, repectively. For example:
```
data/movi_a/train/movi_a_5995/movi_a_5995_ano_017.png
data/movi_a/train/movi_a_5995/movi_a_5995_rgb_017.jpg
data/movi_a/train/movi_a_5995/movi_a_5995_fwd_017.npz
data/movi_a/train/movi_a_5995/movi_a_5995_bwd_017.npz
```
See [this notebook](https://github.com/google-research/kubric/blob/main/challenges/movi/VisualizeMOVi.ipynb) for details how to (down)load and normalise the Kubric datasets.

#### KITTI
For KITTI, RAFT flow is required. We followed processing from [here](https://github.com/charigyang/motiongrouping/tree/main/raft) with appropriate filepath changes for KITTI dataset structure.

### Running
Experiments are controlled through a mix of config files and command line arguments. See config files and [config.py](config.py) for a list of all available options.
For e.g. MOVi C dataset.

```bash
python main.py --config config_sacnn.yaml UNSUPVIDSEG.DATASET MOVi_C
```
or for MOVi D
```bash
# Note the switch to 24 object queries (slots)
python main.py --config config_sacnn.yaml UNSUPVIDSEG.DATASET MOVi_D MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 24  
```

#### Checkpoints
See [here](MODELS.md) for available checkpoints.

## Citation
```
@inproceedings{karazija22unsupervised,
    author = {Karazija, Laurynas and Choudhury, Subhabrata and Laina, Iro and Rupprecht, Christian and Vedaldi, Andrea},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {{U}nsupervised {M}ulti-object {S}egmentation by {P}redicting {P}robable {M}otion {P}atterns},
    volume = {35},
    year = {2022}
}
```


