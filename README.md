# FIRe: Fast Inverse Rendering using Directional and Signed Distance Functions
# WACV 2024
[Arxiv](https://arxiv.org/abs/2203.16284)

This project is the official implementation our work, FIRe. Much of our code is from [DeepSDF's](https://github.com/facebookresearch/DeepSDF), and i3DMM's repositories. We thank Park et al. for making their code publicly available.

The pretrained models can be downloaded [here](#).

### Setup
1. To get started, clone this repository into a local directory.
2. Install [Anaconda](https://www.anaconda.com/products/individual#linux), if you don't already have it.
3. Create a conda environment in the path with the following command:
```
conda create -n fire python==3.8

```
3. Activate the conda environment from the same folder:
```
conda activate ./fire
```
4. Use the following commands to install required packages:
```
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install trimesh[all] opencv-python scikit-image imageio plyfile scikit-learn
```
### Preparing Data

Download and extract the data from [Differentiable Volumetric Rendering](https://github.com/autonomousvision/differentiable_volumetric_rendering) (the NMR dataset).

Clone and run the [DeepSDF's](https://github.com/facebookresearch/DeepSDF) preprocessing code for SDF samples and surface samples for evaluation.

Download the [ShapeNet V2](https://shapenet.org/) dataset.

Link the paths to the current directory as follows:
```
ln -s <path to DeepSDF folder with preprocessed SDF training samples>/data/* DeepSDFData/
ln -s <path to shapenet class folders of the dataset from differentiable volumetric rendering (DVR)>/dvr_rendered_ShapeNetV1/* dvr
ln -s <path to shapenet v2>/ShapeNetCore.v2/* ShapeNetCore.v2/
```
### Preprocessing

The following commands prepare data for sampling DDF from the meshes and the split files for different experiments.

#### Sample positive DDF samples
```
python preprocessData.py -i dataSph -e experiments/ours/<experiment name> --nS 1000000 --nD 1 -s Train --sMode 0 --skip
```
#### Sample negative DDF samples
```
python preprocessData.py -i dataSph -e experiments/ours/<experiment name> --nS 500000 --nD 1 -s Train --sMode 1 --skip
```

### Training the Model

Once data is preprocessed, one can train the model with the following command.

```
python train_fire.py -e experiments/ours/<experiment name> [--batch_split=2] [-c latest]
```

We provide networks for [PRIF](https://arxiv.org/abs/2208.06143), and [NeuralODF](https://arxiv.org/abs/2206.05837). Please see the [experiments](experiments). The training command remains the same, with the changed experiment name.

When working with a large dataset, use the batch_split argument. To continue from a previous iteration, use the '-c latest' argument.

### Infering from the model and evaluations

To fit the model to depth maps from the ShapeNet dataset given a splits file:
```
python infer_fire.py -e experiments/ours/<experiment name> -c latest -d dataSph --imW 512 --imH 512 -t 0.009 --rc fit -s experiments/splits/<splits file>   --fitTo [silhouette or depth] --fitDIST --meshThresh 0.001
```


### Citation

Please cite our paper if you use any part of this repository. Please cite PRIF and NeuralODF if you use their networks.
```
@inproceedings{yenamandra2024fire,
 author = {T Yenamandra and A Tewari and N Yang and F Bernard and C Theobalt and D Cremers},
 title = {FIRe: Fast Inverse Rendering Using Directional and Signed Distance Functions},
 booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
 month = {January},
 year = {2024},
 pages = {3077-3087},
}
```
