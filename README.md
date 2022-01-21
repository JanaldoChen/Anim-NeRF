# Animatable Neural Radiance Fields from Monocular RGB Videos
### [Github](https://github.com/JanaldoChen/Anim-NeRF) | [Paper](https://arxiv.org/abs/2106.13629)

![overview](assets/Overview.png)

> [Animatable Neural Radiance Fields from Monocular RGB Videos](https://arxiv.org/abs/2106.13629)    
> Jianchuan Chen, Ying Zhang, Di Kang, Xuefei Zhe, Linchao Bao, Xu Jia, Huchuan Lu

## Demos

* male-3-casual

    <img width="30%" src='assets/demos/male-3-casual/input.gif'></img><img width="60%" src='assets/demos/male-3-casual/male-3-casual.gif'></img>

More demos please see [Demos](./DEMOS.md).

## Requirements
- Python3.7 or later
- [PyTorch](https://pytorch.org/) 1.6 or later
- Pytorch-lightning
- [KNN_CUDA](https://github.com/unlimblue/KNN_CUDA)

### For visualization
- Trimesh
- PyMCubes
  
Run the following code to install all pip packages:
```sh
pip install -r requirements.txt
```

To install [KNN_CUDA](https://github.com/unlimblue/KNN_CUDA), we provide two ways:
* from source
  ```sh
  git clone https://github.com/unlimblue/KNN_CUDA.git
  cd KNN_CUDA
  make && make install
  ```
* from wheel
  ```sh
  pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
  ```

## Data Preparation
### People-Snapshot datasets
* prepare images and smpls
  ```sh
  python people_snapshot.py --data_root ${path_to_people_snapshot_datasets} --people_ID male-3-casual --gender male --output_dir data/
  ```
* prepare template
  ```sh
  python prepare_template.py --data_root data/ people_ID male-3-casual --model_type smpl --gender male --model_path ${path_to_smpl_models}
  ```

## Training
```sh
python train.py --cfg_file configs/people_snapshot/male-3-casual.yml
```
## Visualization
### Novel view synthesis
```sh
python novel_view.py --ckpt_path checkpoints/male-3-casual/last.ckpt
```
### 3D reconstruction
```sh
python extract_mesh.py --ckpt_path checkpoints/male-3-casual/last.ckpt
```
### Novel pose synthesis
```sh
python novel_pose.py --ckpt_path checkpoints/male-3-casual/last.ckpt
```

## Citation

If you find the code useful, please cite: 

```
@misc{chen2021animatable,
      title={Animatable Neural Radiance Fields from Monocular RGB Videos}, 
      author={Jianchuan Chen and Ying Zhang and Di Kang and Xuefei Zhe and Linchao Bao and Xu Jia and Huchuan Lu},
      year={2021},
      eprint={2106.13629},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
Parts of the code were based on from kwea123's NeRF implementation: https://github.com/kwea123/nerf_pl.
Some functions are borrowed from PixelNeRF https://github.com/sxyu/pixel-nerf
