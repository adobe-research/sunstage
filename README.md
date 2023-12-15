# SunStage: Portrait Reconstruction and Relighting using the Sun as a Light Stage

This is the code for SunStage: Portrait Reconstruction and Relighting using the Sun as a Light Stage.

 * [Project Page](https://sunstage.cs.washington.edu)
 * [Paper](https://arxiv.org/abs/2204.03648)
 * [Video](https://www.youtube.com/watch?v=ZbEKvIpwYEs)

## Setup
The code can be run under any environment with Python 3.9 and above.
(It may run with lower versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:

    conda create -n sunstage python=3.9

Next, install the required packages:

    conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
    conda install fvcore iopath -c fvcore -c iopath -c conda-forge
    conda install pytorch3d==0.6.2 -c pytorch3d
    python -m pip install opencv-python pyquaternion scipy chumpy numpy==1.23.1 tensorboard scikit-image kornia

## Training
### Data
Download [FLAME model](https://flame.is.tue.mpg.de/download.php), choose **FLAME 2020** and unzip it, copy
`generic_model.pkl` into `./data/DECA/data`.  

Download [sample data](https://drive.google.com/file/d/1kyNpmGKCYWZ46osHOA-WzW5HCNAcgDA1/view?usp=sharing) and unzip it into `./data`.
A dataset is a directory with the following structure:

    data/${obj_id}
        ├── 0 # camera poses
        ├── deca_out                # DECA predictions
        ├── test_nohair             # Estimated masks
        ├── predictions.pth         # Estimated keypoints
        ├── to_ignore.txt           # Bad frame IDs
        └── video_sections.txt      # Video sections

### Stage 1
After preparing a dataset, you can train SunStage stage 1 by running:

    export DATASET_PATH=/path/to/dataset
    python train_s1.py \
        --data_dir $DATASET_PATH \
        --obj_name obj_name

Stage 1 optimizes for the camera parameters. When it finished running, you can train SunStage stage 2 by running:

    export DATASET_PATH=/path/to/dataset
    python train_s2.py \
        --data_dir $DATASET_PATH \
        --obj_name obj_name

## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{wang2023sunstage,
  title={Sunstage: Portrait reconstruction and relighting using the sun as a light stage},
  author={Wang, Yifan and Holynski, Aleksander and Zhang, Xiuming and Zhang, Xuaner},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20792--20802},
  year={2023}
}
```
