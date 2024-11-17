# APTPose: Anatomy-aware Pre-Training for 3D Human Pose Estimation

This repo is the official implementation for **APTPose: Anatomy-aware Pre-Training for 3D Human Pose Estimation**. The paper has been accepted to **BMVC 2024**.

<p align="center"><img src="demo/demo_video.gif" width="50%" alt="" /></p>

## Dependencies
The code is developed and tested under the following environment

- Python 3.8.2
- PyTorch 1.7.1
- CUDA 11.0

You can create the environment from our **aptpose.yml**
```bash
conda env create -f aptpose.yml
```
Then activate **aptpose** environment
```
conda activate aptpose
```

<!--
Make sure you have the following dependencies installed:

* PyTorch >= 0.4.0
* NumPy
* Matplotlib=3.1.0
* FFmpeg (if you want to export MP4 videos)
* ImageMagick (if you want to export GIFs)
* Matlab
-->


## Dataset
You can download the processed **Human3.6M & MPI-INF-3DHP data** from [data folder]([https://drive.google.com/drive/folders/1MuG9RMYvJT69Wxx2evYktePoCoTs6f88?usp=share_link](https://drive.google.com/drive/folders/1N2VmRYKDknNW44bMOGiBygcscbj8N1uO?usp=sharing) in our google drive. Please put them in the `./dataset` directory. 
(Preprocessing code are borrowed from [P-STMO](https://github.com/paTRICK-swk/P-STMO))

For **Human3.6M**, `data_2d_h36m_gt.npz` is the ground truth of 2D keypoints. `data_2d_h36m_cpn_ft_h36m_dbb.npz` is the 2D keypoints obatined by [CPN](https://github.com/GengDavid/pytorch-cpn).  `data_3d_h36m.npz` is the ground truth of 3D human joints.

For **MPI-INF-3DHP**, `data_train.3dhp.npz` and `data_test.3dhp.npz` are the training and testing preprocessed data respectively. Each file contain both **data_3d** and **data_2d**.

<!--## Model Checkpoint
You can download our models from [model folder](https://drive.google.com/drive/folders/1MuG9RMYvJT69Wxx2evYktePoCoTs6f88?usp=share_link) in our google drive. Please put them (e.g. h36m_cpn_5_4260.pth) in the `./checkpoint` directory. -->

## Training script
### Human3.6M

**(1) For the Stage I's pre-training stage**, our model aims to solve the Hierarchical masked pose modeling (HMPM) task.Please run:

```bash
python run.py -f 243 -b 160 --MAE --train 1 --layers 4 -tds 2 -tmr 0.8 -smn 2 --lr 0.0001 -lrd 0.97
```

**(2) For the Stage II's fine-tuning stage**, the pre-trained encoder is loaded to our STMO model and fine-tuned. Please run:

```bash
python run.py -f 243 -b 160 --train 1 --layers 4 -tds 2 --lr 0.0007 -lrd 0.97 --MAE_reload 1 --previous_dir your_best_model_in_stage_I.pth
```


### MPI-INF-3DHP

We only train and evaluate our model on MPI-INF-3DHP dataset using the ground truth of 2D keypoints as inputs.

**(1) For the Stage I's pre-training stage**, please run:
```bash
python run_3dhp.py -f 81 -b 160 --MAE --train 1 --layers 3 -tmr 0.7 -smn 2 --lr 0.0001 -lrd 0.97
```

**(2) For the Stage II's fine-tunining stage**, please run:
```bash
python run_3dhp.py -f 81 -b 160 --train 1 --layers 3 --lr 0.0007 -lrd 0.97 --MAE_reload 1 --previous_dir your_best_model_in_stage_I.pth
```


## Evaluating script
After downloading our models from [model folder](https://drive.google.com/drive/folders/1SWrPoHstyKwjoAl63DgdVYS8B955Qtoq?usp=sharing) and put them (e.g. model_243_pretrain/) in the `./checkpoint` directory.

### Human3.6M
**(1) To evaluate our model (trained using GT 2D keypoints)**
```bash
python run.py -k gt -f 243 -tds 2 --reload 1 --layers 8 --previous_dir checkpoint/Best_Result/h36m_gt_13_2689.pth
```

**(2) To evaluate our model (trained using the 2D keypoints from CPN)**
```bash
python run.py -f 243 -tds 2 --reload 1 --layers 8 --previous_dir checkpoint/Best_Result/h36m_cpn_5_4260.pth
```

### MPI-INF-3DHP
(1) To evaluate our model on MPI-INF-3DHP dataset, please run:
```bash
python run_3dhp.py -f 81 --reload 1 --previous_dir checkpoint/Best_Result
/3dhp_66_3081.pth
```

(2) To evaluate our model on MPI-INF-3DHP dataset (pretrain using h36m and coco datas, then finetune on 3dhp data), please run:
```bash
python run_3dhp.py -f 81 --reload 1 --previous_dir checkpoint/Best_Result/3dhp_mixcoco_114_3052.pth
```


## Testing on in-the-wild videos

### For the pre-training stage, please run:

```bash
python run_in_the_wild.py -k detectron_pt_coco -f 243 -b 160 --MAE --train 1 --layers 4 -tds 2 -tmr 0.7 -smn 2 --lr 0.0001 -lrd 0.97 -c in-the-wild/4_16_v1
```

### For the fine-tuning stage, please run:

```bash
python run_in_the_wild.py -k detectron_pt_coco -f 243 -b 160 --train 1 --layers 4 -tds 2 --lr 0.001 -lrd 0.97 --MAE_reload 1 --previous_dir checkpoint/in-the-wild/4_16_v1_243_pretrain/MAE_4_8020.pth -c in-the-wild/4_16_v1
```

## Acknowledgement
Our code refers to the following repositories. Thanks authors for releasing the codes.
* [P-STMO](https://github.com/paTRICK-swk/P-STMO)
