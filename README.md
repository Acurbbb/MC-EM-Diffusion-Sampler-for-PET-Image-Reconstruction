# MC-EM Diffusion Sampler for PET Image Reconstruction

A diffusion sampler that combines manifold constraint projection with MLEM for better and faster conditional sampling.

## Overview

This repo realizes the training of score function and image reconstruction, for both 2D demo and 3D real data. 

## Installation

### Environment

```bash
conda env create -f environment.yml
conda activate guided-diff
```
### List-mode projector for 3D reconstruction. 

Requires PyBind11. This is a minimal CPU-based ray-tracing projector implemented in C++ using OpenMP for parallelization. Build tested on gcc 6.3. It is recommended to build a package after you create your environment using the following code.

```bash
cd ./projector
pip install -e .
```

## Training

The entry points are run_train_2d.sh and run_train_patch.sh for 2D and 3D respectively. You may need to modify the dataset classes to fit your data format. Add --parallel flag to enable multi-GPU training.

We also provide our pretrained 2D and 3D models in this [box folder](https://ucdavis.box.com/s/8byihqwatccjpqtf5cik281xs4hguizt). 

## 2D Image reconstruction

### Step 1: Make system matrix

The 2D demo uses a pre-computed system matrix. Please run ./misc/make_sysmat.py for the non-TOF system matrix and the corresponding image domain PSF model used in our paper. 

### Step 2: Phantom

In each run, a phantom is required to generate the simulated scan with the system matrix. We have provided a 2D phantom under ./phantom. 

### Step 3: Pre-trained model

Download and place it in your directory, or train your own model.

### Step 4: Run reconstruction

All the configurations can be found in ./configs/recon_mc-em_2d.yaml. System matrix, iPSF model, phantom, and model state dict are required. You can also override the configuration in the command line. A running example can be found in run_recon_2d.sh

We also provide the configuration files for baseline methods including DPS and PET-DDS. They can be used by demo_mc-em_2d.py as well.


## 3D Image reconstruction

The 3D version needs your list-mode data and the corresponding additive factor for randoms and scatter, sensitivity image, pre-trained model and scanner crystal positions.

### Step 1: Prepare list-mode data

We use our in-house list-mode format. Each event is represented by 5 int16 numbers [tx1, ax1, tx2, ax2, tof], corresponding to the transaxial and axial crystal IDs for the first and the second event and the TOF ID. For randoms and scatter, you need to convert them into list-mode as well. They are encoded as a list of float32 number for each list-model event, representing the contribution of scatter and randoms as additive factors to each LOR. Please refer to the PETOperatorLM class in utils/operator_pet.py for the detailed usage of these two data.

### Step 2: [Optional] Sensitivity map

Generate your own sensivity map based on your system and the attenuation of your subject.

### Step 3: Pre-trained model 

Download and place it in your directory, or train your own model.

### Step 4: Scanner 

We use a pre-computed LUT for the crystal locations in the xy and z direction. An example for the NX scanner (non-DOI) can be found in ./utils/xtal_pos_xy_uih and xtal_pos_z. They will be used to create the scanner.

### Step 5: Run Reconstruction

The reconstruction script shares similar structure as the 2D version. All the configurations can be found in ./configs. An example entry point can be found in run_recon_patch.sh


## License

The code in this repository is licensed under the MIT License.

## Contact

For any questions or inquiries, please contact bybhuang@ucdavis.edu