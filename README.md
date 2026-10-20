# Mindscan: Synthetic Brain MRI Slices with a GAN

This repository contains the code behind my MSc project and the publication that came out of it:

> M. Drici and K. O. Oseni, **"Synthetic Brain Images: Bridging the Gap in Brain Mapping with Generative Adversarial Model"**, *International Journal of Managing Information Technology (IJMIT)*, vol. 16, no. 1, February 2024. DOI: [10.5121/ijmit.2024.16101](https://doi.org/10.5121/ijmit.2024.16101)

It is a cleaned-up rewrite of the original project code, made so I could share the work. The approach and model architecture are the same as in the paper. The code has been tidied, made portable, and had some bugs fixed (see [Changes from the original code](#changes-from-the-original-code)), so a model retrained with this version will not reproduce the paper's numbers exactly.

The original research proposal (background, literature review, research questions) is in [PROPOSAL.md](PROPOSAL.md).

## What it does

A deep convolutional GAN is trained on 2D sagittal slices of T1-weighted brain MRI scans and learns to generate new, synthetic 256×256 slices. Synthetic scans like these can supplement real data for training AI models in brain imaging without exposing real patients' data.

## Pipeline

| Step | File | Input | Output |
|---|---|---|---|
| 1 | `3.np_to_tensor.ipynb` | `DS_final/sagittal/*.nii.gz` | `dataset.pt`: 10,000 slices scaled to [-1, 1] |
| 2 | `4.GAN.ipynb` | `dataset.pt` | `checkpoint_last/` (model checkpoints), `img_output/` (sample grids every 50 epochs) |
| 3 | `5.generation_using_model.ipynb` | `checkpoint_last/` | A grid of newly generated slices |

`models.py` holds the generator and discriminator, shared by steps 2 and 3.

### Model

- **Generator:** a 512-dimensional noise vector is expanded to 4×4×256 with a transposed convolution, then upsampled six times (conv → batchnorm → ReLU → upsample) to a 256×256×1 image with a `tanh` output.
- **Discriminator:** Gaussian noise on the input to reduce overfitting, six conv → LeakyReLU → dropout → average-pool blocks down to 4×4×256, then two dense layers and a `sigmoid` output.
- **Training:** binary cross-entropy with noisy label smoothing (real labels in [0.95, 1.0], fake in [0.0, 0.05]), Adam at 1e-5 for both networks, batch size 64, 200 epochs.

## Data

The dataset is **not included** in this repository.

The slices were built from T1-weighted scans in public [OpenNeuro](https://openneuro.org) datasets, including AgeRisk, Cerebrovascular Reactivity Normative, Characterizing Habit Learning, Individual Risk, and a listening-task dataset. For each scan, the 31 slices around the centre of the volume were extracted along each axis, rotated and flipped to a common orientation, and padded or cropped to 256×256. This produced about 18,700 sagittal slices, saved as NIfTI files.

The library used to generate these slices from the raw scans has been separated from this project and is not included here. This repository starts from the finished slices.

To run the pipeline, put your own 256×256 sagittal slices (`.nii.gz`) in `DS_final/sagittal/`.

## Setup

TensorFlow 2.15 needs Python 3.9–3.11.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then run the notebooks in order. Training on 10,000 slices takes several hours, so use a GPU (the original model was trained on Lightning AI).

## Changes from the original code

- The discriminator loss used `from_logits=True` on a model that already ends with a sigmoid, so sigmoid was applied twice. It now uses `from_logits=False`.
- The noisy labels were drawn with NumPy inside a `tf.function`, so they were fixed once and reused on every step. They now use `tf.random` and change every step.
- Sample images were all saved under the final epoch number and overwrote each other. They are now saved per epoch.
- Preprocessing is vectorised and preallocated instead of appending one slice at a time, and files are sorted so the same 10,000 slices are used on every run.
- Hardcoded local paths were replaced with relative ones, the model definitions were moved into `models.py`, and the requirements were reduced to the packages actually used.

## Citation

If you use this code, please cite:

```bibtex
@article{drici2024synthetic,
  title   = {Synthetic Brain Images: Bridging the Gap in Brain Mapping with Generative Adversarial Model},
  author  = {Drici, Mourad and Oseni, Kazeem Oluwakemi},
  journal = {International Journal of Managing Information Technology (IJMIT)},
  volume  = {16},
  number  = {1},
  year    = {2024},
  month   = feb,
  doi     = {10.5121/ijmit.2024.16101}
}
```
