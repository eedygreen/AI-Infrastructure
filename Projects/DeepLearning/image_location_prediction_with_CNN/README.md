# Landmark Classification with Convolutional Neural Networks

Given a photo, predict the most likely landmark it shows (and so, where it was taken), out of 50 possible landmarks from across the world. The project builds the full pipeline: data loading and augmentation, a CNN trained from scratch, a transfer-learned CNN, an exported standalone model, and a small interactive app.

## Why this project

Photo sharing and storage services like to have location data for each photo. Many photos have no usable location metadata (the camera has no GPS, or the metadata was removed for privacy). If a recognizable landmark appears in the picture, a model can infer where it was taken. Doing this by hand is not feasible at the scale of uploaded photos, so we train a CNN to do it.

## Results

All numbers are accuracy on a held-out test set of 1,250 images (50 classes).

| Model | Test accuracy |
|---|---|
| CNN trained from scratch | 56.6% (707/1250) |
| ResNet18 transfer learning, frozen backbone (final layer only) | 72.2% (903/1250) |
| **ResNet18 transfer learning, last block fine-tuned** | **81.4% (1017/1250)** |

The final model is the fine-tuned ResNet18. One standard error on a 1,250-image test set is about 1 point, so differences of a point or two between runs should not be over-interpreted.

## What was built

### 1. Data pipeline
- Dataset: a subset of the Google Landmarks Dataset v2, with 50 landmark classes.
- 80/20 train/validation split of the training images; separate test set.
- Training transforms: resize to 256, `RandomResizedCrop(224, scale=(0.5, 1.0))`, `RandomRotation(15)`, random horizontal flip, normalization with the dataset's per-channel mean and standard deviation.
- Validation and test transforms: resize to 256, center crop 224, the same normalization (no augmentation).

### 2. CNN from scratch
- A stride-2 convolution stem (3 to 16 channels), then five blocks of Conv, BatchNorm, ReLU, MaxPool and Dropout2d (16, 32, 64, 128, 256, 512 channels), global average pooling, and a classifier (Dropout, Linear 512 to 512, ReLU, Dropout, Linear 512 to 50).
- The early downsampling was chosen to cut training time on a CPU-only machine (about a third of the cost of a network that keeps full resolution through the first block).
- Trained with Adam, label smoothing, a `ReduceLROnPlateau` scheduler, and checkpointing of the weights with the lowest validation loss.

### 3. Transfer learning
- ResNet18 pretrained on ImageNet, with the final layer replaced by a new `Linear(512, 50)`.
- Step 1: freeze the whole backbone and train only the new layer (72.2%). The loss curves flattened, showing the limit of a linear classifier on frozen features.
- Step 2: also unfreeze `layer4` (the last residual stage) and train it with a low learning rate (1e-4), keeping earlier layers frozen (81.4%).
- Training uses a `ReduceLROnPlateau` scheduler (factor 0.5, patience 3) and early stopping (patience 10); the checkpoint with the best validation loss is saved.

### 4. Exported predictor
- A `Predictor` module bundles the preprocessing (resize, center crop, dtype conversion, normalization), the model and a softmax into one module.
- It is exported with `torch.jit.script` to `checkpoints/transfer_exported.pt`. The exported file is standalone: it needs only `torch` to run, not the code in `src/`.

### 5. App
- A notebook app built with `ipywidgets` and run with Voila.
- The user uploads a photo; the app shows it and lists the top 5 predicted landmarks with their probabilities.
- Run it with:

```
voila app.ipynb --show_tracebacks=True
```

Note: with ipywidgets 8 and later, the uploaded file is read from `btn_upload.value[-1]["content"]`, not from the old `.data` attribute.

## Findings from the experiments

- A weight decay of 0.05 with Adam held the from-scratch model back; 1e-4 trained much better.
- Milder random crops (`scale=(0.5, 1.0)` instead of the default) improved validation loss.
- Label smoothing reduced overfitting and gave roughly 2 points of test accuracy in a controlled comparison, though that is within the noise of one run.
- A from-scratch network with very early downsampling trains about 3 times faster, at some cost in accuracy; the final design with a strided stem and enough epochs recovered it.
- Fine-tuning the last residual block of a pretrained ResNet18 gave about 9 points over a frozen backbone.
- Early stopping with a best-checkpoint rule saved a large part of the training time once validation loss plateaued.

## Project layout

```
src/            data loading, model definitions, training and optimization code,
                transfer learning, the exported Predictor, helpers
checkpoints/    saved weights and exported TorchScript models
cnn_from_scratch.ipynb
transfer_learning.ipynb
app.ipynb
```

## Setup

The experiments were run on a CPU-only machine with Python 3.13 in a virtual environment. Install the dependencies (PyTorch, torchvision, matplotlib, tqdm, livelossplot, ipywidgets, Voila, and JupyterLab), then run the notebooks in order:

1. `cnn_from_scratch.ipynb`
2. `transfer_learning.ipynb`
3. `app.ipynb`

CPU timings for reference: about 95 minutes for 100 epochs of the from-scratch model, and about 60 minutes for the fine-tuning run that stopped early at epoch 30. A GPU would make all of this much faster.

## Dataset

The landmark images are a subset of the Google Landmarks Dataset v2.
