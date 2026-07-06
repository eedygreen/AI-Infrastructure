# Pre-Trained Image Classifier

A Python-based image classification tool that uses pre-trained CNN models (ResNet, AlexNet, and VGG) to classify pet images and compare their performance. This project demonstrates the use of transfer learning with PyTorch to identify dog breeds and distinguish between dogs, cats, and other animals.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Running Batch Tests](#running-batch-tests)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Comparison](#model-comparison)
- [Output](#output)
- [Examples](#examples)

## Overview

This classifier uses pre-trained Convolutional Neural Network (CNN) models to classify pet images. The program:
1. Extracts pet labels from image filenames
2. Classifies images using your chosen CNN architecture
3. Compares predictions with true labels
4. Determines if images contain dogs
5. Provides detailed statistics on model performance

## Features

- Support for three CNN architectures: ResNet, AlexNet, and VGG
- Automated label extraction from filenames
- Dog breed identification
- Performance metrics and statistics
- Batch processing capabilities
- Detailed output reports with accuracy metrics

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)


## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the classifier with default settings (VGG model, pet_images directory):

```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```
This command will classify images in the `pet_images/` directory using the VGG model and compare results against the dog breed names listed in `dognames.txt`.


### Command Line Arguments

The program accepts three command line arguments:

| Argument | Description | Default Value | Example |
|----------|-------------|---------------|---------|
| `--dir` | Path to folder containing images | `pet_images` | `--dir pet_images/` |
| `--arch` | CNN model architecture | `vgg` | `--arch resnet` |
| `--dogfile` | Text file with dog breed names | `dognames.txt` | `--dogfile dognames.txt` |

**Available architectures:**
- `vgg` - VGG16 (recommended for accuracy)
- `resnet` - ResNet18 (good balance of speed and accuracy)
- `alexnet` - AlexNet (fastest but less accurate)

### Running Batch Tests

To test all three models and compare their performance:

```bash
sh run_models_batch.sh
```

This will run all three models and save results to:
- `resnet_pet-images.txt`
- `alexnet_pet-images.txt`
- `vgg_pet-images.txt`

For uploaded images:
```bash
sh run_models_batch_uploaded.sh
```

## Project Structure

```
pre-trained-image-classifier/
│
├── check_images.py                    # Main program
├── get_input_args.py                  # Command line argument parser
├── get_pet_labels.py                  # Extract labels from filenames
├── classify_images.py                 # Image classification logic
├── adjust_results4_isadog.py          # Dog breed verification
├── calculates_results_stats.py        # Performance statistics
├── print_results.py                   # Results display
├── classifier.py                      # CNN model interface
│
├── pet_images/                        # Sample pet images
├── uploaded_images/                   # User uploaded images
│
├── dognames.txt                       # List of valid dog breeds
├── imagenet1000_clsid_to_human.txt   # ImageNet class labels
├── requirements.txt                   # Python dependencies
│
└── run_models_batch.sh               # Batch testing script
```

## How It Works

1. **Label Extraction**: The program reads image filenames and extracts the pet label (e.g., "Golden_retriever_05182.jpg" → "golden retriever")

2. **Image Classification**: Each image is processed through the selected CNN model:
   - Image is resized to 256x256 pixels
   - Center cropped to 224x224 pixels
   - Normalized using ImageNet mean and standard deviation
   - Passed through the pre-trained model

3. **Dog Verification**: Results are compared against [dognames.txt](dognames.txt) to determine if the prediction is a dog breed

4. **Performance Analysis**: Statistics are calculated including:
   - Number of images correctly classified
   - Number of dogs correctly classified
   - Number of dog breeds correctly identified
   - Number of non-dogs correctly classified
   - Overall accuracy percentages

5. **Results Display**: Detailed results are printed showing classifications, matches, and performance metrics

## Model Comparison

Based on typical results:

| Model | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| **VGG** | Highest (~90%+) | Slowest | Best for accuracy-critical applications |
| **ResNet** | High (~85%+) | Medium | Best balance of speed and accuracy |
| **AlexNet** | Good (~80%+) | Fastest | Best for real-time applications |

## Output

The program provides comprehensive output including:

### Summary Statistics
- Total number of images
- Number of dog images
- Number of non-dog images
- Percentage of correct classifications
- Percentage of correct dog classifications
- Percentage of correct breed classifications
- Percentage of correct non-dog classifications

### Detailed Results
- Each image filename with true label
- CNN classifier prediction
- Match status (correct/incorrect)
- Dog breed identification results

### Runtime Information
- Total elapsed time in HH:MM:SS format

## Examples

### Example 1: Classify images using VGG model

```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

### Example 2: Classify images using ResNet model

```bash
python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt
```

### Example 3: Classify custom images

```bash
python check_images.py --dir uploaded_images/ --arch vgg --dogfile dognames.txt
```

### Example 4: Test the classifier with a single image

```bash
python test_classifier.py
```

## Image Naming Convention

For proper label extraction, image files should follow this naming pattern:
- `Dog_breed_name_###.jpg` (e.g., `Golden_retriever_05182.jpg`)
- `Animal_name_##.jpg` (e.g., `cat_01.jpg`)

The program will:
1. Remove numbers and file extensions
2. Convert underscores to spaces
3. Convert to lowercase for matching

## Notes

- The classifier uses models pre-trained on ImageNet (1000 classes)
- Best results are achieved with clear, well-lit images
- VGG model generally provides the highest accuracy for dog breed classification
- The [dognames.txt](dognames.txt) file contains 133 dog breed names for verification
- Results are compared case-insensitively

## Troubleshooting

**Issue**: ModuleNotFoundError for torch or torchvision
- **Solution**: Install PyTorch: `pip install torch torchvision`

**Issue**: Image file not found
- **Solution**: Check that the `--dir` path is correct and contains valid image files

**Issue**: Model not found
- **Solution**: Ensure you're using one of the three supported architectures: vgg, resnet, or alexnet

## Author

Programmer: Idris Isah
Date Created: 10/11/2025
Last Updated: 10/11/2025

## Acknowledgments

This project is based on the Master degree in AI course, AI Programming with Python course image classification project, demonstrating practical applications of transfer learning with pre-trained CNN models.

