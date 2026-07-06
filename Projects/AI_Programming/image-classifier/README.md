# Image Classifier - Command Line App

Image Classifier (imc) is a Python Application with CLI Interface.

Its a custom trained a deep neural network on flower data set that others can use.

## Key Features
- Customizable - can rebuild your own neural network, and
- Retrained the network
- Independent Pridiction - can be use to predict with new Datasets
- Save the trained model as checkpoint and
- Use existing trained model
- Independent Image processing
- with Validation, you can save the configguration with context
- Switch Mode based on Device (gpu or cpu mode)

## Requirements
for conda users
`conda install -f envrionment.yml`



## Usage
python -m --help

[#Optional]
Installation
```
cd image-classifier
pip install -e .

```
With installation
```
imc
                                                                                                                        
 Usage: imc [OPTIONS] COMMAND [ARGS]...                                                                                 
                                                                                                                        
 Global Parameters                                                                                                      
 Usage:                                                                                                                 
 python -m imc -h                                                                                                       
 python -m imc --arch <architecture> --batch-size <batch_size> --data-dir <data_dir> validate                           
                                                                                                                        
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --data-dir            -d               Path to data directory [default: flowers]                                     │
│ --criterion           -lf              Loss Function to use, Default is nn.NLLLOS() [default: LossFunction.NLL]      │
│ --input-size          -i               Input Neuron size of the classifier [default: 0]                              │
│ --data-size           -ds              Number of sample data [default: None]                                         │
│ --hidden-size         -hu              Number of hidden neurons in the classifier [default: 512]                     │
│ --min-size            -ms              Minimum Number of hidden neurons in the classifier [default: 64]              │
│ --batch-norm          -bn              Enable Batch Normalization. Default is False                                  │
│ --dropout             -dr              Dropout rate for the classifier [default: 0.5]                                │
│ --gpu                 -gp              Enable GPU Device for Faster training. Default is False (Disable)             │
│ --weights             -w               Pre-trained weights to use [default: VGG16_Weights.IMAGENET1K_V1]             │
│ --arch                -a               Model architecture: vgg16, vgg13, densenet121 [default: vgg16]                │
│ --shuffle             -sh              Shuffle data in data loaders [default: True]                                  │
│ --learning-rate       -lr              Learning rate for training [default: 0.001]                                   │
│ --pretrained_model    -pr-model        Enable Pretrained Model [default: True]                                       │
│ --batch-size          -bs              Batch size [default: 32]                                                      │
│ --install-completion                   Install completion for the current shell.                                     │
│ --show-completion                      Show completion for the current shell, to copy it or customize the            │
│                                        installation.                                                                 │
│ --help                -h               Show this message and exit.                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ predict      Predict the class (or classes) of an image using a trained deep learning model.                         │
│ process      Preprocess images for training or inference.                                                            │
│ save         Save the trained model checkpoint to a file. Args:                                                      │
│ summary      Display model summary.                                                                                  │
│ train        Train the neural network model.                                                                         │
│ validate     Validate the model on the test dataset. 
```
***without installation***
```
python -m imc --help
                                                                                                                        
 Usage: python -m imc [OPTIONS] COMMAND [ARGS]...                                                                       
                                                                                                                        
 Global Parameters                                                                                                      
 Usage:                                                                                                                 
 python -m imc -h                                                                                                       
 python -m cli --arch <architecture> --batch-size <batch_size> --data-dir <data_dir> validate                           
                                                                                                                        
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --data-dir            -d               Path to data directory [default: flowers]                                     │
│ --criterion           -lf              Loss Function to use, Default is nn.NLLLOS() [default: LossFunction.NLL]      │
│ --input-size          -i               Input Neuron size of the classifier [default: 0]                              │
│ --data-size           -ds              Number of sample data [default: None]                                         │
│ --hidden-size         -hu              Number of hidden neurons in the classifier [default: 512]                     │
│ --min-size            -ms              Minimum Number of hidden neurons in the classifier [default: 64]              │
│ --batch-norm          -bn              Enable Batch Normalization. Default is False                                  │
│ --dropout             -dr              Dropout rate for the classifier [default: 0.5]                                │
│ --gpu                 -gp              Enable GPU Device for Faster training. Default is False (Disable)             │
│ --weights             -w               Pre-trained weights to use [default: VGG16_Weights.IMAGENET1K_V1]             │
│ --arch                -a               Model architecture: vgg16, vgg13, densenet121 [default: vgg16]                │
│ --shuffle             -sh              Shuffle data in data loaders [default: True]                                  │
│ --learning-rate       -lr              Learning rate for training [default: 0.001]                                   │
│ --pretrained_model    -pr-model        Enable Pretrained Model [default: True]                                       │
│ --batch-size          -bs              Batch size [default: 32]                                                      │
│ --install-completion                   Install completion for the current shell.                                     │
│ --show-completion                      Show completion for the current shell, to copy it or customize the            │
│                                        installation.                                                                 │
│ --help                -h               Show this message and exit.                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ predict      Predict the class (or classes) of an image using a trained deep learning model.                         │
│ process      Preprocess images for training or inference.                                                            │
│ save         Save the trained model checkpoint to a file. Args:                                                      │
│ summary      Display model summary.                                                                                  │
│ train        Train the neural network model.                                                                         │
│ validate     Validate the model on the test dataset.

```
## Sub Commands Usage

### Predict

```
imc predict --help
                                                                                                                        
 Usage: imc predict [OPTIONS]                                                                                           
                                                                                                                        
 Predict the class (or classes) of an image using a trained deep learning model.                                        
 Usage:                                                                                                                 
 python -m imc predict --image-path <"path_to_image"> --model-path <path_to_model> --topk 5                             
 python -m imc predict --image-path <"path_to_images"> --model-path <path_to_model_checkpoint> --topk 3 --label-names   
 cat_to_name.json                                                                                                       
                                                                                                                        
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --image-dir    -i         Path to image directory [default: None]                                                    │
│ --model-path   -m         Path to trained model file [default: None]                                                 │
│ --topk         -k         Number of top most likely classes to return [default: 5]                                   │
│ --label-names  -l         Path to JSON file mapping labels to real names [default: None]                             │
│ --gpu          -gp        Enable GPU Device for Faster inference. Default is False (Disable)                         │
│ --save-plot    -sp        Path to save the plot(s) [default: None]                                                   │
│ --show-plot    -dp        Display plot visualization of Top-k predictions [default: True]                            │
│ --help         -h         Show this message and exit.                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
