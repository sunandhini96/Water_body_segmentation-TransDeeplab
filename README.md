# TransDeeplab Model

# Task : Water Body Segmentation using Trans DeepLab model.

This project implements water body segmentation using the Trans DeepLab model. It includes code for training the model, evaluating its performance, and metrics calculations.

# Problem Statement:
Extracting water bodies from satellite images is challenging because water bodies can
appear differently in satellite images. Water pixels have various colors and patterns,
making it hard to tell them apart from other land features.

# Files Required:

- `model.py`: TransDeepLab model
- `training.ipynb`: Training the model code
- `evaluation.py`: Model evaluation code
- `transdeeplab and deeplab comparision.ipynb`: Comparision of trans deeplab and convolution based deeplab models.

# Installation:

git clone https://github.com/sunandhini96/Water_body_segmentation-TransDeeplab.git

cd Water_body_segmentation-TransDeeplab


# Usage:

### Run the training script to train the model:
   
run the training.ipynb 

### To evaluate the trained model:

python evaluation.py

# Dataset:

The project uses RGB satellite images and corresponding masks from Sentinel-2 A/B satellite. You can obtain the dataset https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies

### Example of RGB satellite image and true mask image 
- (white pixels represent water and black pixels represents other than water in true mask)

<img width="490" alt="image" src="https://github.com/sunandhini96/Water_body_segmentation-TransDeeplab/assets/63030539/66c66390-21b7-41ea-9504-d1509cea984b">


# Methodology: Trans Deeplab Architecture

<img width="888" alt="image" src="https://github.com/sunandhini96/Water_body_segmentation-TransDeeplab/assets/63030539/0d23610a-3f3f-4132-9bb3-1b8397d36f77">

# Output: 
### Four Sample input images with its True Mask, Predicted Mask for Trans DeepLab and Predicted Mask for DeepLab Model

<img width="578" alt="image" src="https://github.com/sunandhini96/Water_body_segmentation-TransDeeplab/assets/63030539/f6531785-9c5f-4c78-9bff-ef682a8d4ee6">

# Citation:

If you use this code in your research, please cite our recent paper for more details.

## More Details:

For a detailed explanation of the project and results, refer to our paper.

### Conference Paper Link : https://ieeexplore.ieee.org/document/10116882



