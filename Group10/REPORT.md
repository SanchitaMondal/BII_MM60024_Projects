This report describes the Segmentation and classification pipeline implemented for dermoscopic images
using deep learning. The goal of preprocessing is to improve the quality of dermoscopic images before
feeding them into a neural network (U-Net) for lesion segmentation. Dermoscopy images often contain
artifacts such as hair, illumination variations, and inconsistent image sizes. The preprocessing pipeline
removes these artifacts and standardizes the dataset so that the neural network can learn meaningful
features more effectively.
The model is now trained and tested on these preprocessed data sets and evaluated on unseen images,metrics were calculated for that


# Pipeline Overview
# Segmentation part
1 Raw dermoscopic images are collected in a dataset folder.
2 Hair removal is performed using BlackHat morphological operation.
3 Detected hair pixels are removed using image inpainting.
4 Images are resized to a fixed resolution of 256 × 256 pixels.
5 Pixel values are normalized to the range [0,1].
6 The processed images are saved into a new folder for training and validation.Metrics such DICE score,IOU, pixel accuracy,precision and recall
7.Trained model is exported as weights in .pth file and used for prediction
8.TEsting with unseen images and image metrics were calculated

# Data analysis part ------> code.py file has the data analysis code
. Dataset Description
The dataset used in this project consists of pre-extracted 360 benign and 300 malignant dermoscopic images. For each of them ABCD features is extracted and saved in .Xlsx files.Each row corresponds to one image.
The dataset was split into:
•	Training set: 80% 
•	Validation set: 20% 

# Handcrafted features extraction:
Features used:
•	Asymmetry – Measures irregularity in lesion shape   -------> codes in ABCD1.py
•	Border Compactness – Indicates boundary smoothness 
•	Color Variance – Variation in color distribution 
•	Color Entropy – Complexity of color patterns 
•	Diameter (px) – Size of the lesion 
•	Area Ratio – Relative area coverage 
Target Variable:
•	Cancer_type 
o	Benign 
o	Malignant 
# Machine Learning Models used for comparison  -----codes in  code.py
The following models were implemented:
1. Logistic Regression
•	Linear model for binary classification 
•	Key parameter: max_iter = 1000 
2. Decision Tree
•	Rule-based model 
•	Key parameter: max_depth = 10 
3. Random Forest
•	Ensemble of decision trees 
•	Key parameter: n_estimators = 100 
4. Support Vector Machine (SVM)
•	Finds optimal separating hyperplane 
•	Uses probability estimation (probability=True) 

Model Evaluation Metrics
The following metrics were used for comparison:
•	Accuracy – Overall correctness 
•	Precision – Correct positive predictions 
•	Recall – Ability to detect malignant cases 
•	F1 Score – Balance between precision and recall 
•	AUC (ROC Curve) – Model discrimination ability 

# comparison results of various machine learning models

Model	                Accuracy	Precision	Recall	F1 Score	AUC
Logistic Regression	     0.705	      0.744	     0.533	  0.621 	0.794
Decision Tree	         0.606	      0.571  	 0.533	  0.552	    0.62
Random Forest	         0.674	      0.681	     0.533	  0.598	    0.759
SVM	                     0.705     	  0.744	     0.533	  0.621	    0.798

# Segmentation part 
dataset used:
Standard ISIC dataset with  Training images = 900 (with their ground truth masks)
                            Testing images = 360

# 1. Raw Image Input
The preprocessing pipeline begins with raw dermoscopic images stored in a dataset directory. Each
image contains a skin lesion along with possible artifacts such as hair strands, illumination variations,
and background noise. These artifacts can negatively affect the segmentation model's performance if
not removed.
# 2. Hair Removal using BlackHat Morphological Operation
Hair is one of the most common artifacts in dermoscopic images. To detect hair structures, a
morphological BlackHat operation is applied. The BlackHat transformation highlights dark structures
(such as hair) on a lighter background by computing the difference between the closing of the image
and the original image.
# 3. Hair Removal using Image Inpainting -------> preprocess.py
After detecting hair regions using the BlackHat operation, a binary mask is created using thresholding.
This mask identifies pixels corresponding to hair. The OpenCV inpainting algorithm is then applied to fill
these regions using neighboring pixel information. This effectively removes hair artifacts while
preserving surrounding image structures.
# 4. Image Resizing
Deep learning models require fixed-size inputs. Therefore, each image is resized to 256 × 256 pixels.
This size is commonly used in convolutional neural networks and provides a balance between
computational efficiency and feature preservation.
# 5. Image Normalization
Pixel intensities in the image are normalized by dividing each pixel value by 255. This scales pixel
values from the range [0,255] to [0,1]. Normalization stabilizes the training process and helps neural
networks converge faster.
# 6. Saving Processed Images
After preprocessing, the cleaned and resized images are saved into a new folder. The filenames remain
the same as the original images to maintain consistency with the corresponding ground truth
segmentation masks. These processed images are later used as input for the U-Net segmentation
model.

# U-net architecture ----> unet_model.py
The U-Net architecture consists of three main parts:
Each encoder block contains:

Two 3×3 convolution layers
ReLU activation
2×2 max pooling for downsampling
| Layer        | Feature Maps | Image Size |
| ------------ | ------------ | ---------- |
| Input        | 3            | 256×256    |   -----\   Encoder architecture
| Conv Block 1 | 64           | 256×256    |   -----/
| MaxPool      | 64           | 128×128    |
| Conv Block 2 | 128          | 128×128    |
| MaxPool      | 128          | 64×64      |
| Conv Block 3 | 256          | 64×64      |
| MaxPool      | 256          | 32×32      |
| Conv Block 4 | 512          | 32×32      |

Expanding Path (Decoder)

The decoder reconstructs the segmentation map.

Each decoder block contains:

Up-convolution (Transpose Convolution / UpSampling)
Concatenation with encoder feature maps
Two 3×3 convolution layers
ReLU activation

# Model parameters  -----> trained code in Unet.ipynb file
Loss Function:
Binary Cross Entropy (BCE) + Dice Loss is used for segmentation.

Loss = Binary Cross Entropy
Train/Val split : 80/20
Epochs = 30
Batch size = 16
Optimizer:
Optimizer = Adam
Learning Rate = 0.001

# Evaluation metrics
To evaluate the performance of the segmentation model, several quantitative metrics were used. These metrics compare the predicted segmentation mask with the ground truth mask.

The evaluation metrics used in this project include:

Intersection over Union (IoU)
Dice Coefficient
Precision
Recall
Pixel Accuracy

# Training Results
After 30 epochs
Metric	Value
Dice Score	0.8442
IoU	0.7337
Pixel Accuracy	0.9419
Precision	0.8848
Recall	0.8962
Training Results

# Testing results
The following metrics were obtained on the testing dataset, which was not used during training.
Metric	Value
Dice Score	0.9414
IoU	0.8893
Pixel Accuracy	0.9422
Precision	1.0000         ------>  Later, the model weights was saved into unet_segmentation.pth file for prediction
Recall	0.8893

# more detailed report is on Group 10 report.pdf
