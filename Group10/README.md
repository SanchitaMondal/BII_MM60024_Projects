# Skin Cancer Segmentation — 4-Level U-Net
# dataset
-dataset contains Train_data,Traiining_groundtruth
                  Test_data,Test_groundtruth
# Preprocess.py

-Uses for hair removal of skin cancer training set
-works on by applying blackhat transform inpainting method
-blackhat = closing(I)-I
-normalization(0-255)
-stored in train_images_processed
-stored in test_images_processed

# Data analysis

-ABCD.py
A-asymmetry
B-border compactness
C-color entropy
  color variance
D-diameter(per px)
  diameter(area ratio)  is acquired for data analysis
-features stored in ABCDfeatures.xlsx
-test features stored in Test.xlsx
#  unet.training and validation
# unet.ipynb - notebook contains code for model training
- trained model weights were stored as unet_skin_segmentation.pth
-unet_model.py contain model details

# predict.py
predict.py contains code for testing with unknown sample with ground truth (new data)
-loads the model weights and unet_model.py to perform prediction
-losses and metrics like dicescore
                         IOU
                         Pixel accuracy
                         precision and recall  were computed

# code.py
code.py gives the data analysis with ABCD features stored with different machine learning models
# linear regression
# Logistic regression
# Random forest
# SVM
# decision tree

# model comparison - correlation map ,confusion matrix,feature importance