# 🧬 Breast Cancer Classification using Ultrasound Images

---

## 📌 Introduction

Breast cancer is one of the most common cancers affecting women worldwide. Early detection plays a crucial role in improving survival rates. Ultrasound imaging is widely used for breast cancer diagnosis due to its safety and accessibility.

The objective of this project is to develop an automated classification system using deep learning techniques to distinguish between **benign** and **malignant** tumors from breast ultrasound (BUS) images. The system aims to assist radiologists by providing a computer-aided diagnosis (CAD) tool.

---

## 📊 Data

We used the **BUSBRA Breast Ultrasound Dataset**, which consists of:

- **106 ultrasound images**
- Corresponding **segmentation masks**
- A metadata file (`bus_data.csv`) containing clinical details

### Key Features from Dataset:
- `ID`: Unique identifier for each image
- `Pathology`: Label (benign / malignant)
- `Side`: Left / Right / Single lesion
- `BBOX`: Bounding box of tumor region

### Preprocessing Steps:
- Mapped `ID` → image filenames
- Converted labels:
  - benign → 0  
  - malignant → 1
- Resized images to **224 × 224**
- Normalized pixel values
- Applied segmentation masks to highlight tumor regions
- Converted grayscale images to **3-channel format** for pretrained models

---

## ❓ Questions & Answers

---

### 🔹 Q1: How to classify breast cancer using ultrasound images?

**Answer:**
We implemented a deep learning-based classification pipeline using Convolutional Neural Networks (CNN) and transfer learning (ResNet18).

---

### 🔹 Q2: How was the dataset handled?

**Answer:**
- Extracted image names from `ID` column
- Used `Pathology` column for labels
- Matched images with masks using naming conventions
- Created a structured dataset

---

### 🔹 Q3: What models were used?

#### ✅ Baseline Model: Custom CNN

- 3 convolutional layers
- ReLU activation + MaxPooling
- Fully connected classifier

**Performance:**
- Accuracy: 77%
- AUC: 0.825
- Recall (malignant): 0.60

---

#### ✅ Improved Model: ResNet18 (Transfer Learning)

Improvements:
- Used pretrained ResNet18
- Converted images to 3 channels
- Applied class weighting
- Reduced learning rate for stability

**Performance:**
- Accuracy: **91%**
- AUC: **0.958**
- Recall (malignant): **0.80**

---

### 🔹 Q4: Why did the improved model perform better?

**Answer:**
- Pretrained models capture complex image features
- Transfer learning reduces need for large datasets
- Class weighting improved detection of malignant cases
- Proper preprocessing ensured better input quality

---

### 🔹 Q5: What challenges were faced?

**Answer:**
- Dataset did not directly provide usable labels → required mapping
- Small dataset size → risk of overfitting
- Handling multiple image types (`l`, `r`, `s`)
- Initial errors with model training (resolved by fixing input channels and optimizer)

---

### 🔹 Q6: What are the limitations?

**Answer:**
- Small dataset (106 images)
- No external validation
- BBOX information not used (could improve performance further)

---

## 🧾 Code Snippets

### Dataset Preparation
```python
df['image_name'] = df['ID'] + ".png"
df['label'] = df['Pathology'].map({'benign': 0, 'malignant': 1})
