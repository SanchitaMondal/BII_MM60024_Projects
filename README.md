# Projects for MM60024 : Biomedical Imaging Informatics
> Repository for final project allocation and submission for MM60024 : Biomedical Imaging Informatics offered in Spring 2026 at IIT Kharagpur taught by [Prof Debashree Guha Adhya](https://www.iitkgp.ac.in/department/MM/faculty/mm-debashree_smst).

> This repository might be updated with new projects and/or changes to existing projects. Please check back regularly.

> Final projects are approved by [Prof Debashree Guha Adhya](https://www.iitkgp.ac.in/department/MM/faculty/mm-debashree_smst).

- [Final Projects for MM60024 : Biomedical Imaging Informatics](#final-projects-for-MM60024--Biomedical-Imaging-Informatics)
  - [Instructions](#instructions)
    - [General Instructions](#general-instructions)
    - [Evaluation Policy](#evaluation-policy)
    - [Instructions to get started with the project](#instructions-to-get-started-with-the-project)
    - [What to submit](#what-to-submit)
    - [Submission Instructions](#submission-instructions)
    - [Deadline](#deadline)
  - [Project Allocation](#project-allocation)
  - [Projects](#projects)
    - [Project 1 : Classification of Breast Cancer Using Ultrasound Images](#project-1--Classification-of-Breast-Cancer-Using-Ultrasound-Images)
    - [Project 2 : Maternal Health Risk Prediction in Personalized Patient Care](#project-2--Maternal-Health-Risk-Prediction-in-Personalized-Patient-Care)
    - [Project 3 : Early Detection of Coronary Heart Disease](#project-3--Early-Detection-of-Coronary-Heart-Disease)
    - [Project 4 : Cervical Cancer Diagnosis Based on Machine Learning Techniques](#project-4--Cervical-Cancer-Diagnosis-Based-on-Machine-Learning-Techniques)
    - [Project 5 : Prostate Cancer Diagnosis Using Machine Learning Methods](#project-5--Prostate-Cancer-Diagnosis-Using-Machine-Learning-Methods)
    - [Project 6 : Pneumonia Detection Using Chest X-ray Imaging](#project-6--Pneumonia-Detection-Using-Chest-X-ray-Imaging)
    - [Project 7 : ](#project-7--)
  - [Resources](#resources)
  - [All the best!](#all-the-best)

---


## Instructions

### General Instructions
1. The project is to be done in groups of 5 students. The students are expected to work together collaboratively.
2. The choice of programming language is left to the students. However, the most common language used is Python.
3. Each group will be assigned a mentor TA who will be responsible for guiding the group throughout the project.
4. Meetings with the mentor TA will be scheduled at the beginning of the project and at regular intervals.
5. Each student will be evaluated based on the contribution towards the project. Make sure you are contributing equally to the project.
6. Code plagiarism will not be tolerated. Any submission found to be plagiarized will be awarded a zero grade. 
7. Late submissions will not be accepted.
### Evaluation Policy
1. The final project evaluation is based on the following criteria:
   1. `Code Quality and Documentation : 60%`
   2. `Final Submission and Report : 40%` 
2. `Code Quality and Documentation` : 60%
   - This will be based on the following criteria:
      1. Code Quality : 30% (based on the code quality and readability)
      2. Documentation : 30% (based on the documentation of the code and the project)
3. `Final Submission and Report` : 40%
   - This will be based on the following criteria:
      1. Final Submission : 20% (based on the final submission of the project)
      2. Final Report : 20% (based on the final report of the project)


### Instructions to get started with the project
1. `Fork` this `github.com/SanchitaMondal/BII_MM60024_Projects` repository.
2. `Clone` the forked repository to your local machine using the following command:
   ```bash
   git clone github.com/{your_username}/BII_MM60024_Projects
   ```
3. Your projects are in the `submissions` directory. You can find the project description in the README.md file of the respective project directory.
4. Work on the project and make `regular commits` to your local repository and `push` them to your forked repository.
5. Your mentor TA will review your code and provide feedback.

### What to submit
1. You have to submit the following:
   1. `Final Code` : The final code of your project in the respective project directory. 
      1. Code should be highly readable and well documented.
      2. Try to write efficient code and avoid unnecessary code.
   2. `Final Report` : The final report of your project in the respective project directory. The report should be in the form of a `markdown` file with the name `report.md`. The report should contain the following:
      1. `Introduction` : A brief introduction of the project.
      2. `Data` : A brief description of the data used in the project.
      3. `Questions & Answers` : The questions and their respective answers. Also include the code snippets used to answer the questions and `who solved` the question.
      4. `References` : The references used in the project.

### Submission Instructions
1. Submission of the final project will be done via `GitHub Pull Requests`.
2. Once you are done with the project, you can create a `Pull Request` to the `main` branch of the `github.com/SanchitaMondal/BII_MM60024_Projects` repository.
3. We will review your merge request and provide feedback. You can make changes to your code and update the merge request. If accepted, your project will be merged to the `main` branch of the `github.com/SanchitaMondal/BII_MM60024_Projects` repository.
4. That's it! `Congratulations!!` have successfully submitted your final project.

### Deadline
The deadline for the final project submission is _th April 2026, 23:59 IST**.

## Project Allocation
|     Students                            | Project                |  Mentor TA    |
|:---------------------------:|:-----------------------------:|:------------------:|
|  | [Project 1 : Classification of Breast Cancer Using Ultrasound Images](#project-1--Classification-of-Breast-Cancer-Using-Ultrasound-Images) | _ |
|  | [Project 2 : Maternal Health Risk Prediction in Personalized Patient Care](#project-2--Maternal-Health-Risk-Prediction-in-Personalized-Patient-Care) | _ |
|  | [Project 3 : Early Detection of Coronary Heart Disease](#project-3--Early-Detection-of-Coronary-Heart-Disease) | _ |
|  | [Project 4 : Cervical Cancer Diagnosis Based on Machine Learning Techniques](#project-4--Cervical-Cancer-Diagnosis-Based-on-Machine-Learning-Techniques) | _ |
|  | [Project 5 : Prostate Cancer Diagnosis Using Machine Learning Methods](#project-5--Prostate-Cancer-Diagnosis-Using-Machine-Learning-Methods) | _ |
|  | [Project 6 : Pneumonia Detection Using Chest X-ray Imaging](#project-6--Pneumonia-Detection-Using-Chest-X-ray-Imaging) | _ |
|  |  | _ |


## Projects

### Project 1 : Classification of Breast Cancer using Ultrasound Images

1. This project aims to develop an automated breast cancer classification system using deep learning techniques to distinguish between benign and malignant tumors from BUS images. The dataset is located in the `data/Breast data` directory.
2. The study will utilize the BUS-BRA Breast Ultrasound Dataset, which contains annotated ultrasound images with pathology-confirmed labels and lesion segmentation masks.
3. Preprocessing techniques : image `resizing`, `normalization`, and data `augmentation` will be applied to improve data quality and enhance model generalization.
4. A Convolutional Neural Network (`CNN`) or hybrid deep learning architecture will be used to automatically extract important features from the ultrasound images.
5. The trained model will classify the breast lesions into benign or malignant categories based on the learned features.
6. The performance of the proposed model will be evaluated using standard classification metrics such as accuracy, precision, recall, F1-score, and AUC.
7. The developed system aims to assist radiologists by providing a computer-aided diagnosis (CAD) tool that improves diagnostic accuracy and supports early detection of breast cancer.


### Project 2 : Maternal Health Risk Prediction in Personalized Patient Care

1. Data Loading and Initial Exploration: The dataset is located in the `data/Pregnancy_risk` directory. Load the Excel file containing the maternal health dataset and conduct an initial exploration to understand its structure and contents.
2. Understand the dataset structure, data types, and basic statistics for each feature.
4. The dataset may include clinical features such as:- Age, Systolic Blood Pressure as SystolicBP, Diastolic BP as DiastolicBP, Blood Sugar as BS, Body Temperature as BodyTemp, HeartRate and RiskLevel. All these are the responsible and significant risk factors for high-risk pregnancies.
5. The project is divided into following parts:-
   - `Data Preprocessing and Cleaning`: A pre-processed dataset with no missing values or inconsistencies, ready for further analysis.
   - `Data Analysis` is including visualization (e.g., histograms, scatter plots, box plots) and calculating correlations between features.
   - `Prediction Task`: Predicting Maternal Health Risk Level: Learn classification algorithms such as logistic regression, decision trees, random forests, or support vector machines, and evaluate model performance.
   - Build a classification model to predict the risk level (low, medium, or high) based on the health features provided.
6. Model Tuning and Optimization : Optimize the classification model by tuning hyperparameters and using techniques such as cross-validation.
7. For report writing, compile the analysis, and results.
   

### Project 3 : Early Detection of Coronary Heart Disease

1. The objective of this project is to develop a machine learning-based system for early prediction of CHD using patient clinical and lifestyle data. The dataset is located in the `data/Heart_disease` directory.
2. The dataset contains attributes such as `age`, `gender`, `blood pressure`, `cholesterol level`, `smoking habits`, `glucose level`, and other risk factors.
3. The project is divided into following parts:-
   - `Data Preprocessing and Cleaning`: handling missing values, feature selection, and normalization will be applied to improve data quality and model performance.
   - `Data Analysis` is including visualization (e.g., histograms, scatter plots, box plots) and calculating correlations between features.
   - `Prediction Task`: Predict the chances of heart disease in patients: Learn a classification algorithm such as logistic regression, and evaluate model performance using metrics such as `accuracy`, `precision`, `recall`, `F1-score`, and `confusion matrix`.
4. The system can assist healthcare professionals by providing `early warnings for patients at risk of coronary heart disease`.
5. The proposed model aims to support preventive healthcare and clinical decision-making, helping reduce mortality through timely diagnosis and treatment.


### Project 4 : Cervical Cancer Diagnosis Based on Machine Learning Techniques

1. The aim of this project is to develop a machine learning model for predicting the risk of cervical cancer based on patients' demographic, behavioral, and medical history data. The dataset is located in the `data/Cervical data` directory.
2. It contains medical and lifestyle information collected from 858 patients at Hospital Universitario de Caracas in Venezuela.
3. The dataset contains 36 attributes, including factors such as `age`, `number of sexual partners`, `smoking habits`, `pregnancies`, `contraceptive use`, and `sexually transmitted diseases (STDs)`.
4. Several diagnostic test results, such as `Hinselmann`, `Schiller`, `cytology`, and `biopsy` are included as target variables to determine the presence of cervical cancer.
5. The project is divided into following parts:-
   - `Data Preprocessing and Cleaning`: handling missing values, feature selection, and normalization will be applied to improve data quality and model performance.
   - `Data Analysis` is including visualization (e.g., histograms, scatter plots, box plots) and calculating correlations between features.
   - `Prediction Task`: Detection of cervical cancer: Learn classification algorithms such as Logistic Regression, Decision Tree, Random Forest, or Support Vector Machine will be used to build the prediction model, and evaluate model performance using metrics such as `accuracy`, `precision`, `recall`, `F1-score`, and `confusion matrix`.
6. The developed system can assist healthcare professionals in identifying women at high risk of cervical cancer, enabling early screening and preventive treatment.


---
## Resources
1. [Python Documentation](https://docs.python.org/3/)
2. [Class Materials](http://kgpmoodlenew.iitkgp.ac.in/moodle/)
3. Python Libraries
   1. [Matplotlib : For visualizations](https://matplotlib.org/)
   2. [Pandas : For data analysis](https://pandas.pydata.org/)
   3. [Numpy : For numerical computations](https://numpy.org/)
   4. [Scikit-learn : For predictive data analysis](https://scikit-learn.org/stable/index.html#)
   5. [Tensorflow - For building and developing machine learning model](https://www.tensorflow.org/)

---
## All the best!
