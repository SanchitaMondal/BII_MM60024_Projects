# Final Projects for MM60024 : Biomedical Imaging Informatics
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
    - [Project 1 : Multiclass Classification of Breast Cancer using Ultrasound Images](#project-1--Multiclass-Classification-of-Breast-Cancer-from-Ultrasound-Images)
    - [Project 2 : Maternal Health Risk Prediction in Personalized Patient Care](#project-2--Maternal-Health-Risk-Prediction-in-Personalized-Patient-Care)
    - [Project 3 : Early Detection of Coronary Heart Disease](#project-3--Early-Detection-of-Coronary-Heart-Disease)
    - [Project 4 : Cervical Cancer Diagnosis Based on Machine Learning Techniques](#project-4--Cervical-Cancer-Diagnosis-Based-on-Machine-Learning-Techniques)
    - [Project 5 : Prostate Cancer Diagnosis Using Machine Learning Methods](#project-5--Prostate-Cancer-Diagnosis-Using-Machine-Learning-Methods)
    - [Project 6 : Pneumonia Detection Using Chest X-ray Imaging](#project-6--Pneumonia-Detection-Using-Chest-X-ray-Imaging)
    - [Project 7 : Predicting Maternal Health Risk](#project-7--Predicting-Maternal-Health-Risk)
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
|  |  | _ |
|  |  | _ |
|  |  | _ |
|  |  | _ |
|  |  | _ |
|  |  | _ |
|  |  | _ |


## Projects

### Project 1 : Multiclass classification of breast cancer from BUS images

1. The project aims to analyse the medical transcription dataset. The dataset is located in the `data/Breast data` directory.
2. The dataset is a `csv` file. `CSV` stands for `C`omma `S`eparated `V`alues. It is a simple file format used to store tabular data, such as a spreadsheet or database. Each line of the file is a data record. Each record consists of one or more fields, separated by commas. The use of the comma as a field separator is the source of the name for this file format.
3. The dataset contains following fields : 
   - `description` : Short brief of the interaction between the patient and the doctor.
   - `medical_specialty` : Medical specialty of the issue discussed in the transcription. 
   - `sample_name` : Medical Samples used for the diagnosis.
   - `transcription` : Full transcription of the interaction between the patient and the doctor.
   - `keywords` : Keywords of the transcription
4. The project can be divided into sub-areas as follows : 
   - `Data Preprocessing`
        1. Write functions to read the csv file. Suggestion : Use the `pandas` library.
        2. This dataset needs bit of pre-processing. The `medical_specialty` field contains multiple values. You need to split the values and create a list of values. For example, if the `medical_specialty` field contains `Orthopedics, Neurology`, then you need to split it into `['Orthopedics', 'Neurology']`.
        3. The keywords field contains multiple values. You need to split the values and transform it into a list of values. For example, if the `keywords` field contains `'pain, headache, migraine'`, then you need to split it into `['pain', 'headache', 'migraine']`.
        4. Look into the dataset and find out if there are any other fields that need to be pre-processed.
   - `Data Analysis`
        1. In this part you can prepare a set of questions at least 10 and answer them using the dataset.
        2. Some examples questions to get you started:
            * What is the `most common` medical specialty?
            * What is the `most common` medical sample?
            * What is the `most common` keyword?
            * What is the `average` length of the transcription?
            * What is the `average` length of the description?
            * What is the `average` length of the keywords?
            * And so on... Get creative and come up with your own questions.
   - `Data Visualization`
        1. In this part you can make use of the `matplotlib` and `seaborn` libraries to visualize the answers to the questions you asked in the previous part.
        2. Everyone likes to see the results in the form of `graphs` and `charts`. So, make sure you visualize the answers to the questions you asked in the previous part.

### Project 2 : Agriculture Crop Production Analysis

1. This project aims to analyse the crop production data from 2006 to 2011 from all the states of India. The dataset is located in the `data/crop_production/` directory. 
2. The data directory contains 5 csv files. Go through the data files and understand the data.
3. Different data files contain different types of data. For example `datafile_1.csv` contains the following fields:
    - `Crop` : Name of the crop
    - `State` : Name of the state
    - `Cost of Cultivation (/Hectare) A2+FL` : Cost of cultivation per hectare
    - `Cost of Cultivation (/Hectare) C2` : Cost of cultivation per hectare
    - `Cost of Production (/Quintal) C2` : Cost of production per quintal
    - `Yield (Quintal/ Hectare)` : Yield per hectare
4. The `datafile_2.csv` contains the following fields:
    - `Crop` : Name of the crop
    - `Production (YYYY - YY)` : Production of the crop between two consecutive years
    - `Area (YYYY - YY)` : Area of the crop between two consecutive years
    - `Yield (YYYY - YY)` : Yield of the crop between two consecutive years
    
5. Go through the data files and understand the data. You can use the `pandas` library to read the csv files and perform analysis on the data.
6. The data files are not clean. You need to clean the data before you start analysing it.
7. The project can be divided into the following parts:
    - `Data Processing`
        1. Writing the functions for reading the data files.
        2. Once you have read the data files, you need to clean the data. You can use the `pandas` library to clean the data.
        3. Only keep the data which is relevant to the analysis and drop the rest of the data.
    - `Data Analysis`
        1. In this part, you need to prepare a set of questions and answer them using the data provided.
        2. Answer `at least 15 questions` using the data provided.
        3. A few examples questions to get you started are as follows:
            - Which `crop` has the `highest production` in the country?
            - What are the major `states` where `rice` is grown?
            - What is the `average cost of cultivation` of `rice` in the country?
            - What are seasons where `Sunflower` is grown? (data availabe in `datafile_5.csv`)
            - What is average crop duration for `Paddy`, `Wheat` and `Maize`?
        4. You can come up with your own questions and answer them using the data provided.
    - `Data Visualization`
        1. Visualize the data using `matplotlib` or `seaborn` library.
        2. Visualizing the data will help you understand the data better and answer the questions.


### Project 3 : ISBI 2022 Accepted Submissions Analysis

1. The project aims to analyse the accepted submissions of ISBI 2022. The dataset is located in the `data/isbi2022/` directory. 
2. The dataset comprised of multiple `json` files. `JSON` stands for `J`avaScript `O`bject `N`otation. It is a lightweight data-interchange format. It is easy for humans to read and write. 
3. Each json file contain the information about multiple papers(about 100 papers in each). The information about the paper is stored in the form of key-value pairs. `JSON` is all about key-value pairs (aka `dictionaries` in Python).
4. Each paper contains more than 20 attributes, but the most useful attributes are listed as follows : 
   - `articleTitle` : Title of the paper
   - `authors` : List of authors of the paper
   - `citationCount` : Number of citations of the paper
   - `downloadCount` : Number of downloads of the paper
   -  `startPage` : Starting page of the paper
   -  `endPage` : Ending page of the paper
   -  `abstract` : Stripped abstract of the paper
5. The project can be divided into sub-areas as follows : 
   - `Data Preprocessing`
        1. Write functions to read to multiple json files and concatenate them into a single dataframe.
        2. Also only keep the useful attributes mentioned above and drop the rest.
   - `Data Analysis`
        1. In this part you can prepare a set of questions at least 15 and answer them using the dataset.
        2. Some examples questions to get you started:
            * On which `area` of ISBI 2022, the most number of papers were submitted?
            * Which are the `top 10` downloaded papers and what are they about?
            * Which are the `top 10` cited papers and what are they about?
            * What are the `mean` and `median` number of `authors` per paper?
            * Most common words in the abstracts of the papers? Form a `word cloud`.
            * What is the `average` number of `pages` per paper?
            * And so on... Get creative and come up with your own questions.
   - `Data Visualization`
        1. In this part you can make use of the `matplotlib` and `seaborn` libraries to visualize the answers to the questions you asked in the previous part.
        2. Everyone likes to see the results in the form of `graphs` and `charts`. So, make sure you visualize the answers to the questions you asked in the previous part.
   

### Project 4 : Medical Image Visualization and Analysis

1. The project aims to read, visualize and analyze the medical images. The dataset is located in the `data/medical_images/` directory.
2. The dataset contains medical images of `MRI` and `CT` scans for different anatomical parts of the body. It also contains the `segmentation masks` for the images.
3. The dataset has
   1. `Hippocampus MRI` images and segmentation masks
   2. `Heart MRI` images and segmentation masks
   3. `Prostate MRI` images and segmentation masks
   4. `Abdomen CT` images and segmentation masks
4. These scans are used to diagnose the diseases of the body. The segmentation masks are used to identify the different parts of the body in the images.
5. Scans are in `NIFTI` format. `NIFTI` is a standard format for storing medical images.
6. All the scans are `3D Volumes`. Each 3D volume is a stack of `2D images`. Each 2D image is called a `slice`.
7. Your first task is to read the images and visualize them. You can use the `nibabel` library to read the images.
8. Visualizing the images is important to understand the data. You can use the `matplotlib` library to visualize the images. Visualization can be done in multiple ways. You can visualize the images in the following ways:
   1. Visualize the `slices` of the images and the segmentation masks.
   2. Visualize the `3D volumes` of the images and the segmentation masks.
9. The next task is to analyze the images. You can use the `numpy` library to analyze the images. The analysis part is open ended. 
10. You can perform simple statistical analysis on the images. You can also perform more complex analysis like `image segmentation` and `image classification`.
11. Statistical analysis may include the following:
    1. Calculate the `mean`, `median`, `standard deviation`, `minimum` and `maxmum` for the whole image, segmented image.
    2. Now compare the statistics of the segmented image with the whole image. What do you observe?
12. Complex analysis may include the following:
    1. Perform `image segmentation` on the images. You can use the `scikit-image` library to perform image segmentation.
    2. Perform `image classification` on the images. You can use the `scikit-learn` library to perform image classification.
    3. You can also perform `image registration` on the images. You can use the `SimpleITK` library to perform image registration. 
12. Try with statistical analysis first and then move on to more complex analysis. Although, we do not expect you to perform complex analysis, you can try it if you want to.
13. Remember, the analysis part is open ended. You can come up with your own analysis ideas and implement them.

### Project 5 : Patient Health Statistical Analysis

1. Create a .csv file which contain the following information

      - create an attribute with name `patient` Ten names of your friends or Random names-`string` format
      - add the attribute `patient Identifier` and assign 1 to 10 digits for each person-integer format.
      - add the attribute `Height` and add the respective heights in `float` format

            {5.5,5.6,6.1,6.1,6.0,5.9,5.8,5.8,5.8,9.1}  Float format

      - add the attribute `Temperature` and add the respective heights in `float` format

            {97.2,97.3,97.8,98,98.1,98.2,97.3,98,101,102}  Float format

      - add the attribute '`disease` and assign the following as per their patient identifier

            Randomly assign the disease to patients with the following
              {Headeach ,cold ,fever}

      - add the attribute `Hospital` and assign the following as per the patient identifier randomly.

      - add the attribute `Cost` and assign the following as per the patient identifier randomly.

            {20.0,1000.0,800.0,910.0,950.0,980.0,990.0,890.0,880.0,930.0}  Float format

2. Obtain the statistics from the dataset you created
      - Now create the class to represent the same above data
      - create the methods to calculate the statistics
            
            Mean ,Median ,Mode of 'Height'
            Mean ,Median ,Mode of 'Cost'
            Mean ,Median ,Mode of 'Temperature'
3. comment on the statistic calculations and clearly mention your observation 
          
         Note: This section may hold high weightage so write the observations in short and specific to point.

### Project 6 : Impact of Soil Quality on Crop Growth Analysis

1. This project aims to explore how soil quality (nutrients, pH) and weather conditions (temperature, humidity, rainfall) influence different crops. The dataset is located in the `data/crop_growth/` directory.
2. By analyzing this data, you will identify patterns and insights that can help to classify the type of crop grown in different regions based on these soil and weather factors.
3. Go through the data files and understand the data. You can use the `pandas` library to read the csv files and perform analysis on the data.
4. The project can be divided into the following parts:
   - `Data Preprocessing`
     * `Handling Missing Values`: a) Check for missing values in soil nutrient data (nitrogen, phosphorus, potassium), pH, temperature, humidity, and rainfall. 
        b) Use imputation techniques like mean or median imputation for continuous variables such as nutrient levels, pH, and weather data.
     * `Feature Scaling`: a) Normalize the soil nutrient levels, pH, temperature, humidity, and rainfall to bring them onto a comparable scale.
        This is important for analysis and visualization purposes.
     * `Encoding Categorical Data`: a) Convert the categorical feature crop label into numerical form using one-hot encoding or label encoding,
        so that it can be included in the analysis.
    - `Data Analysis`
      * `Descriptive Statistics`: a) Compute summary statistics (mean, median, min, max) for all continuous features (soil nutrients, pH, temperature, humidity, rainfall)           to understand the distribution of these variables for different crop types.
      * `Correlation Analysis`: a) Examine the correlation between features (soil nutrients, pH, temperature, humidity, rainfall) to understand how soil and weather     
         conditions relate to each other. Use correlation matrices to visualize these relationships.
    - `Data Visualization`
      * `Feature Distributions`: a) Use histograms or box plots to visualize the distribution of key features (nitrogen, phosphorus, potassium, pH, temperature, humidity,           rainfall) across different crop types. This will help in understanding which factors are more prevalent for certain crops.
      * `Pair Plots`: a) Generate pair plots to visualize relationships between different features for each crop type. Pair plots can show how soil nutrients and weather           conditions are distributed across crops and how they might interact with each other.
      * `Heatmap for Correlation`: a) Create a heatmap to show the correlation between features (soil nutrients, pH, temperature, humidity, and rainfall).
         This will highlight strong correlations between variables that may influence crop type classification.
      * `Bar Charts for Crop Type Distribution`: a) Use bar charts to visualize the distribution of the crop type across different ranges of soil and weather features.
         For example, you can create bar charts showing the count of each crop type at various pH levels or nitrogen concentrations.
      * `Scatter Plots`: a) Plot scatter plots with soil nutrients (e.g., nitrogen vs. phosphorus) on the axes, color-coded by crop type, to visually assess which crops             prefer specific soil nutrient combinations.
5. You can use the `pandas` library for handling missing data, imputation, encoding, and descriptive statistics.
6. For scaling features and encoding categorical data, you can use `scikit-learn` library.
7. The `seaborn` and `matplotlib` library is used for creating visualizations like pair plots, heatmaps, bar charts, scatter plots, and histograms.
8. You can try to develop a classification model to predict crop types based on soil and weather features and evaluate their performance. Although, we do not expect you to build the classification model, you can try it if you want to.
        
### Project 7 : Plant Disease Analysis Using Leaf Image Data

1. The aim of this project is to identify plant diseases from images of leaves using image processing and machine learning techniques. The dataset is located in the `data/plant_disease/` directory.
2. By analyzing the key visual differences between healthy and diseased plant leaves, the project aims to automate disease detection, which can help farmers and agricultural experts identify problems early and take action to improve crop health.
3. This project can be divided into the following parts:
   - `Image Processing`
     * `Image Resizing`: Resize all images to a uniform dimension (e.g., 128x128 or 256x256 pixels) to ensure consistent input size across the dataset.
       This step ensures that the images are compatible with deep learning models, which require fixed input dimensions.
       You can use `OpenCV` or `PIL` (Python Imaging Library) to resize the images.
     * `Normalization`: Scale pixel values from a range of 0-255 to 0-1. This ensures that the features (pixel values) are on a comparable scale, which aids in faster and better model convergence during training.
   - `Data Analysis`
     * `Statistical Analysis`: Perform descriptive statistical analysis of the dataset to explore the distribution of images across different disease categories. This includes computing the number of samples per disease class and assessing whether there are any class imbalances (i.e., some diseases having significantly more samples than others).
     * `Pixel Intensity Analysis`: Examine the pixel intensity values of the images to identify key differences between healthy and diseased leaves.
       This analysis involves comparing pixel distributions (e.g., histograms of pixel values) for each class to see if certain patterns (such as darker or lighter regions) are characteristic of diseases.
     * You can use `pandas` to analyze class distributions and check for imbalances.
   - `Data Visualization`
     * `Image Grid Visualization`: Create grids of sample images for each disease class to visually assess the variations within and across the classes.
        This helps in identifying visual patterns, such as texture or color differences, that may be indicative of disease. To create image grids, you can use `matplotlib` library.
     * `Dimensionality Reduction (PCA or t-SNE)`: Apply dimensionality reduction techniques such as PCA (Principal Component Analysis) or t-SNE (t-distributed Stochastic Neighbor Embedding) to reduce the high-dimensional image data to 2D or 3D space. This allows for visualization of the relationships between images and how different diseases cluster together. You can use `scikit-learn` library to perform PCA or t-SNE for feature reduction.
4. Remember, the analysis part is open-ended. You can come up with your own analysis ideas and implement them.
5. You can try to develop a deep learning model for plant disease detection and evaluate their performance. Although, we do not expect you to build the detection model, you can try it if you want to.

### Project 8 : PCOS Data Analysis and Visualization

1. The aim of this project is to analyze data related to Polycystic Ovary Syndrome (PCOS), a hormonal disorder affecting women of reproductive age.
2. The analysis will explore the relationship between various clinical, demographic, and lifestyle factors with the occurrence of PCOS.
3. This project also aims to identify patterns that may help in early detection and understanding of contributing factors for PCOS.
4. The dataset is located in the `data/pcos_data/` directory. The dataset may include clinical and demographic features such as:
   - Clinical Features: BMI, blood pressure, FSH/LH ratio, menstrual cycle length, etc.
   - Demographic Features: Age, marital status.
   - Lifestyle Factors: Fast food(Yes/No), Regular exercise habits.
   - Labels: PCOS (Yes/No).
5. This project is divided into the following parts:
   - `Data Preprocessing`
     * ` Handling Missing Values` : Identify and handle any missing or NaN values in the dataset using imputation or removal methods. You can use `Pandas` and `Scikit-learn` library.
     * ` Data Balancing`: If there is a class imbalance (e.g., significantly more non-PCOS cases), apply techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes using `imblearn.over_sampling` library.
   - `Data Analysis`
     * `Descriptive Analysis`: Compute summary statistics (mean, median, standard deviation) for clinical features such as BMI, and blood glucose.
     * `Correlation Analysis`: Assess relationships between features and their influence on PCOS. Use correlation matrices to visualize these relationships. You can use `seaborn` library for visualization.
     * `Statistical Testing`: Use hypothesis testing (e.g., Chi-Square Test, ANOVA) to examine the association between categorical variables (e.g., Pregnant) and the presence of PCOS. The `scipy.stats` can be used for this testing.
   - `Data Visualization`
     * `Feature Distributions`: Visualize the distribution of key features like BMI, and blood pressure across PCOS and non-PCOS groups using histograms or box plots.
     * `Clustering Analysis`: Use clustering techniques like KMeans or hierarchical clustering to identify patterns or groups within the dataset. You can use `sklearn.cluster` for kmeans clustering.
     * `Feature Importance Visualization`: Use a machine learning model like a Decision Tree or Random Forest to identify the most important features contributing to PCOS and visualize them using bar plots by using `scikit-learn` library.
6. Remember, the analysis part is open-ended. You can come up with your own analysis ideas and implement them.

### Project 9 : Mushroom Edibility Classification Using Data Analysis and Visualization

1. The goal of this project is to analyze a dataset containing descriptions of various mushroom species, focusing on the identification of features that distinguish between edible and poisonous mushrooms. Find the dataset at - https://archive.ics.uci.edu/dataset/73/mushroom
2. The project will involve handling missing values, performing descriptive and correlation analyses, and generating insightful visualizations to understand which characteristics are most associated with edibility or toxicity.
3. This project is divided into the following parts:
   - `Data Preprocessing`:
    * `Handling Missing Values`: Identify and treat missing values by using imputation techniques such as mode imputation, given the categorical nature of the dataset.
    * `Encoding Categorical Data`: Convert all categorical variables to numerical format (e.g., one-hot encoding or label encoding) to enable analysis using the `Scikit-learn` library.
    * `Balancing Classes`: If the dataset shows an imbalance in edible vs. poisonous mushrooms, use resampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance classes using 'imblearn' library.
  - `Data Analysis`:
    * `Descriptive Statistics`: Calculate summary statistics to identify common features associated with edible and poisonous mushrooms.
    * `Correlation Analysis`: Compute correlations among categorical features to identify potential relationships or patterns with mushroom edibility.
    * `Chi-Square Tests`: Perform chi-square tests on categorical variables to determine which features are statistically significant in distinguishing between poisonous and edible mushrooms using `scipy.stats` library.
  - `Data Visualization`:
    * `Feature Distributions`: Create bar plots and box plots to illustrate the distribution of individual features (e.g., cap shape, odor, gill size) across edible and poisonous mushrooms utilizing `matplotlib` library.
    * `Correlation Matrix Heatmap`: Display a heatmap of correlations between features to visually identify relationships among attributes and potential indicators of edibility.
    * `Pair Plot`: Use pair plots to see how combinations of features vary between edible and poisonous mushrooms, which can reveal unique patterns using python libraries.
    * `Decision Tree Visualization`: If a decision tree model is used, visualize the tree to illustrate decision-making criteria and the most relevant features associated with edibility.
4. Remember, the analysis part is open-ended. You can come up with your own analysis ideas and implement them.

### Project 10 : Diabetes Type Classification

1. Extract the diabetes data from the compressed file (`diabetes-data.tar.Z`) and load it into a suitable format (e.g., CSV, Pandas DataFrames). You can find the dataset at - https://archive.ics.uci.edu/dataset/34/diabetes
2. Learn how to work with archived and compressed files, data extraction, and loading datasets.
3. Understand the dataset structure, variable types, missing values, and basic statistics.
4. A summary report of data types, distributions, and any missing values, along with a preliminary understanding of the dataset.
6. The dataset may include clinical features such as:- it contains the distribution for 70 sets of data recorded on diabetes patients (several weeks `to months` worth of glucose, insulin, and lifestyle data per patient + a description of the problem domain).
7. using this dataset we can use for the  Diabetes Type Classification: For datasets including Type 1 and Type 2 diabetes patients, you could classify patients by
diabetes type based on their glucose and insulin response patterns.
8. The project is divided into following parts:
   - `Data Preprocessing and Cleaning`: Clean and preprocess the data to make it suitable for analysis. This includes handling missing values, outliers, and any
necessary transformations (e.g., normalizing glucose levels).
   - `Exploratory Data Analysis (EDA)`: Conduct exploratory data analysis to identify patterns and relationships among features such as glucose, insulin, and lifestyle
factors.
   - `Classification Task`: Define blood glucose levels as categorical variables (e.g., low, normal, high) and build a classification model to predict these levels.
   - `Regression Task`: Predicting Continuous Glucose Levels -- Build a regression model to predict continuous glucose levels based on lifestyle and insulin data.
   - `Model Evaluation and Tuning`:- Fine-tune the models built (classification, regression, or time-series) and evaluate their performance on a test dataset.
10. Compile the analysis, results, and insights into a structured report and prepare a presentation.

### Project 11: Predicting Maternal Health Risk

1. Data Loading and Initial Exploration: Load the Excel file containing the maternal health dataset and conduct an initial exploration to understand its structure and contents.
2. Understand the dataset structure, data types, and basic statistics for each feature.
3. A brief summary report with initial findings on data types, any missing values, and basic statistical measures (mean, median, etc.) for each feature.
4. The dataset link ( https://archive.ics.uci.edu/dataset/863/maternal+health+risk ) is provided.
5. The dataset may include clinical features such as:- Age, Systolic Blood Pressure as SystolicBP, Diastolic BP as DiastolicBP, Blood Sugar as BS, Body Temperature as BodyTemp, HeartRate and RiskLevel. All these are the responsible and significant risk factors for maternal mortality, that is one of the main concern of SDG of UN.
6. The project is divided into following parts:-
   - `Data Preprocessing and Cleaning`: A pre-processed dataset with no missing values or inconsistencies, ready for further analysis.
   - `Data Analysis` is including visualization (e.g., histograms, scatter plots, box plots) and calculating correlations between features.
   - `Classification Task`: Predicting Maternal Health Risk Level: Learn classification algorithms such as logistic regression, decision trees, random forests, or support vector machines, and evaluate model performance.
   - Build a classification model to predict the risk level (low, medium, or high) based on the health features provided.
7. Model Tuning and Optimization : Optimize the classification model by tuning hyperparameters and using techniques such as cross-validation.
8. For report writing, compile the analysis, and results.

---
## Resources
1. [Python Documentation](https://docs.python.org/3/)
2. [Class Code Materials](https://github.com/SanchitaMondal/Introduction-to-programming-MM62201/)
3. [Introduction to Computation and Programming Using Python](https://mitpress.mit.edu/9780262542364/introduction-to-computation-and-programming-using-python/)
4. [Elements of Programming Interviews in Python]()
5. Python Libraries
   1. [Matplotlib : For awesome visualizations](https://matplotlib.org/)
   2. [Pandas : For data analysis](https://pandas.pydata.org/)
   3. [Numpy : For numerical computations](https://numpy.org/)

---
## All the best!
