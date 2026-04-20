
# # ================================
# # SAFE BACKEND (IMPORTANT)
# # ================================
# import matplotlib
# matplotlib.use("Agg")

# # ================================
# # IMPORTS
# # ================================
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import (
#     confusion_matrix, roc_curve, auc,
#     precision_recall_curve, average_precision_score,
#     accuracy_score, precision_score, recall_score, f1_score
# )


# # ================================
# # SETTINGS
# # ================================
# sns.set(style="whitegrid")
# OUT = "analysis_outputs_Validation_model"
# os.makedirs(OUT, exist_ok=True)

# FEATURES = [
#     "Asymmetry", "Border_Compactness", "Color_Variance",
#     "Color_Entropy", "Diameter_px", "Area_Ratio"
# ]

# # ================================
# # LOAD DATA
# # ================================
# def load_data(path):
#     df = pd.read_excel(path, skiprows=1)
#     df.columns = [
#         "Image_File", "Asymmetry", "Border_Compactness",
#         "Color_Variance", "Color_Entropy", "Diameter_px",
#         "Area_Ratio", "Cancer_type"
#     ]

#     df = df[df["Cancer_type"].isin(["Benign", "Malignant"])]

#     for col in FEATURES:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     df.dropna(inplace=True)
#     return df

# # ================================
# # SAVE
# # ================================
# def save(name):
#     plt.savefig(os.path.join(OUT, name))
#     plt.close()

# # ================================
# # MAIN
# # ================================
# def run(train_path, test_path=None):

#     print("Loading data...")
#     df = load_data(train_path)
#     df_test = load_data(test_path) if test_path else None

#     # ============================
#     # BASIC EDA
#     # ============================

#     # Combined distribution plot
#     df[FEATURES].hist(figsize=(12, 8))
#     plt.suptitle("Feature Distributions")
#     save("distributions.png")

#     # # Correlation heatmap
#     # plt.figure(figsize=(10,8))
#     # sns.heatmap(df[FEATURES].corr(), annot=True, cmap="coolwarm")
#     # plt.title("Correlation Heatmap")
#     # save("correlation.png")
    
#     # Correlation heatmap (FIXED)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(df[FEATURES].corr(), annot=True, cmap="coolwarm")

#     plt.title("Correlation Heatmap")

#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)

#     plt.tight_layout()  # <-- critical fix

#     save("correlation.png")

#     # ============================
#     # PREPROCESS
#     # ============================
#     le = LabelEncoder()
#     df["Cancer_type"] = le.fit_transform(df["Cancer_type"])

#     if df_test is not None:
#         df_test = df_test[df_test["Cancer_type"].isin(le.classes_)]
#         df_test["Cancer_type"] = le.transform(df_test["Cancer_type"])

#     X = df[FEATURES]
#     y = df["Cancer_type"]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_val, y_train, y_val = train_test_split(
#         X_scaled, y, stratify=y, test_size=0.2, random_state=42
#     )

#     # ============================
#     # MODELS
#     # ============================
#     models = {
#         "Logistic Regression": LogisticRegression(max_iter=1000),
#         "Decision Tree": DecisionTreeClassifier(max_depth=10),
#         "Random Forest": RandomForestClassifier(n_estimators=100),
#         "SVM": SVC(probability=True),
#         #"Linear Regression": LinearRegression()
#     }

#     trained = {}

#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         trained[name] = model

#     # ============================
# # MODEL COMPARISON METRICS
# # ============================
#     results = []

#     for name, model in trained.items():
#         # Predictions
#         y_pred = model.predict(X_val)

#         # Probabilities for AUC
#         y_prob = model.predict_proba(X_val)[:, 1]

#         # Metrics
#         acc = accuracy_score(y_val, y_pred)
#         prec = precision_score(y_val, y_pred)
#         rec = recall_score(y_val, y_pred)
#         f1 = f1_score(y_val, y_pred)
#         auc_score = auc(*roc_curve(y_val, y_prob)[:2])

#         results.append([name, acc, prec, rec, f1, auc_score])

#     # Convert to DataFrame
#     results_df = pd.DataFrame(results, columns=[
#         "Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"
#     ])

#     # Save table
#     results_df.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)

#     print("\nModel Comparison:")
#     print(results_df)
#     # ============================
#     # CONFUSION MATRICES
#     # ============================
#     fig, axes = plt.subplots(2, 3, figsize=(12, 8))
#     axes = axes.flatten()

#     for i, (name, model) in enumerate(trained.items()):
#         if name == "Linear Regression":
#             preds = (model.predict(X_val) > 0.5).astype(int)
#         else:
#             preds = model.predict(X_val)

#         cm = confusion_matrix(y_val, preds).astype(float)
#         cm = cm / cm.sum(axis=1, keepdims=True)

#         sns.heatmap(cm, annot=True, ax=axes[i], cmap="Blues", fmt=".2f")
#         axes[i].set_title(name)

#     plt.tight_layout()
#     save("confusion_matrices.png")
    
#     # ============================
# # CONFUSION MATRICES (FIXED)
# # ============================
#     fig, axes = plt.subplots(2, 2, figsize=(10, 8))
#     axes = axes.flatten()

#     for i, (name, model) in enumerate(trained.items()):
#         if name == "Linear Regression":
#             preds = (model.predict(X_val) > 0.5).astype(int)
#         else:
#             preds = model.predict(X_val)

#         cm = confusion_matrix(y_val, preds).astype(float)
#         cm = cm / cm.sum(axis=1, keepdims=True)

#         sns.heatmap(cm, annot=True, ax=axes[i], cmap="Blues", fmt=".2f")
#         axes[i].set_title(name)

#     plt.tight_layout()
#     save("confusion_matrices_2.png")

#     # ============================
#     # ROC CURVES
#     # ============================
#     plt.figure()

#     for name, model in trained.items():
#         if name == "Linear Regression":
#             probs = model.predict(X_val)
#         else:
#             probs = model.predict_proba(X_val)[:,1]

#         fpr, tpr, _ = roc_curve(y_val, probs)
#         plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")

#     plt.legend()
#     plt.title("ROC Curves")
#     save("roc_curves.png")

#     # ============================
#     # PR CURVES
#     # ============================
#     plt.figure()

#     for name, model in trained.items():
#         if name == "Linear Regression":
#             probs = model.predict(X_val)
#         else:
#             probs = model.predict_proba(X_val)[:,1]

#         precision, recall, _ = precision_recall_curve(y_val, probs)
#         ap = average_precision_score(y_val, probs)

#         plt.plot(recall, precision, label=f"{name} (AP={ap:.2f})")

#     plt.legend()
#     plt.title("Precision-Recall Curves")
#     save("pr_curves.png")

#     # ============================
#     # FEATURE IMPORTANCE (RF)
#     # ============================
#     rf = trained["Random Forest"]
#     importances = rf.feature_importances_

#     plt.figure()
#     sns.barplot(x=importances, y=FEATURES)
#     plt.title("Random Forest Feature Importance")
#     save("feature_importance.png")

#     # ============================
#     # DECISION TREE VISUALIZATION
#     # ============================
#     dt = trained["Decision Tree"]

#     plt.figure(figsize=(15,8))
#     plot_tree(dt, feature_names=FEATURES, filled=True)
#     plt.title("Decision Tree (max_depth=10)")
#     save("decision_tree.png")

#     # ============================
#     # TEST EVALUATION
#     # ============================
#     if df_test is not None:
#         print("\nTest Evaluation:")

#         X_test = scaler.transform(df_test[FEATURES])
#         y_test = df_test["Cancer_type"]

#         for name, model in trained.items():
#             if name == "Linear Regression":
#                 preds = (model.predict(X_test) > 0.5).astype(int)
#             else:
#                 preds = model.predict(X_test)

#             acc = (preds == y_test).mean()
#             print(f"{name}: Accuracy = {acc:.3f}")

#     print("\n✅ DONE. Clean outputs generated.")

# # ================================
# # RUN
# # ================================
# if __name__ == "__main__":
#     run("ABCDfeatures.xlsx", "Test.xlsx")\
    

# ================================
# SAFE BACKEND (IMPORTANT)
# ================================
import matplotlib
matplotlib.use("Agg")

# ================================
# IMPORTS
# ================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# ================================
# SETTINGS
# ================================
sns.set(style="whitegrid")
OUT = "analysis_outputs"
os.makedirs(OUT, exist_ok=True)

FEATURES = [
    "Asymmetry", "Border_Compactness", "Color_Variance",
    "Color_Entropy", "Diameter_px", "Area_Ratio"
]

# ================================
# LOAD DATA
# ================================
def load_data(path):
    df = pd.read_excel(path, skiprows=1)
    df.columns = [
        "Image_File", "Asymmetry", "Border_Compactness",
        "Color_Variance", "Color_Entropy", "Diameter_px",
        "Area_Ratio", "Cancer_type"
    ]

    df = df[df["Cancer_type"].isin(["Benign", "Malignant"])]

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    return df

# ================================
# SAVE
# ================================
def save(name):
    plt.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close()

# ================================
# MAIN
# ================================
def run(train_path, test_path=None):

    print("Loading data...")
    df = load_data(train_path)
    df_test = load_data(test_path) if test_path else None

    # ============================
    # BASIC EDA
    # ============================

    # Distribution plot
    df[FEATURES].hist(figsize=(12, 8))
    plt.suptitle("Feature Distributions")
    save("distributions.png")

    # Correlation heatmap (fixed)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save("correlation.png")

    # ============================
    # PREPROCESS
    # ============================
    le = LabelEncoder()
    df["Cancer_type"] = le.fit_transform(df["Cancer_type"])

    if df_test is not None:
        df_test = df_test[df_test["Cancer_type"].isin(le.classes_)]
        df_test["Cancer_type"] = le.transform(df_test["Cancer_type"])

    X = df[FEATURES]
    y = df["Cancer_type"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )
    print("\nData Split Info:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    
    # ============================
    # MODELS
    # ============================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(probability=True)
    }

    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    # ============================
    # MODEL COMPARISON METRICS
    # ============================
    results = []

    for name, model in trained.items():
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc_score = auc(*roc_curve(y_val, y_prob)[:2])

        results.append([name, acc, prec, rec, f1, auc_score])

    results_df = pd.DataFrame(results, columns=[
        "Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"
    ])

    results_df.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)

    print("\nModel Comparison:")
    print(results_df)

    # Comparison plot
    results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_melted, x="Model", y="Score", hue="Metric")
    plt.title("Model Comparison Across Metrics")
    plt.xticks(rotation=30)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save("model_comparison.png")

    # ============================
    # CONFUSION MATRICES
    # ============================
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, (name, model) in enumerate(trained.items()):
        preds = model.predict(X_val)

        cm = confusion_matrix(y_val, preds).astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm, annot=True, ax=axes[i], cmap="Blues", fmt=".2f")
        axes[i].set_title(name)
        print(confusion_matrix(y_val, preds))

    plt.tight_layout()
    save("confusion_matrices.png")

    # ============================
    # ROC CURVES
    # ============================
    plt.figure()

    for name, model in trained.items():
        probs = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")

    plt.legend()
    plt.title("ROC Curves")
    save("roc_curves.png")

   
    # PR CURVES
    
    plt.figure()

    for name, model in trained.items():
        probs = model.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, probs)
        ap = average_precision_score(y_val, probs)

        plt.plot(recall, precision, label=f"{name} (AP={ap:.2f})")

    plt.legend()
    plt.title("Precision-Recall Curves")
    save("pr_curves.png")

    
    # FEATURE IMPORTANCE (RF)
    
    rf = trained["Random Forest"]
    importances = rf.feature_importances_

    plt.figure()
    sns.barplot(x=importances, y=FEATURES)
    plt.title("Random Forest Feature Importance")
    save("feature_importance.png")

   
    # DECISION TREE VISUALIZATION
   
    dt = trained["Decision Tree"]

    plt.figure(figsize=(15, 8))
    plot_tree(dt, feature_names=FEATURES, filled=True)
    plt.title("Decision Tree (max_depth=10)")
    save("decision_tree.png")

    
    # TEST EVALUATION
  
    if df_test is not None:
        print("\nTest Evaluation:")

        X_test = scaler.transform(df_test[FEATURES])
        y_test = df_test["Cancer_type"]

        for name, model in trained.items():
            preds = model.predict(X_test)
            acc = (preds == y_test).mean()
            print(f"{name}: Accuracy = {acc:.3f}")

    print("\n✅ DONE. Clean outputs generated.")

# RUN

if __name__ == "__main__":
    run("ABCDfeatures.xlsx", "Test.xlsx")