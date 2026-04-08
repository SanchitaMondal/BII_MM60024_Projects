# -*- coding: utf-8 -*-
"""
====================================================================
  Explainable Survival Prediction of Liver Cirrhosis Patients
====================================================================
  Mathematics-first implementation — minimal library dependencies.

  Model  : Survival SVM  (pairwise ranking, squared-hinge loss)
           trained with mini-batch SGD + gradient clipping

  XAI    : Exact Linear SHAP
           phi_j(x) = w_j * (x_j - mu_j_background)

  Data   : cirrhosis.csv  (418 patients, 20 columns)
====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
SAVE_DIR = "cirrhosis_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  PART 1  DATA LOADING AND PREPROCESSING
# ══════════════════════════════════════════════════════════════════
#
#  Missing-value imputation:
#      x_hat_k = median(X_k)   robust to skewed lab-value distributions
#
#  Z-score normalisation:
#      z = (x - mu) / sigma    maps each feature to zero-mean unit-variance

def load_and_preprocess(path):
    df = pd.read_csv(path)
    print(f"[INFO] Raw data shape : {df.shape}")

    df = df.drop(columns=["ID"])

    # encode categoricals
    binary_map = {"Y": 1.0, "N": 0.0, "M": 1.0, "F": 0.0}
    edema_map  = {"N": 0.0, "S": 0.5, "Y": 1.0}
    drug_map   = {"D-penicillamine": 1.0, "Placebo": 0.0}

    df["Sex"]          = df["Sex"].map(binary_map)
    df["Ascites"]      = df["Ascites"].map(binary_map)
    df["Hepatomegaly"] = df["Hepatomegaly"].map(binary_map)
    df["Spiders"]      = df["Spiders"].map(binary_map)
    df["Edema"]        = df["Edema"].map(edema_map)
    df["Drug"]         = df["Drug"].map(drug_map)

    # binary survival label:  +1 = deceased,  -1 = censored / transplanted
    df["label"] = df["Status"].apply(lambda s: 1.0 if s == "D" else -1.0)
    df = df.drop(columns=["Status"])

    FEATURE_COLS = [
        "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
        "Bilirubin", "Cholesterol", "Albumin", "Copper",
        "Alk_Phos", "SGOT", "Tryglicerides", "Platelets",
        "Prothrombin", "Stage", "Drug", "N_Days"
    ]

    # impute with column median (pandas 2.x CoW-safe reassignment)
    for col in FEATURE_COLS:
        n_miss = int(df[col].isnull().sum())
        if n_miss > 0:
            med = float(df[col].median())
            df[col] = df[col].fillna(med)
            print(f"   Imputed {n_miss:3d} missing  ->  {col:15s}  (median={med:.3f})")

    X = df[FEATURE_COLS].values.astype(float)
    y = df["label"].values.astype(float)

    assert not np.isnan(X).any(), "NaN still present after imputation!"
    print(f"[INFO] Final feature matrix : {X.shape}")
    print(f"[INFO] Class distribution   : +1(deceased)={(y==1).sum()}"
          f"  -1(censored/CL)={(y==-1).sum()}")

    # z-score normalisation
    mu    = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    X_norm = (X - mu) / sigma

    return X_norm, y, FEATURE_COLS, mu, sigma, df, FEATURE_COLS


# ══════════════════════════════════════════════════════════════════
#  PART 2  EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════

def plot_eda(df_work, feature_names):
    continuous = [
        "Bilirubin", "Cholesterol", "Albumin", "Copper",
        "Alk_Phos", "SGOT", "Tryglicerides", "Platelets",
        "Prothrombin", "Age"
    ]

    # 2-A  Histograms
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, continuous):
        vals = df_work[col].dropna().values
        counts, bins = np.histogram(vals, bins=25)
        ax.bar(bins[:-1], counts, width=np.diff(bins),
               edgecolor="white", color="#4C72B0", alpha=0.85)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("Value"); ax.set_ylabel("Count")
    fig.suptitle("Feature Distributions (Histograms)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/1_histograms.png", dpi=150)
    plt.close()
    print("[PLOT] Saved 1_histograms.png")

    # 2-B  Box-plots by label
    deceased = df_work["label"] == 1
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.ravel()
    for ax, col in zip(axes, continuous):
        g_dead  = df_work.loc[deceased,  col].dropna().values
        g_alive = df_work.loc[~deceased, col].dropna().values
        bp = ax.boxplot([g_dead, g_alive], patch_artist=True,
                        labels=["Deceased", "Alive/CL"],
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], ["#DD4949", "#4CAF50"]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_title(col, fontsize=10, fontweight="bold")
    fig.suptitle("Feature Box-Plots by Survival Status", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/2_boxplots.png", dpi=150)
    plt.close()
    print("[PLOT] Saved 2_boxplots.png")

    # 2-C  Scatter: Bilirubin vs Albumin
    fig, ax = plt.subplots(figsize=(7, 5))
    for lbl, color, marker, label in [
        (1,  "#DD4949", "x", "Deceased"),
        (-1, "#4CAF50", "o", "Alive/CL")
    ]:
        mask = df_work["label"] == lbl
        ax.scatter(df_work.loc[mask, "Bilirubin"],
                   df_work.loc[mask, "Albumin"],
                   c=color, marker=marker, alpha=0.6, s=35, label=label)
    ax.set_xlabel("Bilirubin (mg/dL)"); ax.set_ylabel("Albumin (g/dL)")
    ax.set_title("Bilirubin vs Albumin coloured by Survival", fontweight="bold")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/3_scatter_bili_albumin.png", dpi=150)
    plt.close()
    print("[PLOT] Saved 3_scatter_bili_albumin.png")

    # 2-D  Pearson correlation heat-map
    #   r_ij = sum[(x_i - mu_i)(x_j - mu_j)] / (n * sigma_i * sigma_j)
    num_cols = continuous + ["Prothrombin", "Stage"]
    sub = df_work[num_cols].dropna().values
    Z_c = sub - sub.mean(axis=0)
    norms = np.linalg.norm(Z_c, axis=0, keepdims=True) + 1e-8
    corr = (Z_c / norms).T @ (Z_c / norms)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Pearson Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/4_correlation_heatmap.png", dpi=150)
    plt.close()
    print("[PLOT] Saved 4_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════
#  PART 3  SURVIVAL SVM
# ══════════════════════════════════════════════════════════════════
#
#  Objective: find w in R^p such that:
#      f(x_i) > f(x_j)  for all (i,j) in P  (deceased before censored)
#
#  Squared-hinge ranking loss:
#      L(w) = lambda * ||w||^2
#             + (1/|P|) * sum_{(i,j) in P} max(0, 1 - Delta_f_ij)^2
#      where Delta_f_ij = w^T (x_i - x_j)
#
#  Gradient:
#      dL/dw = 2*lambda*w  -  (2/B) * sum_{violated} h_ij * (x_i - x_j)
#      where h_ij = max(0, 1 - Delta_f_ij)
#
#  Update rule (SGD + decaying step + gradient clipping):
#      w <- w - eta_t * dL/dw  ,   eta_t = eta_0 / sqrt(t+1)
#      if ||dL/dw|| > clip_norm:  scale gradient to clip_norm

class SurvivalSVM:

    def __init__(self, lam=1e-3, lr=0.05, n_iter=300,
                 batch_pairs=512, clip_norm=5.0):
        self.lam         = lam
        self.lr          = lr
        self.n_iter      = n_iter
        self.batch_pairs = batch_pairs
        self.clip_norm   = clip_norm
        self.w           = None
        self.loss_history = []

    def _build_pairs(self, y):
        pos = np.where(y ==  1)[0]
        neg = np.where(y == -1)[0]
        ii, jj = np.meshgrid(pos, neg, indexing="ij")
        return ii.ravel(), jj.ravel()

    def _loss(self, X, y):
        pi, pj  = self._build_pairs(y)
        margins = (X[pi] - X[pj]) @ self.w
        return (self.lam * float(np.dot(self.w, self.w)) +
                float(np.mean(np.maximum(0.0, 1.0 - margins) ** 2)))

    def fit(self, X, y):
        n, p = X.shape
        self.w = np.zeros(p)
        all_pi, all_pj = self._build_pairs(y)
        n_pairs = len(all_pi)

        print(f"\n[SVM] {n} patients  |  {n_pairs} ranking pairs")
        print(f"      lambda={self.lam}  eta_0={self.lr}  epochs={self.n_iter}")

        for t in range(self.n_iter):
            idx = np.random.choice(n_pairs,
                                   size=min(self.batch_pairs, n_pairs),
                                   replace=False)
            pi, pj = all_pi[idx], all_pj[idx]
            diff   = X[pi] - X[pj]                    # (B, p)
            scores = diff @ self.w                     # (B,)

            # h = max(0, 1 - score)  (hinge residual)
            h    = np.maximum(0.0, 1.0 - scores)
            grad = (2.0 * self.lam * self.w
                    - (2.0 / len(pi)) * (diff.T @ h))

            # gradient clipping: ||grad|| <= clip_norm
            g_norm = float(np.linalg.norm(grad))
            if g_norm > self.clip_norm:
                grad = grad * (self.clip_norm / g_norm)

            # decaying step: eta_t = eta_0 / sqrt(t+1)
            self.w -= (self.lr / np.sqrt(t + 1)) * grad

            if t % 10 == 0:
                loss = self._loss(X, y)
                self.loss_history.append(loss)
                if t % 50 == 0:
                    print(f"   epoch {t:4d}  loss={loss:.5f}")

        loss = self._loss(X, y)
        self.loss_history.append(loss)
        print(f"   epoch {self.n_iter:4d}  loss={loss:.5f}  [done]")
        return self

    def decision_function(self, X):
        return X @ self.w

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)


# ══════════════════════════════════════════════════════════════════
#  PART 4  EVALUATION
# ══════════════════════════════════════════════════════════════════
#
#  Harrell's C-index:
#      C = |{(i,j) in P : risk_i > risk_j}| / |P|
#      0.5 = random,  1.0 = perfect
#
#  ROC / AUC: sweep thresholds tau
#      TPR(tau) = TP(tau) / P
#      FPR(tau) = FP(tau) / N
#      AUC = integral_0^1 TPR dFPR  (trapezoidal rule)

def concordance_index(scores, labels):
    pos = np.where(labels ==  1)[0]
    neg = np.where(labels == -1)[0]
    concordant = total = 0
    for i in pos:
        for j in neg:
            total += 1
            if scores[i] > scores[j]:
                concordant += 1
            elif scores[i] == scores[j]:
                concordant += 0.5
    return concordant / total if total > 0 else 0.0


def roc_curve_manual(scores, labels):
    thresholds = np.sort(np.unique(scores))[::-1]
    P = int((labels ==  1).sum())
    N = int((labels == -1).sum())
    fprs, tprs = [0.0], [0.0]
    for tau in thresholds:
        pred = scores >= tau
        tprs.append(float(((pred) & (labels ==  1)).sum()) / P)
        fprs.append(float(((pred) & (labels == -1)).sum()) / N)
    fprs.append(1.0); tprs.append(1.0)
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    auc  = float(np.trapezoid(tprs, fprs))
    return fprs, tprs, auc


def evaluate(model, X_tr, y_tr, X_te, y_te):
    print("\n" + "=" * 55)
    print("  MODEL EVALUATION")
    print("=" * 55)
    for name, X, y in [("Train", X_tr, y_tr), ("Test", X_te, y_te)]:
        preds  = model.predict(X)
        scores = model.decision_function(X)
        acc    = float((preds == y).mean())
        ci     = concordance_index(scores, y)
        _, _, auc = roc_curve_manual(scores, y)
        print(f"  {name:5s}  Accuracy={acc:.3f}  C-index={ci:.3f}  AUC={auc:.3f}")


def plot_evaluation(model, X_te, y_te):
    fprs, tprs, auc = roc_curve_manual(model.decision_function(X_te), y_te)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    iters = [i * 10 for i in range(len(model.loss_history))]
    ax1.plot(iters, model.loss_history, color="#4C72B0", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Squared-Hinge Loss")
    ax1.set_title("Training Loss Convergence", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(fprs, tprs, color="#DD4949", linewidth=2,
             label=f"SurvSVM  AUC={auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random  AUC=0.500")
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve (Test Set)", fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/5_loss_and_roc.png", dpi=150)
    plt.close()
    print("[PLOT] Saved 5_loss_and_roc.png")


# ══════════════════════════════════════════════════════════════════
#  PART 5  EXPLAINABILITY  — Linear SHAP
# ══════════════════════════════════════════════════════════════════
#
#  Shapley values (cooperative game theory):
#      phi_j = sum_{S subset F\{j}}
#                [|S|! * (p-|S|-1)! / p!] * [f(S u {j}) - f(S)]
#
#  For linear model f(x) = w^T x  this simplifies exactly to:
#      phi_j(x) = w_j * (x_j - mu_j)
#  where mu_j = E_train[X_j]  is the background (absence) value.
#
#  Shapley axioms satisfied:
#    Efficiency : sum_j phi_j(x) = f(x) - f(mu)
#    Symmetry   : equal marginal contributions  =>  equal phi
#    Dummy      : w_j = 0  =>  phi_j = 0  for all x
#    Linearity  : attributions add across models

class LinearSHAP:

    def __init__(self, weights, background_mean):
        self.w  = weights           # (p,)
        self.mu = background_mean   # E_train[X], shape (p,)

    def shap_values(self, X):
        # phi_ij = w_j * (x_ij - mu_j)   shape (n, p)
        return (X - self.mu) * self.w

    def base_value(self):
        # E[f(X)] = w^T mu
        return float(self.w @ self.mu)


def plot_shap(explainer, X_te, feature_names):
    phi      = explainer.shap_values(X_te)   # (n_test, p)
    base     = explainer.base_value()
    mean_abs = np.abs(phi).mean(axis=0)       # (p,)
    order    = np.argsort(mean_abs)[::-1]
    top_k    = 12

    # 5-A  Global bar chart
    names_top = [feature_names[i] for i in order[:top_k]]
    vals_top  = mean_abs[order[:top_k]]
    colors    = plt.cm.RdBu_r(np.linspace(0.1, 0.9, top_k))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(top_k), vals_top[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names_top[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/6_shap_global_bar.png", dpi=150)
    plt.close()
    print("[PLOT] Saved 6_shap_global_bar.png")

    # 5-B  Beeswarm dot plot
    fig, ax = plt.subplots(figsize=(10, 7))
    for row_pos, feat_idx in enumerate(order[:top_k][::-1]):
        sv   = phi[:, feat_idx]
        fv   = X_te[:, feat_idx]
        fmin = float(fv.min()); fmax = float(fv.max())
        norm = (fv - fmin) / (fmax - fmin + 1e-8)
        jitter = np.random.uniform(-0.2, 0.2, len(sv))
        sc = ax.scatter(sv, np.full_like(sv, row_pos) + jitter,
                        c=norm, cmap="RdBu_r", alpha=0.6, s=18,
                        vmin=0, vmax=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in order[:top_k][::-1]], fontsize=9)
    ax.set_xlabel("SHAP value  (impact on risk score)", fontsize=11)
    ax.set_title("SHAP Beeswarm — Test Set", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025)
    cbar.set_label("Norm. feature value (blue=low, red=high)", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/7_shap_beeswarm.png", dpi=150)
    plt.close()
    print("[PLOT] Saved 7_shap_beeswarm.png")

    # 5-C  Force plots for 3 highest-risk patients
    risk = X_te @ explainer.w
    top3 = np.argsort(risk)[-3:][::-1]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, pt in zip(axes, top3):
        phi_pt = phi[pt]
        top10  = np.argsort(np.abs(phi_pt))[::-1][:10]
        sv     = phi_pt[top10]
        nm     = [feature_names[i] for i in top10]
        cols   = ["#DD4949" if v > 0 else "#4C72B0" for v in sv]
        ax.barh(range(len(sv)), sv, color=cols, edgecolor="white")
        ax.set_yticks(range(len(sv))); ax.set_yticklabels(nm, fontsize=9)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(
            f"Patient #{pt}  |  Risk score={risk[pt]:.3f}"
            f"  |  Base value={base:.3f}",
            fontsize=10, fontweight="bold"
        )
        ax.set_xlabel("SHAP value  (red = increases risk, blue = decreases risk)",
                      fontsize=9)
    fig.suptitle("Individual SHAP Force-Plots  (Top-3 Highest Risk Patients)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/8_shap_individual_force.png", dpi=150)
    plt.close()
    print("[PLOT] Saved 8_shap_individual_force.png")


# ══════════════════════════════════════════════════════════════════
#  UTILITY : stratified train / test split
# ══════════════════════════════════════════════════════════════════

def stratified_split(X, y, test_ratio=0.20):
    pos = np.where(y ==  1)[0]; np.random.shuffle(pos)
    neg = np.where(y == -1)[0]; np.random.shuffle(neg)
    n_pos_te = int(len(pos) * test_ratio)
    n_neg_te = int(len(neg) * test_ratio)
    te_idx = np.concatenate([pos[:n_pos_te], neg[:n_neg_te]])
    tr_idx = np.concatenate([pos[n_pos_te:], neg[n_neg_te:]])
    return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  Explainable Survival Prediction  -- Liver Cirrhosis")
    print("=" * 60)

    DATA_PATH = "cirrhosis.csv"
    X, y, feat_names, mu_raw, sigma_raw, df_work, feat_cols = \
        load_and_preprocess(DATA_PATH)

    print("\n[EDA] Generating exploratory plots ...")
    plot_eda(df_work, feat_names)

    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_ratio=0.20)
    print(f"\n[SPLIT] Train={len(y_tr)}  Test={len(y_te)}")

    svm = SurvivalSVM(lam=5e-4, lr=0.08, n_iter=300,
                      batch_pairs=512, clip_norm=5.0)
    svm.fit(X_tr, y_tr)

    evaluate(svm, X_tr, y_tr, X_te, y_te)
    plot_evaluation(svm, X_te, y_te)

    print("\n[SHAP] Computing exact linear SHAP values ...")
    background_mean = X_tr.mean(axis=0)
    explainer = LinearSHAP(weights=svm.w, background_mean=background_mean)
    plot_shap(explainer, X_te, feat_names)

    phi_te     = explainer.shap_values(X_te)
    importance = np.abs(phi_te).mean(axis=0)
    rank_order = np.argsort(importance)[::-1]

    print("\n" + "=" * 55)
    print("  GLOBAL FEATURE IMPORTANCE  (mean |SHAP|)")
    print("=" * 55)
    print(f"  {'Rank':<5} {'Feature':<18} {'Mean|SHAP|':>12}  {'SVM weight':>12}")
    print("  " + "-" * 52)
    for r, idx in enumerate(rank_order[:12], 1):
        print(f"  {r:<5} {feat_names[idx]:<18} {importance[idx]:>12.5f}"
              f"  {svm.w[idx]:>12.5f}")

    print(f"\n[DONE] All plots saved to '{SAVE_DIR}/'")
    print("       Files:", ", ".join(sorted(os.listdir(SAVE_DIR))))