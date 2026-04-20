import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from unet_model import UNet

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cpu")

model = UNet()
model_path = os.path.join(script_dir, "unet_skin_segmentation.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


test_data_path = os.path.join(script_dir, "Test_Data")
if os.path.exists(test_data_path):
    image_files = [f for f in os.listdir(test_data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if image_files:
        img_path = os.path.join(test_data_path, image_files[0])
    else:
        raise FileNotFoundError(f"No image files found in {test_data_path}")
else:
    raise FileNotFoundError(f"Test_Data directory not found at {test_data_path}")

img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Failed to load image from {img_path}. Check file path and integrity.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

# Save original image for display
original_img = img.copy()

img = img / 255.0
img = np.transpose(img, (2, 0, 1))

img = torch.tensor(img).float().unsqueeze(0)

with torch.no_grad():
    pred = model(img)

mask = pred.squeeze().numpy()


ground_truth_path = os.path.join(script_dir, "Test_GroundTruth")
gt_mask = None
metrics = {}


base_filename = os.path.splitext(os.path.basename(img_path))[0]

if os.path.exists(ground_truth_path):
    # Look for corresponding ground truth file
    gt_files = [f for f in os.listdir(ground_truth_path) if base_filename in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if gt_files:
        gt_path = os.path.join(ground_truth_path, gt_files[0])
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is not None:
            gt_mask = cv2.resize(gt_img, (256, 256)) / 255.0
            gt_mask = (gt_mask > 0.5).astype(np.float32)
            
            # Binarize predicted mask
            pred_mask_binary = (mask > 0.5).astype(np.float32)
            
            # Calculate metrics
            # Dice coefficient
            intersection = np.sum(pred_mask_binary * gt_mask)
            dice = 2 * intersection / (np.sum(pred_mask_binary) + np.sum(gt_mask) + 1e-8)
            metrics['Dice'] = dice
            
            # IoU (Intersection over Union)
            union = np.sum((pred_mask_binary + gt_mask) > 0)
            iou = intersection / (union + 1e-8)
            metrics['IoU'] = iou
            
            # Pixel Accuracy
            pixel_accuracy = np.sum(pred_mask_binary == gt_mask) / gt_mask.size
            metrics['Pixel Accuracy'] = pixel_accuracy
            
            # Precision
            tp = np.sum(pred_mask_binary * gt_mask)
            fp = np.sum(pred_mask_binary * (1 - gt_mask))
            precision = tp / (tp + fp + 1e-8)
            metrics['Precision'] = precision
            
            # Recall
            fn = np.sum((1 - pred_mask_binary) * gt_mask)
            recall = tp / (tp + fn + 1e-8)
            metrics['Recall'] = recall

# Display input image and predicted mask side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(original_img.astype(np.uint8))
axes[0].set_title("Input Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Predicted Mask")
axes[1].axis("off")

plt.tight_layout()

# Display metrics if available
if metrics:
    print("\n" + "="*40)
    print("Segmentation Metrics:")
    print("="*40)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("="*40 + "\n")
else:
    print("\nGround truth mask not found. Metrics cannot be computed.")
    print("Make sure Test_GroundTruth folder contains the corresponding ground truth mask.\n")

plt.show()


