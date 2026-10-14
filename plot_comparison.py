import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from models import Generator

def generate_comparison_plots(
    weights_path="checkpoints/netG_final.pth",
    data_dir="data/processed/val",
    output_dir="results/visual_comparisons",
    num_samples=4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load trained Generator
    netG = Generator().to(device)
    if not os.path.exists(weights_path):
        weights_path = sorted(glob.glob("checkpoints/netG_epoch_*.pth"))[-1]
    
    netG.load_state_dict(torch.load(weights_path, map_location=device))
    netG.eval()
    print(f"Loaded weights from {weights_path}")

    mri_files = sorted(glob.glob(os.path.join(data_dir, "mri", "*.npy")))
    ct_files = sorted(glob.glob(os.path.join(data_dir, "ct", "*.npy")))

    # Pick spaced mid-volume slices
    total_slices = len(mri_files)
    step = total_slices // (num_samples + 1)
    indices = [step * (i + 1) for i in range(num_samples)]

    with torch.no_grad():
        for idx in indices:
            slice_name = os.path.basename(mri_files[idx]).replace(".npy", "")
            
            # Load and normalize
            mri_arr = np.load(mri_files[idx]).astype(np.float32)
            ct_arr = np.load(ct_files[idx]).astype(np.float32)

            mri_tensor = torch.from_numpy(mri_arr).unsqueeze(0).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda"):
                pred_tensor = netG(mri_tensor)

            pred_arr = pred_tensor.squeeze().cpu().numpy()

            # Denormalize to Hounsfield Units [-1000, 1000] HU
            ct_hu = ct_arr * 1000.0
            pred_hu = pred_arr * 1000.0
            diff_hu = np.abs(pred_hu - ct_hu)

            # Compute slice metrics
            body_mask = ct_hu > -900.0
            slice_mae = np.mean(diff_hu[body_mask]) if np.sum(body_mask) > 100 else np.mean(diff_hu)

            # Plot 4-panel comparison
            fig, axs = plt.subplots(1, 4, figsize=(18, 5))

            # MRI (Input)
            im0 = axs[0].imshow(mri_arr, cmap="gray", vmin=-1.0, vmax=1.0)
            axs[0].set_title("Input MRI", fontsize=13, fontweight="bold")
            axs[0].axis("off")

            # Synthetic CT (Prediction)
            im1 = axs[1].imshow(pred_hu, cmap="gray", vmin=-1000, vmax=1000)
            axs[1].set_title("Generated Synthetic CT", fontsize=13, fontweight="bold")
            axs[1].axis("off")

            # Ground Truth CT
            im2 = axs[2].imshow(ct_hu, cmap="gray", vmin=-1000, vmax=1000)
            axs[2].set_title("Ground Truth CT", fontsize=13, fontweight="bold")
            axs[2].axis("off")

            # Absolute Error Map
            im3 = axs[3].imshow(diff_hu, cmap="hot", vmin=0, vmax=500)
            axs[3].set_title(f"Absolute Error (|Diff|)\nMean Tissue MAE: {slice_mae:.1f} HU", fontsize=13, fontweight="bold")
            axs[3].axis("off")

            # Add Colorbar for the error map
            cbar = fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)
            cbar.set_label("Absolute Error (HU)", fontsize=10)

            plt.tight_layout()
            out_file = os.path.join(output_dir, f"{slice_name}_comparison.png")
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Saved: {out_file}")

    print(f"\nAll plots saved to {output_dir}/")

if __name__ == "__main__":
    generate_comparison_plots()