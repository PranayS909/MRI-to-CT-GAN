import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from models import Generator

def evaluate(weights_path="checkpoints/netG_final.pth", data_dir="data/processed/val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load trained Generator
    netG = Generator().to(device)
    if not os.path.exists(weights_path):
        weights_path = sorted(glob.glob("checkpoints/netG_epoch_*.pth"))[-1]
    
    netG.load_state_dict(torch.load(weights_path, map_location=device))
    netG.eval()
    print(f"Loaded weights from {weights_path}")

    mri_files = sorted(glob.glob(os.path.join(data_dir, "mri", "*.npy")))
    ct_files = sorted(glob.glob(os.path.join(data_dir, "ct", "*.npy")))

    assert len(mri_files) == len(ct_files), "Mismatch between validation MRI and CT count!"
    print(f"Evaluating on {len(mri_files)} validation slices...")

    mae_list, psnr_list, ssim_list = [], [], []

    with torch.no_grad():
        for m_path, c_path in tqdm(zip(mri_files, ct_files), total=len(mri_files)):
            # Load slices: shape (256, 256) in range [-1, 1]
            mri_arr = np.load(m_path).astype(np.float32)
            ct_arr = np.load(c_path).astype(np.float32)

            mri_tensor = torch.from_numpy(mri_arr).unsqueeze(0).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda"):
                pred_tensor = netG(mri_tensor)

            pred_arr = pred_tensor.squeeze().cpu().numpy()

            # Denormalize from [-1, 1] back to physical Hounsfield Units [-1000, 1000] HU
            # Formula: HU = (val + 1) * 1000 - 1000 = val * 1000
            ct_hu = ct_arr * 1000.0
            pred_hu = pred_arr * 1000.0

            # Tissue mask (ignore pure background air for fairer anatomical metric)
            body_mask = ct_hu > -900.0
            if np.sum(body_mask) > 100:
                mae = np.mean(np.abs(pred_hu[body_mask] - ct_hu[body_mask]))
            else:
                mae = np.mean(np.abs(pred_hu - ct_hu))
            mae_list.append(mae)

            # Normalized [0, 1] image comparison for PSNR & SSIM
            ct_norm = (ct_arr + 1.0) / 2.0
            pred_norm = np.clip((pred_arr + 1.0) / 2.0, 0.0, 1.0)

            psnr_list.append(psnr_metric(ct_norm, pred_norm, data_range=1.0))
            ssim_list.append(ssim_metric(ct_norm, pred_norm, data_range=1.0))

    print("\n" + "=" * 40)
    print("        VALIDATION RESULTS (SYNTHRAD)     ")
    print("=" * 40)
    print(f"Mean Absolute Error (MAE): {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f} HU")
    print(f"Peak SNR (PSNR):           {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"Structural Similarity:    {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print("=" * 40)

if __name__ == "__main__":
    evaluate()