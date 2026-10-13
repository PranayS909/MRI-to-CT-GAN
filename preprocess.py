import os
import glob
import numpy as np
import nibabel as nib
import cv2

def run_preprocessing(
    raw_brain_dir="data/raw/Task1/brain",
    output_dir="data/processed",
    target_size=(256, 256),
    val_split=0.15,
    min_mask_area=0.10  # Raised slightly to drop tiny peripheral skull slivers
):
    for split in ["train", "val"]:
        os.makedirs(f"{output_dir}/{split}/mri", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/ct", exist_ok=True)

    patient_dirs = [
        d for d in sorted(glob.glob(os.path.join(raw_brain_dir, "*")))
        if os.path.isdir(d) and not os.path.basename(d).startswith("overview")
    ]

    np.random.seed(42)
    np.random.shuffle(patient_dirs)
    val_count = int(len(patient_dirs) * val_split)
    val_patients = set(patient_dirs[:val_count])

    slice_counters = {"train": 0, "val": 0}

    for patient_dir in patient_dirs:
        split = "val" if patient_dir in val_patients else "train"
        patient_id = os.path.basename(patient_dir)

        mr_path = os.path.join(patient_dir, "mr.nii.gz")
        ct_path = os.path.join(patient_dir, "ct.nii.gz")
        mask_path = os.path.join(patient_dir, "mask.nii.gz")

        if not (os.path.exists(mr_path) and os.path.exists(ct_path) and os.path.exists(mask_path)):
            continue

        # Canonical reorientation guarantees Axis 2 is strictly the axial (feet-to-head) plane
        mr_nii = nib.as_closest_canonical(nib.load(mr_path))
        ct_nii = nib.as_closest_canonical(nib.load(ct_path))
        mask_nii = nib.as_closest_canonical(nib.load(mask_path))

        mr_vol = mr_nii.get_fdata(dtype=np.float32)
        ct_vol = ct_nii.get_fdata(dtype=np.float32)
        mask_vol = mask_nii.get_fdata(dtype=np.float32)

        # 1. CT Windowing [-1000, 1000] HU -> [-1, 1]
        ct_vol = np.clip(ct_vol, -1000.0, 1000.0)
        ct_vol = (ct_vol + 1000.0) / 1000.0 - 1.0

        # 2. MRI Normalization with 99th-percentile tissue clipping
        tissue_mask = mask_vol > 0
        if np.any(tissue_mask):
            p99 = np.percentile(mr_vol[tissue_mask], 99.0)
        else:
            p99 = mr_vol.max()

        mr_vol = np.clip(mr_vol, 0.0, p99)
        mr_vol = (mr_vol / (p99 + 1e-8)) * 2.0 - 1.0

        # 3. Axial Slicing along Axis 2
        num_slices = mr_vol.shape[2]
        for z in range(num_slices):
            mask_slice = mask_vol[:, :, z]

            # Drop slices that have less than 10% brain/skull tissue
            if np.mean(mask_slice > 0) < min_mask_area:
                continue

            mr_slice = mr_vol[:, :, z]
            ct_slice = ct_vol[:, :, z]

            # Rotate 90 degrees counter-clockwise so the brain faces up
            mr_slice = np.rot90(mr_slice)
            ct_slice = np.rot90(ct_slice)

            # Resize to network dimensions
            mr_resized = cv2.resize(mr_slice, target_size, interpolation=cv2.INTER_LINEAR)
            ct_resized = cv2.resize(ct_slice, target_size, interpolation=cv2.INTER_LINEAR)

            out_id = f"{patient_id}_z{z:03d}.npy"
            np.save(f"{output_dir}/{split}/mri/{out_id}", mr_resized)
            np.save(f"{output_dir}/{split}/ct/{out_id}", ct_resized)
            slice_counters[split] += 1

    print(f"Preprocessed: {slice_counters['train']} train pairs, {slice_counters['val']} val pairs.")

if __name__ == "__main__":
    run_preprocessing()