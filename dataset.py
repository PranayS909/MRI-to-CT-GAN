import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MRIToCTSliceDataset(Dataset):
    def __init__(self, split_dir):
        self.mri_dir = os.path.join(split_dir, "mri")
        self.ct_dir = os.path.join(split_dir, "ct")
        self.filenames = sorted(os.listdir(self.mri_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        mri = np.load(os.path.join(self.mri_dir, fname))
        ct = np.load(os.path.join(self.ct_dir, fname))

        # Shape: (1, H, W)
        mri_t = torch.from_numpy(mri).unsqueeze(0).float()
        ct_t = torch.from_numpy(ct).unsqueeze(0).float()
        return mri_t, ct_t

def get_dataloaders(data_root="data/processed", batch_size=8, num_workers=4):
    train_ds = MRIToCTSliceDataset(os.path.join(data_root, "train"))
    val_ds = MRIToCTSliceDataset(os.path.join(data_root, "val"))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader