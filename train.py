import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
from dataset import get_dataloaders
from models import Generator, Discriminator

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50
    batch_size = 8
    lr = 2e-4
    lambda_l1 = 100.0

    # Optimize cuDNN kernels for consistent (256x256) input shapes
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    train_loader, val_loader = get_dataloaders("data/processed", batch_size=batch_size, num_workers=0)

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # ----------------- Resume Checkpoint Setup -----------------
    resume_epoch = 40  # Set to the last saved checkpoint epoch (e.g., 20)

    if resume_epoch > 0:
        g_path = f"checkpoints/netG_epoch_{resume_epoch:02d}.pth"
        d_path = f"checkpoints/netD_epoch_{resume_epoch:02d}.pth"   
        if os.path.exists(g_path) and os.path.exists(d_path):
            print(f"Loading checkpoint weights from epoch {resume_epoch}...")
            netG.load_state_dict(torch.load(g_path, map_location=device))
            netD.load_state_dict(torch.load(d_path, map_location=device))
            start_epoch = resume_epoch + 1
        else:
            print("Checkpoint files not found. Starting from scratch.")
            start_epoch = 1
    else:
        start_epoch = 1

    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    scaler_G = torch.amp.GradScaler("cuda")
    scaler_D = torch.amp.GradScaler("cuda")

    # Pick 4 representative mid-brain slices for clear anatomical preview
    val_dataset = val_loader.dataset
    total_val = len(val_dataset)
    # Sample spaced slices centered around mid-volume
    mid_idx = total_val // 2
    step = max(1, total_val // 20)
    sample_indices = [
        max(0, mid_idx - step * 2),
        max(0, mid_idx - step),
        mid_idx,
        min(total_val - 1, mid_idx + step)
    ]

    fixed_mri = torch.stack([val_dataset[i][0] for i in sample_indices]).to(device)
    fixed_ct = torch.stack([val_dataset[i][1] for i in sample_indices]).to(device)

    print(f"Starting training on {device} (RTX 5070 Laptop with Mixed Precision)...")
    print(f"Tracking validation indices {sample_indices} for visual samples.")

    for epoch in range(start_epoch, epochs + 1):
        netG.train()
        netD.train()
        running_g, running_d = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs}]", leave=False)
        for mri, ct in pbar:
            mri, ct = mri.to(device, non_blocking=True), ct.to(device, non_blocking=True)

            # ----------------- Train Discriminator -----------------
            optD.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                fake_ct = netG(mri)
                d_real = netD(mri, ct)
                loss_d_real = criterion_bce(d_real, torch.ones_like(d_real))

                d_fake = netD(mri, fake_ct.detach())
                loss_d_fake = criterion_bce(d_fake, torch.zeros_like(d_fake))

                loss_d = (loss_d_real + loss_d_fake) * 0.5

            scaler_D.scale(loss_d).backward()
            scaler_D.step(optD)
            scaler_D.update()

            # ------------------- Train Generator -------------------
            optG.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                d_fake_g = netD(mri, fake_ct)
                loss_g_gan = criterion_bce(d_fake_g, torch.ones_like(d_fake_g))
                loss_g_l1 = criterion_l1(fake_ct, ct) * lambda_l1
                loss_g = loss_g_gan + loss_g_l1

            scaler_G.scale(loss_g).backward()
            scaler_G.step(optG)
            scaler_G.update()

            running_g += loss_g.item()
            running_d += loss_d.item()
            pbar.set_postfix({"Loss G": f"{loss_g.item():.3f}", "Loss D": f"{loss_d.item():.3f}"})

        avg_g = running_g / len(train_loader)
        avg_d = running_d / len(train_loader)
        print(f"Epoch [{epoch:02d}/{epochs}] | Loss G: {avg_g:.4f} | Loss D: {avg_d:.4f}")

        # Save visual comparison (Row 1: MRI | Row 2: Generated CT | Row 3: Real CT)
        if epoch == 1 or epoch % 5 == 0:
            netG.eval()
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    pred = netG(fixed_mri)
                
                # Combine rows: MRI (top), Prediction (mid), Ground Truth (bottom)
                comparison = torch.cat([fixed_mri, pred, fixed_ct], dim=0)
                # Rescale from [-1, 1] to [0, 1] and clamp to avoid format errors
                comparison = (comparison + 1.0) / 2.0
                comparison = torch.clamp(comparison, 0.0, 1.0)
                
                out_path = os.path.join("samples", f"epoch_{epoch:02d}.png")
                save_image(comparison.cpu(), out_path, nrow=4)
                print(f"--> Saved sample preview: {out_path}")

        if epoch % 10 == 0:
            torch.save(netG.state_dict(), f"checkpoints/netG_epoch_{epoch:02d}.pth")
            torch.save(netD.state_dict(), f"checkpoints/netD_epoch_{epoch:02d}.pth")

    torch.save(netG.state_dict(), "checkpoints/netG_final.pth")
    print("Training complete. Weights saved to checkpoints/netG_final.pth")

if __name__ == "__main__":
    train()