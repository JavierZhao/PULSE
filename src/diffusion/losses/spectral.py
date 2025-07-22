import torch
import torch.fft as fft
import torch.nn.functional as F

def spectral_l1(real, fake, fs=64, fmax=1.0):
    """
    L1 distance between magnitude spectra of real and fake EDA.
    real, fake: tensors [B, T] already z-scored; T = 60 s * 64 Hz = 3840.
    Only frequencies 0–fmax Hz are considered.
    """
    B, T = real.shape
    # 1-D FFT, keep one-sided spectrum
    R = fft.rfft(real, n=T)          # [B, T//2+1]
    F_ = fft.rfft(fake, n=T)
    # Frequency axis
    freqs = torch.linspace(0, fs/2, R.shape[-1], device=real.device)
    mask  = freqs <= fmax            # boolean mask up to 1 Hz
    # L1 distance of magnitudes
    loss  = F.l1_loss(torch.abs(R)[..., mask], torch.abs(F_)[..., mask])
    return loss
