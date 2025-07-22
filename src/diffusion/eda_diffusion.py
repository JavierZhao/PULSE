import torch
import logging
from tqdm import tqdm

class EDADiffusion:
    """
    Manages the diffusion process (forward and reverse) for 1D biomedical signals like EDA.
    
    This class is adapted from a 2D image diffusion model to work with 1D time-series data.
    It handles the noise scheduling, adding noise to signals (forward process), and generating
    new signals from noise (reverse process/sampling).
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, sequence_length=3840, num_channels=1, device="cuda"):
        """
        Initializes the EDADiffusion class.

        Args:
            noise_steps (int): Number of steps in the diffusion process.
            beta_start (float): Starting value for beta in the noise schedule.
            beta_end (float): Ending value for beta in the noise schedule.
            sequence_length (int): The length of the input signal sequences (e.g., 60s * 64Hz = 3840).
            num_channels (int): The number of channels in the input signal (e.g., 1 for just EDA).
            device (str): The computational device ('cuda' or 'cpu').
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """
        Creates a linear noise schedule (beta values).

        Returns:
            torch.Tensor: A 1D tensor of beta values.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def q_sample(self, x, t):
        """
        Adds Gaussian noise to a batch of signals at a specific timestep 't' (forward process).
        (Formerly noise_signals)
        """
        # Ensure x is on the correct device
        x = x.to(self.device)
        
        # Reshape for broadcasting: (batch_size,) -> (batch_size, 1, 1)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        
        # Sample noise and apply it
        epsilon = torch.randn_like(x)
        noised_signal = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        
        return noised_signal, epsilon

    def sample_timesteps(self, n):
        """
        Samples 'n' random timesteps from [1, noise_steps).

        Args:
            n (int): The number of timesteps to sample (typically batch size).

        Returns:
            torch.Tensor: A 1D tensor of 'n' random timesteps.
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, x_start=None):
        """
        Generates new signals by sampling from the diffusion model (reverse process).

        Args:
            model: The trained U-Net model.
            n (int): The number of new signals to generate.
            x_start (torch.Tensor, optional): A starting tensor for the reverse process. 
                                              If None, starts from pure Gaussian noise.
        """
        logging.info(f"Sampling {n} new signals...")
        model.eval()
        with torch.no_grad():
            # Start with random noise or a provided tensor
            if x_start is not None:
                x = x_start.to(self.device)
                assert x.shape == (n, self.num_channels, self.sequence_length)
            else:
                x = torch.randn((n, self.num_channels, self.sequence_length)).to(self.device)
            
            # Denoise step-by-step from T to 1
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                
                # Predict noise
                predicted_noise = model(x, t)
                
                # Get noise schedule values for the current timestep
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                
                # Add noise back in for all steps except the last one
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # Denoising formula
                term1 = 1 / torch.sqrt(alpha)
                term2 = (1 - alpha) / torch.sqrt(1 - alpha_hat)
                x = term1 * (x - term2 * predicted_noise) + torch.sqrt(beta) * noise
                
        model.train()
        
        return x.clamp(-1, 1) 