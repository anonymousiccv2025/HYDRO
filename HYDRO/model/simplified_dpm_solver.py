import torch
import numpy as np
from typing import Optional, List, Union


def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


# Re-written for our purpose for improving readability and control of the denoising step
class SingleStepDPMSolverScheduler:

    def __init__(
            self,
            num_timesteps: int = 1000,
            beta_start: float = 0.00085,
            beta_end: float = 0.012,
            solver_order: int = 1,
            rescale_betas_zero_snr: bool = False,
    ):

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        #self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        if rescale_betas_zero_snr:
            # Close to 0 without being 0 so first sigma is not inf
            # FP16 smallest positive subnormal works well here
            self.alphas_cumprod[-1] = 2 ** -24

        # Currently we only support VP-type noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # setable values
        self.num_timesteps = num_timesteps
        timesteps = np.linspace(0, self.num_timesteps - 1, self.num_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.lower_order_nums = 0
        self.step_index = None
        self.begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def set_begin_index(self, begin_index: int = 0):
        self.begin_index = begin_index

    def set_timesteps(
            self,
            num_inference_steps: int = None,
            device: Union[str, torch.device] = None,
    ):

        clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), -float("inf"))
        last_timestep = ((1000 - clipped_idx).numpy()).item()

        step_ratio = last_timestep // (self.num_timesteps + 1)
        timesteps = (
            (np.arange(0, self.num_timesteps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
        )
        timesteps += 1

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)

        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        sigma_last = 0

        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = 1 / ((sigma ** 2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t

        return alpha_t, sigma_t

    def step(
            self,
            model_output: torch.Tensor,
            timestep: Union[int, torch.Tensor],
            sample: torch.Tensor,
    ) -> torch.Tensor:

        sigma_s = self.sigmas[timestep]
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        x_t = (sample - sigma_s * model_output) / alpha_s

        return x_t

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timestep: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)

        step_indices = self.num_timesteps - timestep

        sigma = sigmas[timestep].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples


class SingleStepDDIMSolverScheduler:

    def __init__(
            self,
            num_timesteps: int = 1000,
            beta_start: float = 0.00085,
            beta_end: float = 0.012,
            solver_order: int = 1,
            rescale_betas_zero_snr: bool = False,
    ):

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        if rescale_betas_zero_snr:
            # Close to 0 without being 0 so first sigma is not inf
            # FP16 smallest positive subnormal works well here
            self.alphas_cumprod[-1] = 2 ** -24

        # Currently we only support VP-type noise schedule
        self.final_alpha_cumprod = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_timesteps = num_timesteps
        self.timesteps = torch.from_numpy(np.arange(0, num_timesteps)[::-1].copy().astype(np.int64))

    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = 1 / ((sigma ** 2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t

        return alpha_t, sigma_t

    def step(
            self,
            model_output: torch.Tensor,
            timestep: Union[int, torch.Tensor],
            step_size: int,
            sample: torch.Tensor,
    ) -> torch.Tensor:

        prev_timestep = timestep - self.num_timesteps // step_size

        sigma_s = self.sigmas[self.num_timesteps - timestep]
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        x_t = (sample - sigma_s * model_output) / alpha_s

        return x_t

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timestep: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)

        step_indices = self.num_timesteps - timestep

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

