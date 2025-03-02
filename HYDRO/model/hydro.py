import torch
import torch.nn as nn
import torch.nn.functional as F

from HYDRO.model.simplified_dpm_solver import SingleStepDPMSolverScheduler
from HYDRO.model.tracker import TrackerTorch
from HYDRO.model.arcface import iresnet100
from HYDRO.model.networks import Generator, DiffusionGenerator
from HYDRO.config.config import diffusion_plus_addition_skips_attention_mlp, baseline_plus_addition_skips_attention


class HYDRO(nn.Module):
    def __init__(self,
                 tracker_enabled=True,
                 tracker_track=True,
                 tracker_max_size=10000,
                 tracker_threshold=0.6,
                 tracker_margin=0.3,
                 tracker_anchor_size=100000,
                 path_target_oriented='HYDRO/pretrained/HYDRO_target_oriented/50000_model_state.pth',
                 path_diffusion_model='HYDRO/pretrained/HYDRO_diffusion_model/150000_model_state.pth',
                 path_arcface='HYDRO/pretrained/arcface/backbone.pth',
                 target_oriented_config=baseline_plus_addition_skips_attention,
                 diffusion_model_config=diffusion_plus_addition_skips_attention_mlp,
                 ):
        super(HYDRO, self).__init__()

        self.arcface = iresnet100()
        self.arcface.load_state_dict(torch.load(path_arcface))
        self.arcface.eval()

        self.generator = Generator(**target_oriented_config)
        self.generator.load_state_dict(torch.load(path_target_oriented)["generator"])
        self.generator.eval()

        self.diffusion = DiffusionGenerator(**diffusion_model_config)
        self.diffusion.load_state_dict(torch.load(path_diffusion_model)["generator"])
        self.diffusion.eval()

        self.tracker_enabled = tracker_enabled
        self.tracker = TrackerTorch(max_size=tracker_max_size,
                                      threshold=tracker_threshold,
                                      margin=tracker_margin,
                                      track=tracker_track,
                                      anchor_size=tracker_anchor_size)
        self.scheduler = SingleStepDPMSolverScheduler(num_timesteps=1000,
                                                      beta_start=0.00085,
                                                      beta_end=0.012,
                                                      solver_order=1,
                                                      rescale_betas_zero_snr=False)

    def forward(self, x, noise_strength=0.1, apply_diffusion=True):
        condition = self.arcface(F.interpolate(x, size=112))

        if self.tracker_enabled:
            condition = self.tracker(condition)

        deid, mask = self.generator(x, condition)
        deid = (deid * mask + x * (1 - mask)).clamp(-1, 1)

        if apply_diffusion:
            t = torch.tensor(int(noise_strength * 1000), device=deid.device).long()
            noise = torch.randn(deid.shape, device=deid.device)
            noisy_deid = self.scheduler.add_noise(deid, noise, t)
            timestep_batch = t.repeat(noisy_deid.shape[0])

            noise_prediction = self.diffusion(noisy_deid, -1 * condition, timestep_batch)

            deid = self.scheduler.step(noise_prediction, t, noisy_deid)

            deid = (deid * mask + x * (1 - mask)).clamp(-1, 1)

        return deid
