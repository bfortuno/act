import copy

import IPython
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F

from detr.main import (
    build_ACT_model_and_optimizer,
    build_CNNMLP_model_and_optimizer,
    build_DiffusionFlow_model_and_optimizer,
    build_DiffusionPolicy_model_and_optimizer,
)

e = IPython.embed


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class DiffusionFlowPolicy(nn.Module):
    """Flow-matching (rectified-flow) policy with a gr00t-style DiT denoiser and the
    ACT CNN backbone. See detr/models/diffusion_flow.py."""

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_DiffusionFlow_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.num_queries = model.num_queries
        # weight EMA: sampling/eval use the EMA copy (standard for diffusion/flow BC).
        self.ema_decay = float(args_override.get("ema_decay", 0.9999))
        if self.ema_decay > 0:
            self.ema = copy.deepcopy(self.model)
            self.ema.requires_grad_(False)
        else:
            self.ema = None
        print(
            f"DiffusionFlow: dit_arch={model.dit_arch} num_queries={model.num_queries} "
            f"n_inference_steps={model.n_inference_steps} "
            f"state_dropout={model.state_dropout_prob} cam_dropout={model.cam_dropout_prob} "
            f"gripper_rescale={model.gripper_rescale} ema_decay={self.ema_decay}"
        )

    @torch.no_grad()
    def ema_step(self):
        if self.ema is None:
            return
        d = self.ema_decay
        for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1.0 - d)
        for be, bm in zip(self.ema.buffers(), self.model.buffers()):
            be.copy_(bm)

    def _net(self):
        """EMA weights for val/eval, raw weights while training."""
        if self.ema is not None and not self.training:
            self.ema.eval()
            return self.ema
        return self.model

    def __call__(self, qpos, image, actions=None, is_pad=None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        net = self._net()
        if actions is not None:  # training / validation
            actions = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]
            return net.compute_loss(qpos, image, actions, is_pad)
        else:  # inference time
            return net.sample(qpos, image)  # (bs, num_queries, action_dim)

    def configure_optimizers(self):
        return self.optimizer


class DiffusionPolicy(nn.Module):
    """Original Diffusion Policy (Chi et al.): DDPM over the action chunk with a
    CNN 1D-UNet (``--dp_denoiser unet``) or the gr00t-style DiT
    (``--dp_denoiser dit``) denoiser. See detr/models/diffusion_policy.py."""

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_DiffusionPolicy_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.num_queries = model.num_queries

        # Weight EMA with a power-function warmup (the decay ramps from ~0), so
        # sampling/eval get a meaningful average even with few optimizer steps per
        # epoch. sampling/eval use the EMA copy, training uses the raw weights.
        self.ema_decay = float(args_override.get("ema_decay", 0.9999))  # == max decay
        self.ema_power = float(args_override.get("ema_power", 0.75))
        self.ema_inv_gamma = float(args_override.get("ema_inv_gamma", 1.0))
        self.ema_min_value = float(args_override.get("ema_min_value", 0.0))
        if self.ema_decay > 0:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.requires_grad_(False)
        else:
            self.ema_model = None
        self.register_buffer("ema_step_count", torch.zeros((), dtype=torch.long))

        print(
            f"DiffusionPolicy: denoiser={model.denoiser} num_queries={model.num_queries} "
            f"obs_pool={model.obs_pool} num_kp={model.num_kp} down_dims={model.down_dims} "
            f"num_train_timesteps={model.num_train_timesteps} pred={model.prediction_type} "
            f"n_inference_steps={model.n_inference_steps} "
            f"inference_scheduler={model.inference_scheduler} ema_decay={self.ema_decay}"
        )

    @torch.no_grad()
    def ema_step(self):
        if self.ema_model is None:
            return
        self.ema_step_count += 1
        s = float(self.ema_step_count.item())
        decay = 1.0 - (1.0 + s / self.ema_inv_gamma) ** (-self.ema_power)
        decay = min(max(decay, self.ema_min_value), self.ema_decay)
        for pe, pm in zip(self.ema_model.parameters(), self.model.parameters()):
            pe.mul_(decay).add_(pm.detach(), alpha=1.0 - decay)
        for be, bm in zip(self.ema_model.buffers(), self.model.buffers()):
            if be.shape == bm.shape:  # skip derived buffers whose shape is input-dependent
                be.copy_(bm)

    def _net(self):
        """EMA weights for val/eval, raw weights while training."""
        if self.ema_model is not None and not self.training:
            self.ema_model.eval()
            return self.ema_model
        return self.model

    def __call__(self, qpos, image, actions=None, is_pad=None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        net = self._net()
        if actions is not None:  # training / validation
            actions = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]
            return net.compute_loss(qpos, image, actions, is_pad)
        else:  # inference time
            return net.sample(qpos, image)  # (bs, num_queries, action_dim)

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
