# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse

import IPython
import torch

from .models import (
    build_ACT_model,
    build_CNNMLP_model,
    build_DiffusionFlow_model,
    build_DiffusionPolicy_model,
)

e = IPython.embed


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)  # will be overridden
    parser.add_argument("--lr_backbone", default=1e-5, type=float)  # will be overridden
    parser.add_argument("--batch_size", default=2, type=int)  # not used
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)  # not used
    parser.add_argument("--lr_drop", default=200, type=int)  # not used
    parser.add_argument(
        "--clip_max_norm",
        default=0.1,
        type=float,  # not used
        help="gradient clipping max norm",
    )

    # Model parameters
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet18",
        type=str,  # will be overridden
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--camera_names",
        default=[],
        type=list,  # will be overridden
        help="A list of camera names",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=4,
        type=int,  # will be overridden
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,  # will be overridden
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,  # will be overridden
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,  # will be overridden
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,  # will be overridden
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries",
        default=400,
        type=int,  # will be overridden
        help="Number of query slots",
    )
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument(
        "--state_dim",
        default=14,
        type=int,  # will be overridden
        help="Proprioceptive state dimension",
    )
    parser.add_argument(
        "--action_dim",
        default=14,
        type=int,  # will be overridden
        help="Action dimension",
    )

    # * Segmentation
    parser.add_argument(
        "--masks", action="store_true", help="Train segmentation head if the flag is provided"
    )

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class", action="store", type=str, help="policy_class, capitalize", required=True
    )
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--dataset_dir", action="store", type=str, default=None)
    parser.add_argument("--num_episodes", action="store", type=int, default=None)
    parser.add_argument("--task_space", action="store_true")
    parser.add_argument("--action_repr", action="store", type=str, default="absolute")
    parser.add_argument("--rot_repr", action="store", type=str, default="quat")
    parser.add_argument("--num_rollouts", action="store", type=int, default=None)
    parser.add_argument("--num_checkpoints", action="store", type=int, default=5)

    # for DiffusionFlow (flow-matching DiT policy)
    parser.add_argument(
        "--dit_arch",
        action="store",
        type=str,
        default="cross_attn",
        choices=("cross_attn", "concat"),
    )
    parser.add_argument("--dit_layers", action="store", type=int, default=8)
    parser.add_argument("--n_inference_steps", action="store", type=int, default=4)
    parser.add_argument("--state_dropout_prob", action="store", type=float, default=0.0)
    parser.add_argument("--cam_dropout_prob", action="store", type=float, default=0.0)
    parser.add_argument("--ema_decay", action="store", type=float, default=0.9999)

    # for DiffusionPolicy (CNN 1D-UNet or gr00t-DiT denoiser + DDPM)
    parser.add_argument(
        "--dp_denoiser", action="store", type=str, default="unet", choices=("unet", "dit")
    )
    parser.add_argument(
        "--dp_down_dims", action="store", type=int, nargs="+", default=[256, 512, 1024]
    )
    parser.add_argument("--dp_kernel_size", action="store", type=int, default=5)
    parser.add_argument("--dp_n_groups", action="store", type=int, default=8)
    parser.add_argument("--dp_diffusion_step_embed_dim", action="store", type=int, default=128)
    parser.add_argument("--dp_num_train_timesteps", action="store", type=int, default=100)
    parser.add_argument("--dp_beta_schedule", action="store", type=str, default="squaredcos_cap_v2")
    parser.add_argument(
        "--dp_prediction_type",
        action="store",
        type=str,
        default="epsilon",
        choices=("epsilon", "sample"),
    )
    parser.add_argument("--dp_action_scale", action="store", type=float, default=4.0)
    parser.add_argument(
        "--dp_obs_pool",
        action="store",
        type=str,
        default="spatial_softmax",
        choices=("spatial_softmax", "avg"),
    )
    parser.add_argument("--dp_num_kp", action="store", type=int, default=32)
    parser.add_argument(
        "--dp_inference_scheduler",
        action="store",
        type=str,
        default="ddim",
        choices=("ddim", "ddpm"),
    )
    parser.add_argument("--ema_power", action="store", type=float, default=0.75)
    parser.add_argument("--ema_inv_gamma", action="store", type=float, default=1.0)
    parser.add_argument("--ema_min_value", action="store", type=float, default=0.0)

    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer


def build_DiffusionFlow_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_DiffusionFlow_model(args)
    model.cuda()

    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer


def build_DiffusionPolicy_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_DiffusionPolicy_model(args)
    model.cuda()

    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer
