import os
import argparse
import logging
import yaml


def str2bool(value):
    if value.lower() in ("true", "1", "t", "y", "yes"):
        return True
    elif value.lower() in ("false", "0", "f", "n", "no"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def make_logger(train_time):
    config = load_config("config.yaml")

    args = cmdline_args()
    args = merge_config_with_args(config, args)

    # NOTE: run.sh already redirects stdout to logs/{model}_{dataset}_{time}.log,
    # so we do NOT create a separate FileHandler or nested log directories here.

    if not os.path.exists(args.log_file):
        print(f"Creating base log directory: {args.log_file}")
        os.makedirs(args.log_file)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    return logger, args


def cmdline_args():
    parser = argparse.ArgumentParser()

    # ============================================================
    # Dataset & Device
    # ============================================================
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument(
        "--device", type=str, help="Device"
    )
    parser.add_argument("--random_seed", type=int, help="Random seed")

    # ============================================================
    # Training Configuration
    # ============================================================
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="L2 regularization")
    parser.add_argument("--eval_interval", type=int, help="Evaluation interval")
    parser.add_argument("--patience", type=int, help="Early stopping patience")

    # ============================================================
    # Model Architecture (Common)
    # ============================================================
    parser.add_argument("--max_len", type=int, help="Maximum sequence length")
    parser.add_argument("--hidden_size", type=int, help="Hidden size")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--emb_dropout", type=float, help="Embedding dropout")
    parser.add_argument("--is_causal", type=str2bool, help="Causal attention")

    # ============================================================
    # Diffusion Model (Common)
    # ============================================================
    parser.add_argument("--diffusion_steps", type=int, help="Diffusion steps")
    parser.add_argument("--noise_schedule", type=str, help="Noise schedule")
    parser.add_argument("--beta_a", type=float, help="Beta parameter a")
    parser.add_argument("--beta_b", type=float, help="Beta parameter b")
    parser.add_argument("--schedule_sampler_name", type=str, help="Timestep sampler")
    parser.add_argument("--rescale_timesteps", help="Rescale timesteps")

    # ============================================================
    # Method-Specific
    # ============================================================
    parser.add_argument("--lambda_uncertainty", type=float, help="Uncertainty weight")
    parser.add_argument("--independent", type=str2bool, help="Independent diffusion")
    parser.add_argument("--geodesic", type=str2bool, help="Geodesic diffusion")
    parser.add_argument("--cfg_scale", type=float, help="CFG scale")
    parser.add_argument(
        "--diff_decoder", type=str, choices=["att", "mlp"], help="Decoder type"
    )
    parser.add_argument("--loss_scale", type=float, help="Loss scale")
    parser.add_argument("--var_max", type=float, help="Max variance for diffusion")

    # Training strategy
    parser.add_argument("--split_onebyone", type=str2bool, help="Split one-by-one")
    parser.add_argument("--parallel_ag", type=str2bool, help="Parallel autoregressive")
    parser.add_argument("--pretrained", type=str2bool, help="Use pretrained embedding")
    parser.add_argument("--freeze_emb", type=str2bool, help="Freeze embedding")
    parser.add_argument("--pcgrad", type=str2bool, help="Use PCGrad")

    # ============================================================
    # Logging & Evaluation
    # ============================================================
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--log_file", help="Log directory")
    parser.add_argument("--description", type=str, help="Experiment description")
    parser.add_argument("--metric_ks", nargs="+", type=int, help="Metrics@K")

    args = parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def merge_config_with_args(config, args):
    config_namespace = argparse.Namespace(**config)

    for key, value in vars(args).items():
        if value is not None:
            setattr(config_namespace, key, value)

    return config_namespace
