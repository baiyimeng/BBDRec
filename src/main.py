import random
import time

import numpy as np
import torch

from core.trainer import (
    choose_model,
    item_num_create,
    load_data,
    model_train,
)
from utils.logger import make_logger


def setup_seed(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    start_time = time.time()
    train_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger, args = make_logger(train_time)
    setup_seed(args.random_seed)
    args = item_num_create(args)
    model = choose_model(args)
    # =====================================================================
    # !! IMPORTANT: choose_model() MUST come before load_data().
    # choose_model modifies args (split_onebyone, parallel_ag, is_causal)
    # that load_data() / Data_Train depends on. DO NOT swap this order.
    # =====================================================================
    print(model)
    tra_data_loader, val_data_loader, test_data_loader = load_data(args)

    print(args.description)
    logger.info(args.description)
    print(args)
    formatted_args = "\n".join(f"{key}: {value}" for key, value in vars(args).items())
    logger.info("Arguments:\n%s", formatted_args)

    best_model, test_results = model_train(
        model,
        tra_data_loader,
        val_data_loader,
        test_data_loader,
        args,
        logger,
        train_time,
    )
    training_duration_seconds = time.time() - start_time
    minutes = training_duration_seconds // 60
    seconds = training_duration_seconds % 60
    logger.info(f"Training duration: {minutes} minutes and {seconds} seconds")


if __name__ == "__main__":
    main()
