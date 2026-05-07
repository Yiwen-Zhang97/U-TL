import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tl_helpers import (
    build_pretrain_dataloaders,
    build_pretrain_model,
    get_normalization_tensors,
    process_pretrain_batch,
    save_pickle,
    set_random_seed,
    super_loss,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

PATH_TO_DATA = "/path/to/pretrain_hdf5_dir"
DIR_TO_STORE = "/path/to/output_dir"

os.makedirs(DIR_TO_STORE, exist_ok=True)

RUN_NAME = (
    "_more_linear_samesize_all_cities_LSTNorm_CL_6_L1"
    "_onehot_day_ERA5_US_monthly_0-1_periodicFalse"
    "_2013-2024_18cities_1yearComposite"
)

NUMBER_HDFS_WANTED_TRAIN = 10
NUMBER_HDFS_WANTED_TEST = 1

EPOCHS = 100
BATCH_SIZE = 1024

LEARNING_RATE = 1e-3
DECAY_RATE = 0.96

PLATEAU_PATIENCE = 5

TAU_INIT = 0.05
SUPERLOSS_LAMBDA = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Setup
# ============================================================

set_random_seed(SEED)

torch.backends.cudnn.benchmark = True

print(PATH_TO_DATA)
print(f"Using device: {DEVICE}")
print(f"CUDA device count: {torch.cuda.device_count()}")


# ============================================================
# Data
# ============================================================

pretrain_trainloader, pretrain_testloader, len_train, len_test = (
    build_pretrain_dataloaders(
        path_to_data_dir=PATH_TO_DATA,
        number_hdfs_wanted_train=NUMBER_HDFS_WANTED_TRAIN,
        number_hdfs_wanted_test=NUMBER_HDFS_WANTED_TEST,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )
)


# ============================================================
# Normalization constants
# ============================================================

(
    image_normalize,
    forcing_mean,
    forcing_std,
    lst_mean,
    lst_std,
) = get_normalization_tensors(DEVICE)


# ============================================================
# Model
# ============================================================

model = build_pretrain_model(dropout_pct=0.02)
model = model.to(DEVICE)
model = torch.nn.DataParallel(model)


# ============================================================
# Loss and optimizer
# ============================================================

loss_fn = nn.L1Loss()
loss_fn_l1 = nn.L1Loss()

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=DECAY_RATE,
)

scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=PLATEAU_PATIENCE,
)


# ============================================================
# Training records
# ============================================================

train_loss = []
test_loss = []

train_loss_l1 = []
test_loss_l1 = []

min_test_loss = np.inf


# ============================================================
# Training loop
# ============================================================

for epoch in range(EPOCHS):
    current_lr = round(optimizer.param_groups[0]["lr"], 6)
    print(f"****** EPOCH: [{epoch}/{EPOCHS}] LR: {current_lr} ******")

    running_train_loss = 0.0
    running_train_l1 = 0.0
    train_n_iter = 0

    running_test_loss = 0.0
    running_test_l1 = 0.0
    test_n_iter = 0

    if epoch == 0:
        tau_running = TAU_INIT
    else:
        tau_running = avg_train_loss

    print(f"tau: {tau_running}")

    # -------------------------
    # Train
    # -------------------------

    model.train()

    for forcing, image, month, lst in pretrain_trainloader:
        lst_l1 = lst.view(-1, 1).to(DEVICE).to(torch.float32)

        forcing, image, month, lst = process_pretrain_batch(
            forcing=forcing,
            image=image,
            month=month,
            lst=lst,
            device=DEVICE,
            image_normalize=image_normalize,
            forcing_mean=forcing_mean,
            forcing_std=forcing_std,
            lst_mean=lst_mean,
            lst_std=lst_std,
        )

        one_hot_mon = F.one_hot(month, num_classes=12).to(torch.float32)

        optimizer.zero_grad()

        pred = model(image, forcing, one_hot_mon)

        loss = loss_fn(pred, lst)
        loss_super = super_loss(loss, tau_running, SUPERLOSS_LAMBDA)

        pred_l1 = pred * lst_std + lst_mean
        train_l1 = loss_fn_l1(pred_l1, lst_l1)

        loss_super.backward()
        optimizer.step()

        running_train_loss += loss.item()
        running_train_l1 += train_l1.item()
        train_n_iter += 1

    # -------------------------
    # Evaluate
    # -------------------------

    model.eval()

    with torch.no_grad():
        for forcing, image, month, lst in pretrain_testloader:
            lst_l1 = lst.view(-1, 1).to(DEVICE).to(torch.float32)

            forcing, image, month, lst = process_pretrain_batch(
                forcing=forcing,
                image=image,
                month=month,
                lst=lst,
                device=DEVICE,
                image_normalize=image_normalize,
                forcing_mean=forcing_mean,
                forcing_std=forcing_std,
                lst_mean=lst_mean,
                lst_std=lst_std,
            )

            one_hot_mon = F.one_hot(month, num_classes=12).to(torch.float32)

            pred = model(image, forcing, one_hot_mon)

            batch_test_loss = loss_fn(pred, lst)

            pred_l1 = pred * lst_std + lst_mean
            batch_test_l1 = loss_fn_l1(pred_l1, lst_l1)

            running_test_loss += batch_test_loss.item()
            running_test_l1 += batch_test_l1.item()
            test_n_iter += 1

    # -------------------------
    # Epoch summary
    # -------------------------

    avg_train_loss = running_train_loss / train_n_iter
    avg_test_loss = running_test_loss / test_n_iter

    avg_train_l1 = running_train_l1 / train_n_iter
    avg_test_l1 = running_test_l1 / test_n_iter

    train_loss.append(avg_train_loss)
    test_loss.append(avg_test_loss)

    train_loss_l1.append(avg_train_l1)
    test_loss_l1.append(avg_test_l1)

    scheduler.step()
    scheduler_plateau.step(avg_test_loss)

    # -------------------------
    # Save model and loss records
    # -------------------------

    if avg_test_loss < min_test_loss:
        print("Saving best model...")

        min_test_loss = avg_test_loss

        model_path = os.path.join(
            DIR_TO_STORE,
            f"best_resnet{RUN_NAME}.pt",
        )

        torch.save(model.state_dict(), model_path)

    print("Saving losses...")

    save_pickle(
        train_loss,
        os.path.join(DIR_TO_STORE, f"train_loss{RUN_NAME}.pkl"),
    )

    save_pickle(
        test_loss,
        os.path.join(DIR_TO_STORE, f"test_loss{RUN_NAME}.pkl"),
    )

    save_pickle(
        train_loss_l1,
        os.path.join(DIR_TO_STORE, f"train_loss_l1{RUN_NAME}.pkl"),
    )

    save_pickle(
        test_loss_l1,
        os.path.join(DIR_TO_STORE, f"test_loss_l1{RUN_NAME}.pkl"),
    )

    print(f"------ Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss} ------")
    print(f"------ Train L1 Loss: {avg_train_l1}, Test L1 Loss: {avg_test_l1} ------")