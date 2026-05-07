import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tl_helpers import (
    build_finetune_dataloaders,
    build_finetune_model,
    load_pretrained_weights,
    process_finetune_batch,
    get_normalization_tensors,
    super_loss,
    save_pickle,
    set_random_seed,
)


# =========================
# Configuration
# =========================

SEED = 42
SCRATCH = False
FREEZE = False
RANDOM = True

MODE = "day"
DROPOUT_PCT = 0.05

DEVICE = "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 80
TRAIN_PCT = 0.9

LEARNING_RATE = 5e-4
DECAY_RATE = 0.99

PLATEAU_PATIENCE = 3
PLATEAU_FACTOR = 0.5

TAU_INIT = 0.05
SUPERLOSS_LAMBDA = 0.1


# =========================
# Path placeholders
# =========================

PATH_TO_DATA = "/path/to/finetune_all.csv"
PATH_TO_PRETRAINED_MODEL = "/path/to/pretrained_model.pt"
DIR_TO_STORE = "/path/to/output_dir"

os.makedirs(DIR_TO_STORE, exist_ok=True)

RUN_NAME = (
    f"_finetune_1_US_ERA5_{MODE}"
    f"_scratchAllFC_nofreeze_moreDrop{DROPOUT_PCT}"
    f"_lessParam_skipcon_lst_2013-2024"
    f"_3yearComposite_production"
)


# =========================
# Reproducibility
# =========================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# =========================
# Data
# =========================

finetune_trainloader, finetune_testloader, len_train_data, len_test_data = (
    build_finetune_dataloaders(
        path_to_data=PATH_TO_DATA,
        batch_size=BATCH_SIZE,
        train_pct=TRAIN_PCT,
        num_workers=4,
    )
)

print(PATH_TO_DATA)
print(f"len_train: {len_train_data}")
print(f"len_test: {len_test_data}")


# =========================
# Normalization constants
# =========================

(
    image_normalize,
    forcing_mean,
    forcing_std,
    lst_mean,
    lst_std,
) = get_normalization_tensors(DEVICE)


# =========================
# Model
# =========================

model = resnet_simplified(dropout_pct=DROPOUT_PCT)

if not SCRATCH:
    load_pretrained_weights(
        model,
        PATH_TO_PRETRAINED_MODEL,
        freeze=FREEZE,
    )

model = model.to(DEVICE)
model = torch.nn.DataParallel(model)


# =========================
# Loss and optimizer
# =========================

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
    factor=PLATEAU_FACTOR,
)


# =========================
# Training records
# =========================

train_loss = []
test_loss = []

train_loss_l1 = []
test_loss_l1 = []

min_test_loss = np.inf


# =========================
# Training loop
# =========================

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

    for forcing, image, month, t2m, lst in finetune_trainloader:
        t2m_l1 = t2m.view(-1, 1).to(DEVICE).to(torch.float32)

        forcing, image, month, t2m, lst = process_data(
            forcing=forcing,
            image=image,
            month=month,
            t2m=t2m,
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

        pred = model(image, forcing, one_hot_mon, lst)

        loss = loss_fn(pred, t2m)
        loss_super = super_loss(loss, tau_running, SUPERLOSS_LAMBDA)

        pred_l1 = pred * lst_std + lst_mean
        train_l1 = loss_fn_l1(pred_l1, t2m_l1)

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
        for forcing, image, month, t2m, lst in finetune_testloader:
            t2m_l1 = t2m.view(-1, 1).to(DEVICE).to(torch.float32)

            forcing, image, month, t2m, lst = process_data(
                forcing=forcing,
                image=image,
                month=month,
                t2m=t2m,
                lst=lst,
                device=DEVICE,
                image_normalize=image_normalize,
                forcing_mean=forcing_mean,
                forcing_std=forcing_std,
                lst_mean=lst_mean,
                lst_std=lst_std,
            )

            one_hot_mon = F.one_hot(month, num_classes=12).to(torch.float32)

            pred = model(image, forcing, one_hot_mon, lst)

            batch_test_loss = loss_fn(pred, t2m)

            pred_l1 = pred * lst_std + lst_mean
            batch_test_l1 = loss_fn_l1(pred_l1, t2m_l1)

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
    # Save model and losses
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
        train_loss_l1,
        os.path.join(DIR_TO_STORE, f"train_loss_l1{RUN_NAME}.pkl"),
    )

    save_pickle(
        test_loss_l1,
        os.path.join(DIR_TO_STORE, f"test_loss_l1{RUN_NAME}.pkl"),
    )

    print(f"------ Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss} ------")
    print(f"------ Train L1 Loss: {avg_train_l1}, Test L1 Loss: {avg_test_l1} ------")