import os
import math
import pickle
from collections import OrderedDict
from typing import Callable, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# Dataset loaders
# ============================================================

class PreTrainLoader(Dataset):
    def __init__(self, path: str):
        self.hf = h5py.File(path, "r")
        self.key = list(self.hf.keys())[0]

    def __len__(self):
        return self.hf[self.key].shape[0]

    def __getitem__(self, index):
        sample = self.hf[self.key][index, :]

        forcing = sample[5:14]
        month = sample[3]
        lst = sample[14]
        image = sample[15:].reshape(8, 33, 33)

        return forcing, image, month, lst


class FineTuneLoader(Dataset):
    def __init__(self, dataset: pd.DataFrame, cities=None, sites=None):
        self.dataset = dataset.copy()

        if cities is not None:
            self.dataset = self.dataset[self.dataset["city"].isin(cities)]

        if sites is not None:
            self.dataset = self.dataset[
                self.dataset["loc_idx"].astype(int).isin(sites)
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.iloc[index, :].values

        forcing = sample[5:14].astype("float32")
        month = sample[3].astype("float32")
        lst = sample[14].astype("float32")
        t2m = sample[15].astype("float32")
        image = sample[16:].reshape(8, 33, 33).astype("float32")

        return forcing, image, month, t2m, lst


def build_pretrain_dataloaders(
    path_to_data_dir: str,
    number_hdfs_wanted_train=None,
    number_hdfs_wanted_test=None,
    batch_size: int = 1024,
    num_workers: int = 4,
):
    train_paths = [
        os.path.join(path_to_data_dir, file)
        for file in os.listdir(path_to_data_dir)
        if file.endswith(".h5") and "training" in file
    ]

    test_paths = [
        os.path.join(path_to_data_dir, file)
        for file in os.listdir(path_to_data_dir)
        if file.endswith(".h5") and "testing" in file
    ]

    train_paths = sorted(
        train_paths,
        key=lambda x: int(x.partition("training_")[2].partition(".")[0]),
    )

    test_paths = sorted(
        test_paths,
        key=lambda x: int(x.partition("testing_")[2].partition(".")[0]),
    )

    if number_hdfs_wanted_train is not None:
        train_paths = train_paths[:number_hdfs_wanted_train]

    if number_hdfs_wanted_test is not None:
        test_paths = test_paths[:number_hdfs_wanted_test]

    train_data = torch.utils.data.ConcatDataset(
        [PreTrainLoader(path) for path in train_paths]
    )

    test_data = torch.utils.data.ConcatDataset(
        [PreTrainLoader(path) for path in test_paths]
    )

    trainloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    testloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print("Total samples for pretraining:", len(train_data))
    print("Total samples for pretraining test:", len(test_data))

    return trainloader, testloader, len(train_data), len(test_data)


def build_finetune_dataloaders(
    path_to_data: str,
    batch_size: int = 32,
    train_pct: float = 0.7,
    random_split: bool = True,
    cities_train=None,
    cities_test=None,
    sites_train=None,
    sites_test=None,
    sites_val=None,
    num_workers: int = 4,
):
    print("Loading fine-tuning dataset...")

    dataset_all = pd.read_csv(path_to_data, index_col=0)
    dataset_all = dataset_all[dataset_all["city"].astype(int) != 0]

    valloader = None
    len_val_data = 0

    if random_split:
        if sites_train is None:
            dataset_df = dataset_all.copy()
        else:
            dataset_df = dataset_all[
                dataset_all["loc_idx"].astype(int).isin(sites_train)
            ]

        if sites_val is not None:
            dataset_df = dataset_df[
                ~dataset_df["loc_idx"].astype(int).isin(sites_val)
            ]

        dataset = FineTuneLoader(dataset_df)

        split_idx = int(len(dataset) * train_pct)

        train_data = torch.utils.data.Subset(dataset, range(split_idx))
        test_data = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

        if sites_val is not None:
            val_df = dataset_all[
                dataset_all["loc_idx"].astype(int).isin(sites_val)
            ]
            val_data = FineTuneLoader(val_df)
            len_val_data = len(val_data)

            valloader = DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

    else:
        train_data = FineTuneLoader(
            dataset_all,
            cities=cities_train,
            sites=sites_train,
        )

        test_data = FineTuneLoader(
            dataset_all,
            cities=cities_test,
            sites=sites_test,
        )

    trainloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    testloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    print("Fine-tuning dataset loaded.")

    return (
        trainloader,
        testloader,
        len(train_data),
        len(test_data),
        valloader,
        len_val_data,
    )


# ============================================================
# Normalization and preprocessing
# ============================================================

def get_normalization_tensors(device):
    image_normalize = transforms.Normalize(
        mean=[0, 0, 0, 0, 0, 0, 0, 2.5839e02],
        std=[1, 1, 1, 1, 1, 1, 1, 3.3506e02],
    )

    forcing_mean = torch.tensor(
        [
            2.7475e06,
            1.1216e06,
            6.3862e-04,
            9.7475e04,
            3.8102e01,
            2.9312e02,
            9.8286e-01,
            2.6067e-01,
            6.6253e00,
        ],
        device=device,
    )

    forcing_std = torch.tensor(
        [
            6.3123e05,
            2.1583e05,
            3.1177e-03,
            5.1533e03,
            1.3052e01,
            1.0468e01,
            2.4544e00,
            2.6655e00,
            1.8508e00,
        ],
        device=device,
    )

    lst_mean = torch.tensor([301.1683], device=device)
    lst_std = torch.tensor([10.3701], device=device)

    return image_normalize, forcing_mean, forcing_std, lst_mean, lst_std


def process_common_inputs(
    forcing,
    image,
    month,
    lst,
    device,
    image_normalize,
    forcing_mean,
    forcing_std,
    lst_mean,
    lst_std,
):
    image = image.to(device).to(torch.float32)
    forcing = forcing.to(device).to(torch.float32)
    lst = lst.to(device).to(torch.float32)

    month = (month - 1).to(device).to(torch.int64)

    image[:, :5, :, :] = torch.clip(image[:, :5, :, :], min=0, max=1)
    image[:, 5:7, :, :] = torch.clip(image[:, 5:7, :, :], min=-1, max=1)

    image = image_normalize(image)

    forcing = ((forcing - forcing_mean) / forcing_std).to(torch.float32)
    lst = ((lst - lst_mean) / lst_std).to(torch.float32).view(-1, 1)

    return forcing, image, month, lst


def process_pretrain_batch(
    forcing,
    image,
    month,
    lst,
    device,
    image_normalize,
    forcing_mean,
    forcing_std,
    lst_mean,
    lst_std,
):
    return process_common_inputs(
        forcing,
        image,
        month,
        lst,
        device,
        image_normalize,
        forcing_mean,
        forcing_std,
        lst_mean,
        lst_std,
    )


def process_finetune_batch(
    forcing,
    image,
    month,
    t2m,
    lst,
    device,
    image_normalize,
    forcing_mean,
    forcing_std,
    lst_mean,
    lst_std,
):
    forcing, image, month, lst = process_common_inputs(
        forcing,
        image,
        month,
        lst,
        device,
        image_normalize,
        forcing_mean,
        forcing_std,
        lst_mean,
        lst_std,
    )

    t2m = t2m.to(device).to(torch.float32)
    t2m = ((t2m - lst_mean) / lst_std).to(torch.float32).view(-1, 1)

    return forcing, image, month, t2m, lst


# ============================================================
# Model architecture
# ============================================================

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64.")

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in BasicBlock.")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.leaky_relu(out)

        return out


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        block,
        layers,
        in_channel=8,
        groups=1,
        width_per_group=64,
        norm_layer=None,
        zero_init_residual=False,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            in_channel,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = norm_layer(self.inplanes)
        self.avgpool1 = nn.AvgPool2d(2)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights(zero_init_residual)

    def _initialize_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                self.dilation,
                norm_layer,
            )
        ]

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.avgpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool2(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(x)

        return x


class PretrainResNet(nn.Module):
    def __init__(self, dropout_pct=0.02, forcing_shape=9):
        super().__init__()

        self.backbone = ResNetBackbone(BasicBlock, [3, 3, 0, 0])

        self.fc0_1 = nn.Linear(forcing_shape, 64)
        self.fc0_2 = nn.Linear(64 + 12, 128)
        self.fc0_3 = nn.Linear(128, 256)

        self.fc1 = nn.Linear(128 + 256, 512)
        self.drop1 = nn.Dropout(dropout_pct)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, image, forcing, one_hot_mon):
        x = self.backbone(image)

        forcing = self.fc0_1(forcing)
        forcing = F.leaky_relu(forcing)

        forcing = torch.cat((forcing, one_hot_mon), dim=1)

        forcing = self.fc0_2(forcing)
        forcing = F.leaky_relu(forcing)

        forcing = self.fc0_3(forcing)

        x = torch.cat((x, forcing), dim=1)

        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)

        return x


class FineTuneResNet(nn.Module):
    def __init__(self, dropout_pct=0.05, forcing_shape=9):
        super().__init__()

        self.backbone = ResNetBackbone(BasicBlock, [3, 3, 0, 0])

        self.fc0_1 = nn.Linear(forcing_shape, 64)
        self.fc0_2 = nn.Linear(64 + 12, 128)
        self.fc0_3 = nn.Linear(128, 256)

        self.fc1 = nn.Linear(128 + 256, 64)
        self.drop1 = nn.Dropout(dropout_pct)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.fc_skip0_1 = nn.Linear(1, 64)
        self.fc_skip0_2 = nn.Linear(64, 128)
        self.fc_skip0_3 = nn.Linear(128, 1)

    def _skip_connection(self, lst):
        lst = self.fc_skip0_1(lst)
        lst = F.leaky_relu(lst)

        lst = self.fc_skip0_2(lst)
        lst = F.leaky_relu(lst)

        lst = self.fc_skip0_3(lst)

        return lst

    def forward(self, image, forcing, one_hot_mon, lst):
        lst_s = self._skip_connection(lst)

        x = self.backbone(image)

        x = self.drop1(x)

        forcing = self.fc0_1(forcing)
        x = self.drop1(x)
        forcing = F.leaky_relu(forcing)

        forcing = torch.cat((forcing, one_hot_mon), dim=1)

        forcing = self.fc0_2(forcing)
        x = self.drop1(x)
        forcing = F.leaky_relu(forcing)

        forcing = self.fc0_3(forcing)

        x = torch.cat((x, forcing), dim=1)

        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.drop1(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)
        x = x + lst_s

        return x


def build_pretrain_model(dropout_pct=0.02):
    return PretrainResNet(dropout_pct=dropout_pct)


def build_finetune_model(dropout_pct=0.05):
    return FineTuneResNet(dropout_pct=dropout_pct)


# ============================================================
# Loss, checkpoint, and utility functions
# ============================================================

def rmse_loss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


def sigma_li(l_i, tau, lam):
    beta = (l_i - tau) / lam
    sigma = math.e ** (-beta)
    return sigma


def super_loss(l_i, tau, lam):
    sigma = sigma_li(l_i, tau, lam)
    loss = (l_i - tau) * sigma + lam * (torch.log(sigma)) ** 2
    return loss


def load_pretrained_weights(
    model,
    path,
    freeze=False,
    exclude_keys=("fc1", "fc2", "fc3"),
):
    print(f"Loading pretrained weights from: {path}")

    state_dict = torch.load(path, map_location=torch.device("cpu"))
    cleaned_state_dict = OrderedDict()

    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[clean_key] = value

    if exclude_keys is not None:
        cleaned_state_dict = {
            k: v
            for k, v in cleaned_state_dict.items()
            if not any(excluded in k for excluded in exclude_keys)
        }

    model_dict = model.state_dict()
    compatible_dict = {
        k: v
        for k, v in cleaned_state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(compatible_dict)} compatible tensors.")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        print("Trainable parameters after freezing:")

        for name, param in model.named_parameters():
            if any(key in name for key in exclude_keys):
                param.requires_grad = True
                print(name)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def set_random_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)