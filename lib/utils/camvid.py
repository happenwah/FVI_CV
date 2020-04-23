"""
    PyTorch dataloaders for CamVid dataset.
    based on: https://github.com/wjmaddox/swa_gaussian
"""

import os
import torch
from torchvision import transforms
from .camvid_utils import CamVid
from .joint_transforms_seg import (JointRandomResizedCrop,
    JointRandomHorizontalFlip,
    JointCompose,
    LabelToLongTensor,
    )

def camvid_loaders(
    path,
    batch_size,
    num_workers,
    transform_train,
    transform_test,
    shuffle_train=True,
    joint_transform=None,
    ft_joint_transform=None,
    ft_batch_size=1,
    test_batch_size=1,
    **kwargs
                ):

    # load training and finetuning datasets
    print(path)
    train_set = CamVid(
        root=path,
        split="train",
        joint_transform=joint_transform,
        transform=transform_train,
        **kwargs
    )
    ft_train_set = CamVid(
        root=path,
        split="train",
        joint_transform=ft_joint_transform,
        transform=transform_train,
        **kwargs
    )

    val_set = CamVid(
        root=path, split="val", joint_transform=None, transform=transform_test, **kwargs
    )
    test_set = CamVid(
        root=path,
        split="test",
        joint_transform=None,
        transform=transform_test,
        **kwargs
    )

    num_classes = 11  # hard coded labels ehre

    return (
        {
            "train": torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "fine_tune": torch.utils.data.DataLoader(
                ft_train_set,
                batch_size=ft_batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "val": torch.utils.data.DataLoader(
                val_set,
                batch_size=ft_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
            "test": torch.utils.data.DataLoader(
                test_set,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        },
        num_classes,
    )

def get_camvid(batch_size, batch_size_ft, dir_camvid=None):
    H_crop, W_crop = 224, 224
    H, W = 360, 480
    if dir_camvid is None:
        dir_camvid = '/rdsgpfs/general/user/etc15/home/datasets/CamVid'
    print('CamVid train data dir: ', dir_camvid)

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
                    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
                    )

    joint_transform = JointCompose(
        [
            JointRandomResizedCrop(H_crop),
            JointRandomHorizontalFlip(),
        ]
                    )
    ft_joint_transform = JointCompose([JointRandomHorizontalFlip()])

    target_transform = transforms.Compose([LabelToLongTensor()])

    loaders, num_classes = camvid_loaders(
                path=dir_camvid,
                batch_size=batch_size,
                num_workers=0,
                ft_batch_size=batch_size_ft,
                transform_train=transform_train,
                transform_test=transform_test,
                joint_transform=joint_transform,
                ft_joint_transform=ft_joint_transform,
                target_transform=target_transform,
                )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    ft_loader = loaders["fine_tune"]
    return train_loader, val_loader, test_loader, ft_loader, num_classes
