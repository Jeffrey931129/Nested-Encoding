import os

import numpy as np
import torch
from PIL import Image
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
data_dir = os.path.join(root_dir, "data")
experiment_dir = os.path.join(root_dir, "experiment")
figure_dir = os.path.join(root_dir, "figures")


def get_idx_val() -> np.ndarray:
    """Calculate and return the indices for validation data."""
    seed = 20200220
    train_img_concepts = np.arange(1654)
    img_per_concept = 10
    val_concepts = np.sort(resample(train_img_concepts, replace=False, n_samples=100, random_state=seed))
    idx_val = np.zeros((len(train_img_concepts) * img_per_concept), dtype=bool)
    for i in val_concepts:
        idx_val[i * img_per_concept : i * img_per_concept + img_per_concept] = True
    return idx_val


def load_images() -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Load and preprocess the training, validation and test images.

    Returns:
        Tuple containing:
            - X_train: Training images.
            - X_val: Validation images.
            - X_test: Test images.
    """

    idx_val = get_idx_val()
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img_dirs = os.path.join(data_dir, "image_set", "training_images")
    image_list = []
    for root, dirs, files in os.walk(img_dirs):
        for file in files:
            if file.endswith(".jpg"):
                image_list.append(os.path.join(root, file))
    image_list.sort()
    X_train = []
    X_val = []
    for i, image in enumerate(tqdm(image_list)):
        img = Image.open(image).convert("RGB")
        img = preprocess(img)
        if idx_val[i] == True:
            X_val.append(img)
        else:
            X_train.append(img)

    img_dirs = os.path.join(data_dir, "image_set", "test_images")
    image_list = []
    for root, dirs, files in os.walk(img_dirs):
        for file in files:
            if file.endswith(".jpg"):
                image_list.append(os.path.join(root, file))
    image_list.sort()
    X_test = []
    for image in tqdm(image_list):
        img = Image.open(image).convert("RGB")
        img = preprocess(img)
        X_test.append(img)

    return X_train, X_val, X_test


def load_eeg_data(sub: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], np.ndarray]:
    """Load the EEG training and test data.

    Args:
        sub: Subject ID.

    Returns:
        Tuple containing:
            - y_train: Training EEG data.
            - y_val: Validation EEG data.
            - y_test: Test EEG data.
            - ch_names: EEG channel names.
            - times: EEG time points.
    """

    idx_val = get_idx_val()
    eeg_data_dir = os.path.join("eeg_dataset", "preprocessed_data", "sub-" + format(sub, "02"))

    training_file = "preprocessed_eeg_training.npy"
    data = np.load(os.path.join(data_dir, eeg_data_dir, training_file), allow_pickle=True).item()
    ch_names = data["ch_names"]
    times = data["times"]
    y_train = data["preprocessed_eeg_data"]
    y_train = np.mean(y_train, 1)
    y_val = y_train[idx_val]
    y_train = np.delete(y_train, idx_val, 0)
    y_train = torch.tensor(np.float32(y_train))
    y_val = torch.tensor(np.float32(y_val))

    test_file = "preprocessed_eeg_test.npy"
    data = np.load(os.path.join(data_dir, eeg_data_dir, test_file), allow_pickle=True).item()
    y_test = data["preprocessed_eeg_data"]
    y_test = np.mean(y_test, 1)
    y_test = torch.tensor(np.float32(y_test))

    return y_train, y_val, y_test, ch_names, times


def create_dataloader(
    batch_size: int,
    g_cpu: torch.Generator,
    X_train: list[torch.Tensor],
    X_val: list[torch.Tensor],
    X_test: list[torch.Tensor],
    y_train: torch.Tensor,
    y_val: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Put the training, validation and test data into a PyTorch-compatible Dataloader format.

    Args:
        batch_size: Batch size for DataLoader.
        g_cpu: Generator object for DataLoader random batching.
        X_train: Training images.
        X_val: Validation images.
        X_test: Test images.
        y_train: Training EEG data.
        y_val: Validation EEG data.
        y_test: Test EEG data.

    Returns:
        Tuple containing:
            - train_dl: Training DataLoader.
            - val_dl: Validation DataLoader.
            - test_dl: Test DataLoader.
    """

    class EegDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.stack(X) if isinstance(X, list) else X
            self.y = torch.reshape(y, (y.shape[0], -1))

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_ds = EegDataset(X_train, y_train)
    val_ds = EegDataset(X_val, y_val)
    test_ds = EegDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g_cpu, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, pin_memory=True)

    return train_dl, val_dl, test_dl
