import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
data_dir = os.path.join(root_dir, "data")
experiment_dir = os.path.join(root_dir, "experiment")
figure_dir = os.path.join(root_dir, "figures")

def get_idx_val():
    """Calculate and return the indices for validation data."""
    import numpy as np
    from sklearn.utils import resample
    
    train_img_concepts = np.arange(1654)
    img_per_concept = 10
    val_concepts = np.sort(resample(train_img_concepts, replace=False, n_samples=100, random_state=20200220))
    idx_val = np.zeros((len(train_img_concepts) * img_per_concept), dtype=bool)
    for i in val_concepts:
        idx_val[i * img_per_concept : i * img_per_concept + img_per_concept] = True
    return idx_val


def load_images():
    """Load and preprocess the training, validation and test images.

    Returns
    -------
    X_train : list of tensor
            Training images.
    X_val : list of tensor
            Validation images.
    X_test : list of tensor
            Test images.

    """

    import os
    from torchvision import transforms
    from tqdm import tqdm
    from PIL import Image

    idx_val = get_idx_val()

    ### Define the image preprocesing ###
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ### Load and preprocess the training and validation images ###
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

    ### Load and preprocess the test images ###
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

    ### Output ###
    return X_train, X_val, X_test


def load_eeg_data(sub):
    """Load the EEG training and test data.

    Parameters
    ----------
    sub : int
            Subject ID.

    Returns
    -------
    y_train : tensor
            Training EEG data.
    y_val : tensor
            Validation EEG data.
    y_test : tensor
            Test EEG data.
    ch_names : list of str
            EEG channel names.
    times : float
            EEG time points.

    """

    import os
    import numpy as np
    import torch

    idx_val = get_idx_val()

    ### Load the EEG training data ###
    eeg_data_dir = os.path.join(
        "eeg_dataset", "preprocessed_data", "sub-" + format(sub, "02")
    )
    training_file = "preprocessed_eeg_training.npy"
    data = np.load(
        os.path.join(data_dir, eeg_data_dir, training_file), allow_pickle=True
    ).item()
    y_train = data["preprocessed_eeg_data"]
    ch_names = data["ch_names"]
    times = data["times"]
    # Average across repetitions
    y_train = np.mean(y_train, 1)
    # Extract the validation data
    y_val = y_train[idx_val]
    y_train = np.delete(y_train, idx_val, 0)
    # Convert to float32 and tensor (for DNN training with Pytorch)
    y_train = torch.tensor(np.float32(y_train))
    y_val = torch.tensor(np.float32(y_val))

    ### Load the EEG test data ###
    test_file = "preprocessed_eeg_test.npy"
    data = np.load(
        os.path.join(data_dir, eeg_data_dir, test_file), allow_pickle=True
    ).item()
    y_test = data["preprocessed_eeg_data"]
    # Average across repetitions
    y_test = np.mean(y_test, 1)
    # Convert to float32 and tensor (for DNN training with Pytorch)
    y_test = torch.tensor(np.float32(y_test))

    ### Output ###
    return y_train, y_val, y_test, ch_names, times


def create_dataloader(
    batch_size, time_point, g_cpu, X_train, X_val, X_test, y_train, y_val, y_test
):
    """Put the training, validation and test data into a PyTorch-compatible
    Dataloader format.

    Parameters
    ----------
    batch_size : int
            Batch size for DataLoader.
    time_point : int
            Modeled EEG time point.
    g_cpu : torch.Generator
            Generator object for DataLoader random batching.
    X_train : list of tensor
            Training images.
    X_val : list of tensor
            Validation images.
    X_test : list of tensor
            Test images.
    y_train : float
            Training EEG data.
    y_val : float
            Validation EEG data.
    y_test : float
            Test EEG data.

    Returns
    ----------
    train_dl : Dataloader
            Training Dataloader.
    val_dl : Dataloader
            Validation Dataloader.
    test_dl : Dataloader
            Test Dataloader.

    """

    import torch
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader

    ### Dataset class ###
    class EegDataset(Dataset):
        def __init__(
            self, X, y, time, transform=None, target_transform=None
        ):
            self.time = time
            self.X = X
            self.y = torch.reshape(y, (y.shape[0], -1))
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            image = self.X[idx]
            target = self.y[idx]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                target = self.target_transform(target)
            return image, target

    ### Convert the data to PyTorch's Dataset format ###
    train_ds = EegDataset(X_train, y_train, time_point)
    val_ds = EegDataset(X_val, y_val, time_point)
    test_ds = EegDataset(X_test, y_test, time_point)

    ### Convert the Datasets to PyTorch's Dataloader format ###
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g_cpu)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=test_ds.__len__(), shuffle=False)

    ### Output ###
    return train_dl, val_dl, test_dl
