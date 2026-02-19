# EEG Encoding Steps

## 01_data_preparation

Convert the source EEG data into raw EEG data, reformat the resting state data, and extract behavioral results.

### `source_to_raw.py`

- **Input**: `project_directory/eeg_dataset/source_data`
- **Output**: `project_directory/eeg_dataset/raw_data`
- **Action**: 將二進位數據轉換成 numpy 格式。

## 02_eeg_preprocessing

Preprocess the raw EEG data.

### `preprocessing.py`

- **Input**: `project_directory/eeg_dataset/raw_data`
- **Output**: `project_directory/eeg_dataset/preprocessed_data`
- **Action**: 進行數據過濾與預處理。

## 03_dnn_feature_maps_extraction

Extract the feature maps of all images using four DNN architectures (AlexNet, ResNet-50, CORnet-S, MoCo), and downsample them using principal component analysis (PCA).

### `extract_feature_maps_alexnet.py`

- **Input**: `project_directory/image_set`
- **Output**: `project_directory/dnn_feature_maps/full_feature_maps`
- **Action**: 將圖片轉成高維度的向量

### `feature_maps_pca.py`

- **Input**: `project_directory/dnn_feature_maps/full_feature_maps`
- **Output**: `project_directory/dnn_feature_maps/pca_feature_maps`
- **Action**: 降維
