# Nested Learning for Encoding Visual Responses

This repository primarily tests and evaluates a novel **Nested Learning** architecture for machine learning models. To measure the effectiveness of this architecture, we use the complex task of encoding EEG visual responses as our primary evaluation benchmark.

> **Note:** The nested learning architecture, core machine learning models, training logic, and evaluation pipelines in this repository are our original research contributions. The EEG dataset and data preprocessing methods used as the benchmark standard are adapted from prior work (see Acknowledgements).

## Environment Setup
To run the code, first install [Anaconda][conda], then create and activate a dedicated Conda environment by typing the following into your terminal:

```shell
conda env create -f environment.yml
conda activate nested_encoding
```

## Data Availability & Directory Structure
For our EEG benchmark evaluation, the original dataset and image stimuli must be downloaded and placed into the appropriate directories. You can find the raw and preprocessed data on [OSF][osf] provided by the original authors.

Please ensure your `./data/` folder follows this exact structure before running the models:

```text
data/
├── eeg_dataset/
│   └── preprocessed_data/    # Preprocessed EEG data ready for modeling
│       ├── sub-01/           # Subject 1 preprocessed data
│       ├── sub-02/           # Subject 2 preprocessed data
│       └── ...               # Up to sub-10
└── image_set/                # Image stimuli dataset
    ├── training_images/      # Training image set
    ├── test_images/          # Test image set
    ├── image_metadata.npy    # Metadata for images
    └── LICENSE.txt           # License for images
```

## Code Structure
* **`src/`**: Contains the source code for our core **nested learning** architectures, models, hyperparameter tuning, and visualization.
* **`scripts/`**: Contains utility scripts and tools for auxiliary tasks, such as analyzing hyperparameter tuning logs.

## Acknowledgements and License
The EEG dataset and the preprocessing methodologies used as the evaluation standard in this project are sourced from the following work:
**"A large and rich EEG dataset for modeling human visual object recognition"** by Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy.

The original data and preprocessing code are licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. For details, please see the [LICENSE.md](./LICENSE.md) file.

### Citation
If you use the benchmark dataset or preprocessing code in your own work, please ensure you cite the original paper:
> Gifford AT, Dwivedi K, Roig G, Cichy RM. 2022. A large and rich EEG dataset for modeling human visual object recognition. _NeuroImage_, 264:119754. DOI: [https://doi.org/10.1016/j.neuroimage.2022.119754][paper_link]

[conda]: https://www.anaconda.com/
[osf]: https://osf.io/3jk45/
[paper_link]: https://doi.org/10.1016/j.neuroimage.2022.119754
