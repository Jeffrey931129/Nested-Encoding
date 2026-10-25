# Nested Learning for Visual-to-EEG Encoding

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](./LICENSE.md)

This repository contains the implementation and experimental framework for the research proposal **"Nested Learning for Visual-to-EEG Encoding"**.

Predicting electroencephalography (EEG) signals from visual stimuli is a complex challenge due to the intricate spatial and temporal dynamics of brain responses and inherently low signal-to-noise ratio (SNR). Inspired by the biological multi-timescale updating mechanisms of the human brain introduced in **Nested Learning** ([Behrouz et al., NeurIPS 2025][nested_learning]), this project investigates how nested optimization paradigms and multi-timescale mechanisms can be adapted to visual-to-EEG encoding.

---

## 📌 Key Highlights & Research Contributions

1. **Hybrid Spatiotemporal Architecture (`AlexEEGNet`)**:
   - Extends the purely convolutional baseline (`AlexNetLite`) by coupling an AlexNet spatial visual feature extractor with a Long Short-Term Memory (LSTM) sequence model and channel decoder to effectively capture temporal brain dynamics across channels and time points.
2. **Macroscopic State Optimizer (`NestedAdam`)**:
   - A novel optimizer integrating macroscopic optimization states. It maintains standard local first and second moments ($m^{\mathrm{fast}}, v$) updated at every step, while accumulating gradients over a macro period ($K$ steps) to update a slower macroscopic first moment ($m^{\mathrm{slow}}$).
3. **Three Multi-Timescale Mechanisms**:
   - Systematic exploration of structural interpretations for biological multi-timescale learning:
     - **Asynchronous updates**: Update optimizer moments at every step, but apply parameter updates to specific layer groups at different intervals.
     - **Layer-wise batching**: Buffer data across multiple steps to optimize different network depths with varying effective batch sizes.
     - **Layer-wise learning rates**: Assign hierarchical learning-rate scales reflecting disparate timescales across network representations.

---

## 📁 Repository Structure

```text
nested_encoding/
├── data/                      # Dataset directory (EEG epochs & stimulus images)
├── experiment/                # Experiment logs, grid search results, and checkpoints
├── figures/                   # Evaluation plots, correlation dynamics, and noise ceilings
├── scripts/                   # Utility and analysis scripts
│   ├── analyze.py             # Parses tuning logs & analyzes loss landscapes
│   ├── args.py                # Inspects saved configuration parameters (.npy)
│   └── visualize.py           # Plots spatiotemporal correlation & noise ceiling curves
├── src/                       # Core source code
│   ├── data_utils.py          # Data loading, caching, and DataLoader utilities
│   ├── models.py              # Model definitions (AlexNetLite & AlexEEGNet)
│   ├── optimizers.py          # PyTorch implementation of NestedAdam optimizer
│   ├── train.py               # Main training pipeline, early stopping & evaluation
│   └── tune.py                # Hyperparameter grid search script
├── environment.yml            # Conda environment specification
├── LICENSE.md                 # License file
└── README.md                  # Project documentation
```

---

## 🛠️ Environment Setup

Install [Anaconda][conda] or Miniconda, then create and activate the dedicated environment:

```shell
conda env create -f environment.yml
conda activate nested_encoding
```

---

## 💾 Data Availability & Directory Structure

This project uses the visual object recognition EEG dataset from [Gifford et al. (2022)][paper_link]. Raw and preprocessed data are hosted on [OSF][osf].

Ensure the `./data/` folder follows this structure before executing models:

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

---

## 📚 References & Acknowledgements

1. **Nested Learning**:
   > Behrouz, A., Mirrokni, V., et al. (2025). *Nested Learning: The Illusion of Deep Learning Architectures*. Advances in Neural Information Processing Systems (NeurIPS 2025).
2. **EEG Visual Benchmark Dataset**:
   > Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). A large and rich EEG dataset for modeling human visual object recognition. *NeuroImage*, 264, 119754. DOI: [10.1016/j.neuroimage.2022.119754][paper_link].
3. **Visual Encoding Foundations**:
   > Seeliger, K., et al. (2018). Convolutional neural network-based encoding and decoding of visual object recognition in space and time. *NeuroImage*, 180, 253–266.

The EEG benchmark data and preprocessing methods are licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. See [LICENSE.md](./LICENSE.md) for details.

[conda]: https://www.anaconda.com/
[osf]: https://osf.io/3jk45/
[paper_link]: https://doi.org/10.1016/j.neuroimage.2022.119754
[nested_learning]: https://arxiv.org/abs/2410.23160
