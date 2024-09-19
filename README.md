This repository contains the implementation of a Convolutional Neural Network (CNN) model to estimate Aerosol Optical Depth (AOD) using Sentinel-2 satellite data, based on the methods described in the paper *"Aerosol Optical Depth Retrieval for Sentinel-2 Based on Convolutional Neural Network Method"* by Jie Jiang et al., 2023.

## Competition Information
- **Competition Host:** [Solafune - Aerosol Optical Depth Estimate](https://solafune.com/competitions/ca6ee401-eba9-4f7d-95e6-d1b378a17200?menu=about&tab=overview)
- **Preprocessed Data:** The competition organizers provided the preprocessed dataset, so the data preprocessing steps mentioned in the paper were not applied here.
  
## Methodology Overview

The CNN model takes multi-spectral satellite images as input and retrieves the AOD values. The architecture follows the method outlined in the paper, leveraging a series of convolutional layers to extract image features for aerosol retrieval.

### Model Architecture
- 10 Convolution Layers
- 4 Max-Pooling Layers
- 1 Global Max-Pooling Layer
- 1 Fully Connected Layer
- Dropout Regularization for improved generalization

### Performance Metrics
- The model was evaluated using the following metrics:
    - Mean Absolute Error (MAE)
    - Pearson R Coefficient

## Getting Started

### Prerequisites

- Pytorch 2.3.0
- Python 3.12.3
- Tifffile 2024.5.10
- Lightning 2.2.4

### Dataset
The preprocessed dataset is provided by the competition hosts. Ensure that the data is stored in the `data` directory.
### Training
```bash
python main.py --data_dir data --batch_size 128 --num_workers 3 --max_epochs 300 --sainity_check False
```
## References
- Paper Citation: Jiang, J., Liu, J., & Jiao, D. (2023). Aerosol Optical Depth Retrieval for Sentinel-2 Based on Convolutional Neural Network Method, Atmosphere, 14(9), 1400. [Link to paper](https://www.mdpi.com/2073-4433/14/9/1400)
