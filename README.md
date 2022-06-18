# genre-classifier

## Description

This is a machine learning project with the goal of creating a neural network
capable of predicting the genre of an audio track.

At a high-level, this is split into three steps:
1. Creating a trainable dataset using feature engineering
2. Configuring the model
3. Training the model

This project is an end-to-end implementation; each of the three steps can be
accomplished using a single command-line argument.

The dataset used is from the **Free Music Archive (FMA)**. Due to both hardware
requirements and ease of training, it is recommend to use the small variant
which is composed of 8,000 MP3-encoded tracks equally divided amongst eight
genres.

## Getting Started

### Dependencies

- **[Free Music Archive (FMA)](https://github.com/mdeff/fma)** dataset:
    - The small variant is recommended:
        - **fma_small.zip**
    - Per track metadata file:
        - **tracks.csv**

- **Python**
    - 3.8.10

- **TensorFlow**
    - 2.9.1

- **Hardware capable of machine learning**
    - An Nvidia GPU with at least 6GB of memory is recommended
    - 32GB of RAM is recommended

### Installing

1. Download (and extract) the FMA dataset and per track metadata file
referenced above
2. Clone repository
3. Activate virtual environment
4. Install dependencies using requirements.txt

### Executing program

1. Create trainable dataset to be saved as *outfile* using the directory
*tracks_filepath* containing the FMA dataset and the per track metadata saved
at *metadata_filepath*

```python dataset/create.py [-h] tracks_filepath metadata_filepath outfile```

2. Configure the model to be saved as *export* to the trainable dataset at
*dataset_filepath*

```python model/configure.py [-h] dataset_filepath export```

3. Train the model saved at *model_filepath* using the trainable dataset at
*dataset_filepath*. By default, the model will be overwritten by the trained
version; to keep the untrained model, invoke the optional ```-e``` or
```--export``` argument followed by the filepath to the save location

```python model/train.py [-h] [-e EXPORT] model_filepath dataset_filepath```

## Help

Invoke the optional ```-h``` or ```--help``` argument in the previously
mentioned files to display help messages regarding usage.

## Notes

Due to the size of the FMA dataset, creating the trainable dataset can take
some time. As a reference, expect 15 to 20 minutes for the small variant.

Downloading the dataset will result in warnings being displayed regarding
a PySoundFile failure. This is due to a dependency not directly supporting
MP3-encoded files but does not affect program functionality.

## Authors

[Brooks Burns](https://github.com/Brrookss)