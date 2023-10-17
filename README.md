# servier-technical-test
Servier Data Science Technical Test

# Servier Technical Test

This project is a Technical Test consisting of a machine learning application for predicting molecular properties. It includes training, evaluation, and prediction modules, and Docker Deployement.

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Setup](#setup)
- [Usage](#usage)
- [Dockerization](#dockerization)

## Overview

This machine learning application predicts the binary property 'P1' of a molecule given its SMILES representation. It includes two models for predicting single and multiple properties and is deployable as a Docker container.

## Models

### Model1

This model takes the extracted features of a molecule as input and predicts the P1 property. The features are extracted from the molecule's SMILES string using a provided feature extraction method.

### Model2

Model2 takes the SMILES string characters as input and predicts the 'P1' property. It employs an LSTM network to process the sequence data.

<!-- ### Model3 (Optional)

An extension of Model1 or Model2 that predicts multiple properties (P1, P2, ..., P9) for a given molecule. -->

## Setup

### Prerequisites

- Python 3.6 or later
- Pip (Python Package Installer)
- Docker (for Dockerization)

### Installation

1. Clone this repository:

```bash
git clone https://github.com/ouahbi13/servier-technical-test.git
cd servier-technical-test
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Install the package:

```bash
pip install .
```

## Usage

### Training

Train the model using the following command:

```bash
servier train 
    [--model_type] model_1 or model_2
    [--epochs] Number of training epochs
    [--batch_size] Batch Size
```

### Evaluation

Evaluate the model's performance using Cross Validation on the Overall Dataset:

```bash
servier evaluate
    [--model_type] model_1 or model_2
    [--epochs] Number of training epochs
    [--batch_size] Batch Size
    [--n_splits] Number of splits
```

### Prediction

Predict the P1 property for a given SMILES string:

```bash
servier predict
    [--smiles] SMILES
```

## Dockerization

### Building the Docker Image

Build the Docker image using the following command:

```bash
docker build . -t servier
```

### Running the Docker Container

Run the Docker container with:

```bash
docker run -p 5000:80 servier
```

### Using Data with Docker

Use Docker volumes to access the dataset on your local machine:

```bash
docker run -p 5000:80 -v servier/data:/app/data servier
```