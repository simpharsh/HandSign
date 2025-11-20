# HandSign - ASL Alphabet Recognition

This project implements a machine learning model to recognize American Sign Language (ASL) alphabets from images.

## Project Structure

- `main.ipynb`: The main Jupyter Notebook containing the data loading, preprocessing, model training, and evaluation code.
- `requirements.txt`: List of Python dependencies required to run the project.
- `asl_model.tflite`: TFLite version of the trained model.
- `best_model.h5`: The best performing Keras model saved during training.

## Setup

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Open the `main.ipynb` notebook using Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook main.ipynb
    ```
2.  Run the cells in the notebook to train the model or evaluate it.

## Dataset

The dataset used is the ASL Alphabet dataset, which should be extracted into the project directory (folders `asl_alphabet_train` and `asl_alphabet_test`).
