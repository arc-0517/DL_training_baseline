# CLIP Zero-Shot & Few-Shot Skin Disease Classification

This project provides scripts to perform zero-shot and few-shot classification on a skin disease dataset using OpenAI's CLIP models.

## 1. Setup

### Environment
It is recommended to use a virtual environment (e.g., conda or venv).

```bash
conda create -n clip_exp python=3.9
conda activate clip_exp
```

### Dependencies
Install the required packages using `requirements_clip.txt`.

```bash
pip install -r requirements_clip.txt
```
If you encounter issues with `clip`, you can install it directly from the source:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### Dataset
The scripts expect the following directory structure for the skin disease dataset:
```
data/skin_dataset/
├── Training/
│   └── 01.원천데이터/
│       ├── acne/
│       │   ├── image1.png
│       │   └── ...
│       ├── atopic/
│       └── ...
└── Validation/
    └── 01.원천데이터/
        ├── acne/
        ├── atopic/
        └── ...
```
The paths can be adjusted via command-line arguments.

## 2. Zero-Shot Classification

This method uses the pre-trained CLIP model to classify images based on their similarity to text prompts, without any additional training.

### How to Run
Execute `clip_zero_shot.py` with the desired arguments.

**Basic Example:**
This command runs zero-shot classification using the `ViT-B/32` model on all classes found in the validation directory.

```bash
python clip_zero_shot.py \
    --model_name "ViT-B/32" \
    --val_dir "data/skin_dataset/Validation/01.원천데이터"
```

**Example with Specific Classes:**
To run on a subset of classes (e.g., acne, normal, rosacea):

```bash
python clip_zero_shot.py \
    --model_name "ViT-B/16" \
    --classes acne normal rosacea \
    --val_dir "data/skin_dataset/Validation/01.원천데이터"
```

### Arguments (`clip_zero_shot.py`)
- `--val_dir`: Path to the validation data directory. (Default: `data/skin_dataset/Validation/01.원천데이터`)
- `--results_dir`: Directory to save results (JSON and confusion matrix). (Default: `results/clip_zero_shot`)
- `--model_name`: CLIP model to use. (Choices: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `RN50`, `RN101`. Default: `ViT-B/32`)
- `--classes`: (Optional) A list of specific class names to use. If not provided, all subdirectories in `val_dir` will be used as classes.
- `--batch_size`: Batch size for feature extraction. (Default: 32)
- `--num_workers`: Number of CPU workers for the data loader. (Default: auto-detect)

## 3. Few-Shot Classification

This method uses a small number of labeled examples (the "support set") from the training data to adapt the model for better classification on the validation data (the "query set"). Two methods are supported: K-Nearest Neighbors (K-NN) and Linear Probe.

### How to Run
Execute `clip_few_shot.py` and specify the method (`knn` or `linear_probe`).

**Example 1: K-NN Few-Shot**
This command runs 10-shot classification using a K-NN classifier with `k=5`.

```bash
python clip_few_shot.py \
    --method knn \
    --n_shot 10 \
    --k 5 \
    --model_name "ViT-B/32"
```

**Example 2: Linear Probe Few-Shot**
This command runs 10-shot classification by training a linear probe (a small MLP) on top of the CLIP features.

```bash
python clip_few_shot.py \
    --method linear_probe \
    --n_shot 10 \
    --model_name "ViT-B/32" \
    --lr 0.005 \
    --epochs 300
```

### Arguments (`clip_few_shot.py`)
- `--method`: The few-shot method to use. (Choices: `knn`, `linear_probe`. Default: `linear_probe`)
- `--train_dir`: Path to the training data directory. (Default: `data/skin_dataset/Training/01.원천데이터`)
- `--val_dir`: Path to the validation data directory.
- `--results_dir`: Directory to save results. (Default: `results/clip_few_shot`)
- `--model_name`: CLIP model to use. (Default: `ViT-B/32`)
- `--classes`: (Optional) A list of specific classes to use.
- `--n_shot`: Number of support samples per class. (Default: 10)
- `--seed`: Random seed for reproducibility. (Default: 42)

**K-NN Specific Arguments:**
- `--k`: The number of neighbors for the K-NN classifier. (Default: 3)

**Linear Probe Specific Arguments:**
- `--lr`: Learning rate for training the linear probe. (Default: 0.01)
- `--weight_decay`: Weight decay for the optimizer. (Default: 1e-4)
- `--epochs`: Maximum number of training epochs. (Default: 200)
- `--hidden_dim`: Size of the hidden layer in the MLP. (Default: 256)
- `--dropout`: Dropout rate in the MLP. (Default: 0.3)
- `--patience`: Patience for early stopping. (Default: 20)

## 4. Results
All experiments will save their results in the specified `--results_dir`. Each run creates a timestamped folder containing:
- `results_*.json`: A JSON file with detailed performance metrics (accuracy, F1, precision, recall).
- `cm_*.png`: A PNG image of the confusion matrix.