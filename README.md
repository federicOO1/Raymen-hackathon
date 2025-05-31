# Deep Learning Project | Raymen

**Dataset Source**  
The datasets used in this project are available at:  
[https://drive.google.com/drive/folders/1Z-1JkPJ6q4C6jX4brvq1VRbJH5RPUCAk?usp=drive_link)

---

## Description
This project aims to achieve a high F1 score in graph classification with noisy labels. The model created has been trained on 4 separate datasets, involving a series of steps including pre-training, fine-tuning, and the use of an ensemble of models. The final model was used to generate predictions on test data, producing a CSV file with predictions for each dataset.

### Key Steps:
- **Pre-training:** The model is pre-trained on all four datasets using a Graph Neural Network (GNN) model (Gin with 4 layers). This general pre-training step helps the model generalize better across datasets.
- **Fine-tuning:** After pre-training, fine-tuning is performed using dataset-specific training data to further optimize the model for each dataset.
- **Ensemble:** For some datasets (C and D), an ensemble of two fine-tuned models is used to combine predictions, improving accuracy and robustness.

## Project Structure
The repository is organized as follows:

- **checkpoints/**: Contains the trained model checkpoints.
- **source/**: Contains the code for the model, loss functions, data loaders, and algorithms.
- **submission/**: Contains the CSV files with the predictions generated for each test dataset (testset_A.csv, testset_B.csv, testset_C.csv, testset_D.csv).
- **logs/**: Contains log files for each training dataset, including details on accuracy, loss, and other metrics.
- **requirements.txt**: Lists the Python dependencies required to run the project.
- **README.md**: This documentation file.

## Workflow

### 1. **Pre-training**
The model was pre-trained on **all four datasets** (A, B, C, D). The architecture used for pre-training is a Graph Isomorphism Network (GIN) with 4 layers, which is a crucial step to improve generalization across datasets. The same pre-trained model is used for all datasets.

### 2. **Fine-tuning**
After pre-training, **fine-tuning** is performed on the pre-trained model using the training data from each specific dataset. This step further optimizes the model for the unique characteristics of each dataset, improving overall performance.

### 3. **Ensemble**
For datasets **C** and **D**, an **ensemble approach** was applied. Two different fine-tuned models are combined to generate the final prediction, enhancing the robustness and accuracy of the results.

## How to Use the Code

### 1. **General Flags:**

- `--use_saved`:  
  If set, it will use a previously saved dataset without reprocessing or reloading data. If used, it requires the use of `--saved_dataset_dir` flag.
  
- `--start_pretrain`:  
  If set, it will restarts the pre-train process of the whole dataset.

- `--ensamble_test`:  
  Enables ensemble testing where multiple models are used to generate predictions. This can improve performance by averaging predictions from different models.

### 2. **Directories and Dataset Parameters:**

- `--saved_dataset_dir`:  
  Specifies the directory where saved datasets are located (default is `saved_datasets`).

- `--data_root`:  
  Defines the root directory where the data files are stored (default is `data`).

- `--data_name`:  
  Specifies the name of the dataset to use. Available options are 'A', 'B', 'C', and 'D' (default is 'B').

### 3. **Training Parameters:**

- `--epochs`:  
  The number of epochs for training the model (default is 20). This determines how many times the model will iterate over the entire training dataset.

- `--loss_fn`:  
  Specifies the loss function to use during training. Options include:
  - `ce`: Cross-entropy loss
  - `gce`: Generalized cross-entropy loss
  - `focal`: Focal loss
  - `lsl`: Log-Sigmoid loss
  - `sce`: Symmetric cross-entropy loss
  - `cbf`: Custom balanced focal loss
  - `focal_symmetric`: Focal-Symmetric cross-entropy loss

- `--batch_size`:  
  The batch size used during training, which controls how many samples are processed in each iteration (default is 32).

- `--lr`:  
  The learning rate for the optimizer, controlling how much to adjust the model weights (default is 0.0001).

### 4. **Model Hyperparameters:**

- `--emb_dim`:  
  The embedding dimension for the model (default is 300). This defines the size of the embedding vectors used in the model.

- `--drop_ratio`:  
  The dropout ratio used to prevent overfitting. A dropout ratio of 0.5 gave the best results for this project.

- `--p_noisy`:  
  The probability of having noisy labels in the dataset (default is 0.3). This simulates noisy data scenarios.

- `--temperature`:  
  The temperature parameter used in certain loss functions to scale the logits (default is 1.2).

### 5. **Early Stopping:**

- `--patience`:  
  The number of epochs with no improvement in the validation loss before the training process stops early. This helps prevent overfitting by stopping training when the model stops improving (default is 5).

## How to Train and Test the Model

### 1. **Training the Model:**
To train the model on a specific dataset, use the following command:

```bash
python main.py --train_path <path_to_train.json.gz> --test_path <path_to_test.json.gz>