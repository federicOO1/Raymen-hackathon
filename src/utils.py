import random, torch, numpy as np
from src.loadData import GraphDataset
from torch.serialization import add_safe_globals

add_safe_globals([GraphDataset])  # aggiunge GraphDataset alla lista "sicura"

import os
import torch
from src.loadData import GraphDataset
import pandas as pd
import matplotlib.pyplot as plt
import os, argparse
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from pathlib import Path

def get_data_root(path_str):
    path = Path(path_str)
    parts = path.parts
    for i, part in enumerate(parts):
        if len(part) == 1 and part.isalpha():  # controlla se è una singola lettera
            return str(Path(*parts[:i]))  # restituisce tutto prima di quella lettera
    return str(path.parent)  # fallback: restituisce tutto tranne il file

def save_predictions(predictions, dataset_name):
    os.makedirs("submission", exist_ok=True)
    output_path = f"submission/testset_{dataset_name}.csv"
    df = pd.DataFrame({'id': list(range(len(predictions))), 'pred': predictions})
    df.to_csv(output_path, index=False)
    print(f"📄 Predizioni salvate in: {output_path}")


def load_or_create_datasets(args):
    dataset_names = ["A", "B", "C", "D"]
    full_dataset = {}
    save_dir = args.saved_dataset_dir if hasattr(args, 'saved_dataset_dir') else "saved_datasets"
    os.makedirs(save_dir, exist_ok=True)

    if args.use_saved:
        print("🔄 Caricamento dataset salvati da:", save_dir)
        for name in dataset_names:
            dataset = {}
            for split in ["train", "test"]:
                path = os.path.join(save_dir, f"{name}_{split}.pt")
                if os.path.exists(path):
                    dataset[split] = torch.load(path, weights_only=False)  # ✅ importante
                    print(f"✅ {name} {split} caricato.")
                else:
                    raise FileNotFoundError(f"❌ File mancante: {path}")
            full_dataset[name] = dataset
    else:
        print("📥 Caricamento da file .json.gz e salvataggio in .pt per il futuro...")
        for name in dataset_names:
            dataset = {}
            for split in ["train", "test"]:
                data_root = get_data_root(args.test_path)
                raw_path = os.path.join(data_root, name, f"{split}.json.gz")
                ds = GraphDataset(raw_path)
                dataset[split] = ds
                save_path = os.path.join(save_dir, f"{name}_{split}.pt")
                torch.save(ds, save_path)
                print(f"💾 Salvato: {save_path}")
            full_dataset[name] = dataset

    return full_dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_logs(train_losses, val_losses, val_accuracies, val_f1s, args):
      # Salvataggio CSV
      log_df = pd.DataFrame({
          'epoch': list(range(1, len(train_losses)+1)),
          'train_loss': train_losses,
          'val_loss': val_losses,
          'val_acc': val_accuracies,
          'val_f1': val_f1s
      })
      log_filename = f"logs/log_{args.data_name}.csv"
      log_df.to_csv(log_filename, index=False)
      print(f"📁 Log CSV salvato in: {log_filename}")

      # Plot dei grafici
      plt.figure(figsize=(10, 5))
      plt.subplot(1, 2, 1)
      plt.plot(train_losses, label='Train Loss')
      plt.plot(val_losses, label='Val Loss')
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.title("Loss Curve")
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(val_f1s, label='Val F1 Score', color='green')
      plt.xlabel("Epoch")
      plt.ylabel("F1 Score")
      plt.title("F1 Score Curve")
      plt.legend()

      plot_filename = f"logs/plot_{args.data_name}.png"
      plt.tight_layout()
      plt.savefig(plot_filename)
      plt.close()
      print(f"📊 Grafico salvato in: {plot_filename}")
