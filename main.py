import os
import torch
import argparse
import pandas as pd
import glob
import re
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from src.models import GNN
from src.loadData import GraphDataset
from src.utils import set_seed, load_or_create_datasets, save_logs, save_predictions
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pathlib import Path

'''def ensemble_predictions(model1, model2, loader, device):
    """
    Funzione che fa l'ensemble tra due modelli e calcola le predizioni mediate.
    """
    model1.eval()
    model2.eval()

    all_preds = []
    with torch.no_grad():
        for data in tqdm(loader, desc="🔍 Inference with ensemble"):
            data = data.to(device)
            out1 = model1(data)
            out2 = model2(data)

            # Calcolare le probabilità mediate
            probs1 = F.softmax(out1, dim=1)
            probs2 = F.softmax(out2, dim=1)

            avg_probs = (probs1 + probs2) / 2  # Media delle probabilità

            pred = avg_probs.argmax(dim=1)  # Predizione finale
            all_preds.extend(pred.cpu().numpy())

    return all_preds
    '''

def load_model(model_path, device):
    """
    Loads a saved GNN model.
    """
    # Define the model architecture (this should match the saved model)
    model = GNN(
        num_class=6,  # Assuming 6 classes based on your code
        gnn_type='gin', # Assuming gin based on your code
        num_layer=4,  # Assuming 4 layers
        emb_dim=300,  # Assuming 300 embedding dimensions
        drop_ratio=0.5, # Assuming 0.0 drop ratio
        JK='sum',     # Assuming JK='sum'
        residual=True, # Assuming residual=True
        virtual_node=True, # Assuming virtual_node=True
        graph_pooling='attention' # Assuming graph_pooling='attention'
    ).to(device)

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    return model

class EnsembleModel(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, data):
        # Calcola le predizioni di entrambi i modelli
        out1 = self.model1(data)
        out2 = self.model2(data)

        # Media delle probabilità per l'ensamble
        probs1 = F.softmax(out1, dim=1)
        probs2 = F.softmax(out2, dim=1)
        avg_probs = (probs1 + probs2) / 2

        return avg_probs

def save_ensemble_model(model, save_path):
    """
    Salva il modello ensemble in formato .pth
    """
    torch.save(model.state_dict(), save_path)
    print(f"📦 Modello dell'Ensemble salvato in: {save_path}")


# === Early Stopper ===
class EarlyStopper:
    def __init__(self, patience=5, mode='max'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode  # 'max' or 'min'

    def __call__(self, current_score):
        if self.best_score is None or \
           (self.mode == 'max' and current_score > self.best_score) or \
           (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# === Label Smooting Loss ===
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, x, target):
        pred = x.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# === WeightedSymmetricCrossEntropy Loss ===
class WeightedSymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=2, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        label_one_hot = F.one_hot(labels, self.num_classes).float()

        # Standard CE
        ce = -torch.sum(label_one_hot * torch.log(pred), dim=1)

        # Reverse CE
        rce = -torch.sum(pred * torch.log(label_one_hot + 1e-7), dim=1)

        # Applica pesi per classe se forniti
        if self.class_weights is not None:
            weights = self.class_weights[labels]
            ce = ce * weights
            rce = rce * weights

        return self.alpha * ce.mean() + self.beta * rce.mean()

# === FocalSymmetric CrossEntropy Loss ===
class FocalSymmetricCrossEntropy(nn.Module):
    """Combina SCE con Focal Loss per gestire sbilanciamento + rumore"""
    def __init__(self, alpha=0.1, beta=1.0, gamma=2.0, num_classes=6, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        label_one_hot = F.one_hot(labels, self.num_classes).float()

        # Standard CE con componente focal
        ce = -torch.sum(label_one_hot * torch.log(pred), dim=1)
        pt = torch.sum(label_one_hot * pred, dim=1)
        focal_weight = (1 - pt) ** self.gamma
        focal_ce = focal_weight * ce

        # Reverse CE
        rce = -torch.sum(pred * torch.log(label_one_hot + 1e-7), dim=1)

        # Applica pesi per classe
        if self.class_weights is not None:
            weights = self.class_weights[labels]
            focal_ce = focal_ce * weights
            rce = rce * weights

        return self.alpha * focal_ce.mean() + self.beta * rce.mean()


# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()

# === Symmetric Cross Entropy Loss ===
class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        # Standard CE
        ce = -torch.sum(F.one_hot(labels, self.num_classes) * torch.log(pred), dim=1)

        # Reverse CE
        rce = -torch.sum(pred * torch.log(F.one_hot(labels, self.num_classes) + 1e-7), dim=1)

        return self.alpha * ce.mean() + self.beta * rce.mean()

# === Enhanced Noisy Loss ===
class EnhancedNoisyCrossEntropyLoss(nn.Module):
    def __init__(self, p_noisy=0.2, temperature=1.0, label_smoothing=0.1, min_weight=0.1, max_weight=1.0):
        super().__init__()
        self.p = p_noisy
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.num_epochs = None
        self.current_epoch = None

    def set_epoch_info(self, num_epochs, current_epoch):
        self.num_epochs = num_epochs
        self.current_epoch = current_epoch

    def forward(self, logits, targets):
        scaled_logits = logits / self.temperature
        losses = self.ce(scaled_logits, targets)
        probs = torch.softmax(scaled_logits, dim=1)
        max_probs, _ = probs.max(dim=1)
        base_weight = (1 - self.p) + self.p * (1 - max_probs)
        if self.num_epochs is not None and self.current_epoch is not None:
            epoch_progress = self.current_epoch / self.num_epochs
            epoch_weight = 1 - (1 - self.p) * epoch_progress
            base_weight = base_weight * epoch_weight
        weights = torch.clamp(base_weight, self.min_weight, self.max_weight)
        return (losses * weights).mean()

# === Train & Evaluate ===
def train(loader, model, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    if hasattr(criterion, 'set_epoch_info'):
        criterion.set_epoch_info(total_epochs, epoch)
    total_loss, correct = 0, 0
    for data in tqdm(loader, desc=f"Train {epoch+1}", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(loader, model, device, criterion=None):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for data in tqdm(loader, desc="Validating", leave=False):
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(data.y.cpu().numpy())
            if criterion is not None:
                loss = criterion(out, data.y)
                total_loss += loss.item()

    acc = sum([p == t for p, t in zip(preds, targets)]) / len(preds)
    f1 = f1_score(targets, preds, average='macro')
    avg_loss = total_loss / len(loader) if criterion is not None else None
    return acc, f1, avg_loss

def predict(loader, model, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data in tqdm(loader, desc="🔍 Inference test"):
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
    return all_preds

class EnsembleModel(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, data):
        # Calcola le predizioni di entrambi i modelli
        out1 = self.model1(data)
        out2 = self.model2(data)

        # Media delle probabilità per l'ensamble
        probs1 = F.softmax(out1, dim=1)
        probs2 = F.softmax(out2, dim=1)
        avg_probs = (probs1 + probs2) / 2

        return avg_probs

def save_ensemble_model(model, save_path):
    """
    Salva il modello ensemble in formato .pth
    """
    torch.save(model.state_dict(), save_path)
    print(f"📦 Modello dell'Ensemble salvato in: {save_path}")

# === Main Pipeline ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=None, help='Percorso al file .pt con il dataset di training')
    parser.add_argument('--test_path', type=str, required=True, help='Percorso al file .pt con il dataset di test')
    parser.add_argument('--use_saved', action='store_true')
    parser.add_argument('--start_pretrain', action='store_true')
    parser.add_argument('--ensamble_test', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--saved_dataset_dir', type=str, default='saved_datasets')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--data_name', type=str, default='B')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--loss_fn', type=str, default='cbf', choices=['ce','gce','focal','lsl', 'sce','cbf','focal_symmetric'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--drop_ratio', type=float, default=0.5)
    parser.add_argument('--p_noisy', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    args = parser.parse_args()

    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = load_or_create_datasets(args)

    if args.ensamble_test:
        assert args.data_name is not None, "❗ Specifica --data_name (A, B, C, D)"
        dataset_name = args.data_name
        print(f"\n🧪 Test sul dataset: {dataset_name}")
        test_dataset = full_dataset[dataset_name]["test"]
        print(f"✅ Dataset di test caricato: {len(test_dataset)} grafi")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


        # Carica i modelli pre-addestrati
        model1_path = "/content/drive/MyDrive/DeepLearning3/checkpoints/finetuned_D_f1=0.81.pth"
        model2_path = "/content/drive/MyDrive/DeepLearning3/checkpoints/finetuned_D_f1=0.82.pth"
        model1 = load_model(model1_path, device)
        model2 = load_model(model2_path, device)

            # Define model architecture before loading state_dict
        model1 = GNN(
            num_class=6,  # Assuming 6 classes based on your code
            gnn_type='gin', # Assuming gin based on your code
            num_layer=4,  # Assuming 4 layers
            emb_dim=300,  # Assuming 300 embedding dimensions
            drop_ratio=0.5, # Assuming 0.0 drop ratio
            JK='sum',     # Assuming JK='sum'
            residual=True, # Assuming residual=True
            virtual_node=True, # Assuming virtual_node=True
            graph_pooling='attention' # Assuming graph_pooling='attention'
        ).to(device)
        model2 = GNN(
            num_class=6,  # Assuming 6 classes based on your code
            gnn_type='gin', # Assuming gin based on your code
            num_layer=4,  # Assuming 4 layers
            emb_dim=300,  # Assuming 300 embedding dimensions
            drop_ratio=0.5, # Assuming 0.0 drop ratio
            JK='sum',     # Assuming JK='sum'
            residual=True, # Assuming residual=True
            virtual_node=True, # Assuming virtual_node=True
            graph_pooling='attention' # Assuming graph_pooling='attention'
        ).to(device)

        model1.load_state_dict(torch.load(model1_path))
        model2.load_state_dict(torch.load(model2_path))

        # Set models to evaluation mode
        model1.eval()
        model2.eval()

        # Create the ensemble model
        ensemble_model = EnsembleModel(model1, model2).to(device)


           # Salva il modello dell'ensamble
        ensemble_model_path = f"checkpoints/ensemble_model_{dataset_name}_f1=0.81e82.pth"  # Puoi modificare il nome come preferisci
        save_ensemble_model(ensemble_model, ensemble_model_path)


        # Calcola le predizioni usando il modello dell'ensamble
        all_predictions = []
        with torch.no_grad():
            for data in tqdm(test_loader, desc="🔍 Inference with ensemble"):
                data = data.to(device)
                avg_probs = ensemble_model(data) # The EnsembleModel forward now returns avg_probs
                pred = avg_probs.argmax(dim=1).cpu().numpy()
                all_predictions.extend(pred)


        save_predictions(all_predictions, dataset_name)
        print("🏁 Test completato e modello dell'ensamble salvato.")
        return  # esci se test



    if args.start_pretrain:
        all_train_graphs = []
        for name in full_dataset:
            all_train_graphs.extend(full_dataset[name]["train"])
        pretrain_size = int(0.8 * len(all_train_graphs))

        val_size = len(all_train_graphs) - pretrain_size
        pretrain_data, preval_data = random_split(all_train_graphs, [pretrain_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(pretrain_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(preval_data, batch_size=args.batch_size, shuffle=False)

        model = GNN(num_class=6, gnn_type='gcn', num_layer=3, emb_dim=args.emb_dim,
                    drop_ratio=args.drop_ratio, JK='sum', residual=True,
                    virtual_node=True, graph_pooling='attention').to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        if args.loss_fn == 'ce':
            criterion = torch.nn.CrossEntropyLoss()
        elif args.loss_fn == 'gce':
          criterion = EnhancedNoisyCrossEntropyLoss(p_noisy=args.p_noisy, temperature=args.temperature)
        elif args.loss_fn == 'focal':
          criterion = FocalLoss()
        elif args.loss_fn == 'sce':
              criterion = SymmetricCrossEntropy(num_classes=6)
        elif args.loss_fn == 'lsl':
          criterion = LabelSmoothingLoss(classes=6)
        elif args.loss_fn == 'focal_symmetric':
                  train_labels = []
                  for data in train_loader:
                      train_labels.extend(data.y.cpu().numpy())

                  class_weights = compute_class_weight('balanced',
                                                    classes=np.unique(train_labels),
                                                    y=train_labels)
                  class_weights = torch.FloatTensor(class_weights).to(device)
                  criterion = FocalSymmetricCrossEntropy(alpha=0.1, gamma=2.0, num_classes=6, class_weights=class_weights)
        else:
          raise ValueError(f"Loss function {args.loss_fn} non supportata.")
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_f1=0
        early_stopper=EarlyStopper(patience=args.patience, mode='min')

        for epoch in range(args.epochs):
            loss, acc = train(train_loader, model, optimizer, criterion, device, epoch, args.epochs)
            val_acc, val_f1, val_loss = evaluate(val_loader, model, device, criterion)
            print(f"🌍 Pretrain Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f} | TrainAcc: {acc:.4f} | ValLoss: {val_loss:.4f} |ValAcc: {val_acc:.4f} | ValF1: {val_f1:.4f}")

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"📉 LR attuale: {current_lr:.2e}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), f"checkpoints/pretrained_model.pth")
                print(f"💾 Salvato miglior modello pretrain (F1={best_f1:.4f})")
            if early_stopper(val_loss):
                print(f"🛑 Early stopping attivato.")
                break

        print("✅ Pretraining completato.")


    if args.train_path is not None:

        for name in full_dataset:
            dataset_name = Path(args.train_path).parts[-2]
            if name == dataset_name:
                  print(f"\n📁 Fine-tuning su dataset: {name}")
                  dataset = full_dataset[name]["train"]
                  train_size = int(0.8 * len(dataset))
                  val_size = len(dataset) - train_size
                  train_data, val_data = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
                  train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
                  val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

                  model_ft = GNN(
                      num_class=6,
                      gnn_type='gin',
                      num_layer=4,
                      emb_dim=args.emb_dim,
                      drop_ratio=args.drop_ratio,
                      JK='sum',
                      residual=True,
                      virtual_node=True,
                      graph_pooling='attention'
                  ).to(device)
                  model_ft.load_state_dict(torch.load("checkpoints/pretrained_model.pth"))
                  optimizer = torch.optim.Adam(model_ft.parameters(), lr=args.lr, weight_decay=1e-2)
                  if args.loss_fn == 'ce':
                      criterion = torch.nn.CrossEntropyLoss()
                  elif args.loss_fn == 'gce':
                      criterion = EnhancedNoisyCrossEntropyLoss(p_noisy=args.p_noisy, temperature=args.temperature)
                  elif args.loss_fn == 'sce':
                      criterion = SymmetricCrossEntropy(alpha=0.05, num_classes=6)
                  elif args.loss_fn == 'focal':
                      criterion = FocalLoss()
                  elif args.loss_fn == 'focal_symmetric':
                      train_labels = []
                      for data in train_loader:
                          train_labels.extend(data.y.cpu().numpy())

                      class_weights = compute_class_weight('balanced',
                                                        classes=np.unique(train_labels),
                                                        y=train_labels)
                      class_weights = torch.FloatTensor(class_weights).to(device)
                      criterion = FocalSymmetricCrossEntropy(alpha=0.1, gamma=2.5, num_classes=6, class_weights=class_weights)
                  elif args.loss_fn == 'lsl':
                      criterion = LabelSmoothingLoss(classes=6)
                  elif args.loss_fn == 'cbf':
                      train_labels = []
                      for data in train_loader:
                          train_labels.extend(data.y.cpu().numpy())

                      classes = np.array(np.unique(train_labels))

                      class_weights = compute_class_weight(class_weight='balanced',
                                                          classes=classes,
                                                          y=train_labels)

                      class_weights = torch.FloatTensor(class_weights).to(device)
                      criterion = WeightedSymmetricCrossEntropy(alpha=0.2, num_classes=6, class_weights=class_weights)
                  else:
                    raise ValueError(f"Loss function {args.loss_fn} non supportata.")
                  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

                  early_stopper = EarlyStopper(patience=args.patience, mode='max')
                  best_f1 = 0.0
                  train_losses = []
                  val_losses = []
                  val_accuracies = []
                  val_f1s = []

                  for epoch in range(args.epochs):
                      loss, acc = train(train_loader, model_ft, optimizer, criterion, device, epoch, args.epochs)
                      val_acc, val_f1, val_loss = evaluate(val_loader, model_ft, device, criterion)

                      print(f"📈 Learning rate: {optimizer.param_groups[0]['lr']}")
                      print(f"📊 FineTune Epoch {epoch+1}/{args.epochs} | LossTrain: {loss:.4f}| LossVal: {val_loss:.4f} | ValAcc: {val_acc:.4f} | ValF1: {val_f1:.4f}")
                      scheduler.step(val_loss)

                      train_losses.append(loss)
                      val_losses.append(val_loss)
                      val_accuracies.append(val_acc)
                      val_f1s.append(val_f1)

                      if val_f1 > best_f1:
                          best_f1 = val_f1
                          torch.save(model_ft.state_dict(), f"checkpoints/model_{name}_epoch_{epoch}.pth")
                          print(f"💾 Salvato modello migliore per: {name} (F1={best_f1:.4f})")

                      if early_stopper(val_f1):
                          print(f"🛑 Early stopping attivato per: {name}")
                          break
                  save_logs(train_losses,val_losses, val_accuracies, val_f1s, args)

    # Test phase
    dataset_name = Path(args.test_path).parts[-2]
    print(f"\n🧪 Test sul dataset: {dataset_name}")

    test_dataset = full_dataset[dataset_name]["test"]
    print(f"✅ Dataset di test caricato: {len(test_dataset)} grafi")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


################################################################
    if dataset_name == 'C' or dataset_name == 'D':
         # Carica i modelli pre-addestrati
        model1_path = f"checkpoints/model_{dataset_name}_ensemble1.pth"
        model2_path = f"checkpoints/model_{dataset_name}_ensemble2.pth"
        model1 = load_model(model1_path, device)
        model2 = load_model(model2_path, device)

            # Define model architecture before loading state_dict
        model1 = GNN(
            num_class=6,  # Assuming 6 classes based on your code
            gnn_type='gin', # Assuming gin based on your code
            num_layer=4,  # Assuming 4 layers
            emb_dim=300,  # Assuming 300 embedding dimensions
            drop_ratio=0.5, # Assuming 0.0 drop ratio
            JK='sum',     # Assuming JK='sum'
            residual=True, # Assuming residual=True
            virtual_node=True, # Assuming virtual_node=True
            graph_pooling='attention' # Assuming graph_pooling='attention'
        ).to(device)
        model2 = GNN(
            num_class=6,  # Assuming 6 classes based on your code
            gnn_type='gin', # Assuming gin based on your code
            num_layer=4,  # Assuming 4 layers
            emb_dim=300,  # Assuming 300 embedding dimensions
            drop_ratio=0.5, # Assuming 0.0 drop ratio
            JK='sum',     # Assuming JK='sum'
            residual=True, # Assuming residual=True
            virtual_node=True, # Assuming virtual_node=True
            graph_pooling='attention' # Assuming graph_pooling='attention'
        ).to(device)

        model1.load_state_dict(torch.load(model1_path))
        model2.load_state_dict(torch.load(model2_path))

        # Set models to evaluation mode
        model1.eval()
        model2.eval()

        # Create the ensemble model
        ensemble_model = EnsembleModel(model1, model2).to(device)

        # Calcola le predizioni usando il modello dell'ensamble
        all_predictions = []
        with torch.no_grad():
            for data in tqdm(test_loader, desc="🔍 Inference with ensemble"):
                data = data.to(device)
                avg_probs = ensemble_model(data) # The EnsembleModel forward now returns avg_probs
                pred = avg_probs.argmax(dim=1).cpu().numpy()
                all_predictions.extend(pred)


        save_predictions(all_predictions, dataset_name)
        print("🏁 Test completato.")
        return  # esci se test


    else:
      best_checkpoint = f'checkpoints/model_{dataset_name}_best.pth'
      print(f"✅ Trovato miglior checkpoint: {best_checkpoint}")
      checkpoint_path = best_checkpoint
      
      model = GNN(
          num_class=6,
          gnn_type='gin',
          num_layer=4,
          emb_dim=args.emb_dim,
          drop_ratio=args.drop_ratio,
          JK='sum',
          residual=True,
          virtual_node=True,
          graph_pooling='attention'
      ).to(device)

      model.load_state_dict(torch.load(checkpoint_path))
      print(f"📦 Modello caricato da: {checkpoint_path}")

      predictions = predict(test_loader, model, device)
      save_predictions(predictions, dataset_name)
      print("🏁 Test completato.")
      return  # esci se test

if __name__ == "__main__":
    main()
