import random, torch, numpy as np
import torch
from tqdm import tqdm

def pretrain(model, loader, optimizer, criterion, device):
    model.train()
    model.to(device)
    total_class_loss, total_vgae_loss = 0, 0

    for batch in tqdm(loader, desc="Pretraining"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out, vgae_loss = model(batch)
        class_loss = criterion(out, batch.y)
        loss = class_loss + vgae_loss
        loss.backward()
        optimizer.step()
        total_class_loss += class_loss.item()
        total_vgae_loss += vgae_loss.item()

    return total_class_loss / len(loader), total_vgae_loss / len(loader)

def finetune(model, loader, optimizer, criterion, device):
    model.train()
    model.to(device)
    total_loss = 0
    for batch in tqdm(loader, desc="Fine-tuning"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out, _ = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

import torch
from sklearn.metrics import f1_score

def evaluate(models, loader, device):
    preds = []
    truths = []
    for model in models:
        model.eval()
        model.to(device)
        model_preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                pred = torch.argmax(out, dim=1).cpu()
                model_preds.append(pred)
        preds.append(torch.cat(model_preds))

    # Weighted voting
    preds = torch.stack(preds)  # (num_models, N)
    final_preds = torch.mode(preds, dim=0).values  # Majority vote

    for batch in loader:
        truths.append(batch.y.cpu())
    true_labels = torch.cat(truths)

    f1 = f1_score(true_labels, final_preds, average='macro')
    return f1

