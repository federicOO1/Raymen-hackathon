# optuna_tune.py
import optuna
import argparse
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from main import GNN, EnhancedNoisyCrossEntropyLoss, train, evaluate, set_seed, load_or_create_datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    set_seed()

    # 🔍 Spazio degli iperparametri
    drop_ratio = trial.suggest_float("drop_ratio", 0.0, 0.5, step=0.05)          # valori: 0.0, 0.05, ..., 0.5
    lr = round(trial.suggest_float("lr", 1e-4, 5e-3, log=True), 5)              # valori log: 0.0001, 0.0002, ..., 0.005
    p_noisy = trial.suggest_float("p_noisy", 0.1, 0.4, step=0.05)                # valori: 0.10, 0.15, ..., 0.4
    temperature = trial.suggest_float("temperature", 0.7, 1.5, step=0.1)         # valori: 0.7, 0.8, ..., 1.5
    batch_size_iper = trial.suggest_categorical("batch_size", [32, 64, 128])     # rimane invariato
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])  # Scelta discreta


    # === Dataset e split ===
    args = argparse.Namespace(
        use_saved=True,
        saved_dataset_dir="saved_datasets",
        data_root="data",
        data_name="B",      # <--- cambia nome dataset se vuoi ottimizzare altri
        batch_size=32,
        epochs=10,
        patience=5
    )

    full_dataset = load_or_create_datasets(args)
    dataset = full_dataset[args.data_name]["train"]
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size_iper, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size_iper, shuffle=False)

    # === Modello, ottimizzatore, loss ===
    model = GNN(
        num_class=6,
        gnn_type='gin',
        num_layer=5,
        emb_dim=300,
        drop_ratio=drop_ratio,
        JK='sum',
        residual=True,
        virtual_node=True,
        graph_pooling='mean'
    ).to(device)

    model.load_state_dict(torch.load("checkpoints/pretrained_model.pth"))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    criterion = EnhancedNoisyCrossEntropyLoss(p_noisy=p_noisy, temperature=temperature)
    best_f1 = 0.0
    early_stop_counter = 0

    for epoch in range(args.epochs):
        train_loss, _ = train(train_loader, model, optimizer, criterion, device, epoch, args.epochs)
        val_acc, val_f1, val_loss = evaluate(val_loader, model, device, criterion)
        scheduler.step(val_loss)

        if val_f1 > best_f1:
            best_f1 = val_f1
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= args.patience:
            break

    return best_f1  # 🎯 Obiettivo da massimizzare

# === Lancia lo studio ===
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("🏆 Migliori iperparametri trovati:")
    print(study.best_params)
