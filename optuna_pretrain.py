# optuna_pretrain.py
import optuna
import torch
import argparse
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from main import GNN, EnhancedNoisyCrossEntropyLoss, train, evaluate, EarlyStopper, load_or_create_datasets, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    set_seed()

    # Parametri iperottimizzati (inclusi quelli strutturali)
    drop_ratio = trial.suggest_float("drop_ratio", 0.0, 0.5, step=0.05)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    p_noisy = trial.suggest_float("p_noisy", 0.1, 0.4, step=0.05)
    temperature = trial.suggest_float("temperature", 0.7, 1.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3])

    # Strutturali
    gnn_type = trial.suggest_categorical("gnn_type", ["gin", "gcn"])
    num_layer = trial.suggest_int("num_layer", 3, 6)
    emb_dim = trial.suggest_categorical("emb_dim", [64, 128, 256, 300])
    graph_pooling = trial.suggest_categorical("graph_pooling", ["mean", "max", "attention"])
    virtual_node = trial.suggest_categorical("virtual_node", [True, False])

    # === Dataset unificato ===
    args = argparse.Namespace(
        use_saved=True,
        saved_dataset_dir="saved_datasets",
        data_root="data",
        batch_size=batch_size,
        epochs=5,
        patience=5
    )

    full_dataset = load_or_create_datasets(args)

    all_train_graphs = []
    for name in full_dataset:
        all_train_graphs.extend(full_dataset[name]["train"])
    pretrain_size = int(0.8 * len(all_train_graphs))
    val_size = len(all_train_graphs) - pretrain_size

    pretrain_data, preval_data = random_split(all_train_graphs, [pretrain_size, val_size])
    train_loader = DataLoader(pretrain_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(preval_data, batch_size=batch_size, shuffle=False)

    model = GNN(
        num_class=6,
        gnn_type=gnn_type,
        num_layer=num_layer,
        emb_dim=emb_dim,
        drop_ratio=drop_ratio,
        JK='sum',
        residual=True,
        virtual_node=virtual_node,
        graph_pooling=graph_pooling
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = EnhancedNoisyCrossEntropyLoss(p_noisy=p_noisy, temperature=temperature)

    best_f1 = 0.0
    early_stopper = EarlyStopper(patience=args.patience, mode="min")

    for epoch in range(args.epochs):
        loss, acc = train(train_loader, model, optimizer, criterion, device, epoch, args.epochs)
        val_acc, val_f1, val_loss = evaluate(val_loader, model, device, criterion)
        print(f"Epoch {epoch+1} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"checkpoints/pretrained_model_trial_{trial.number}.pth")
        if early_stopper(val_loss):
            break

    return best_f1

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("🏆 Migliori iperparametri strutturali trovati:")
    print(study.best_params)
