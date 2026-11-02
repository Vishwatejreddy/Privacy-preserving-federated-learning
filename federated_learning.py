"""Educational noise-augmented FedAvg simulation on the UCI Adult dataset."""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]


def read_adult(path: str, skiprows: int = 0) -> pd.DataFrame:
    return pd.read_csv(
        path, header=None, names=COLUMNS, skiprows=skiprows,
        skipinitialspace=True, na_values="?"
    )


def prepare_data(train_path: str, test_path: str):
    train = read_adult(train_path)
    test = read_adult(test_path, skiprows=1)

    for frame in (train, test):
        frame["income"] = (
            frame["income"].astype(str).str.replace(".", "", regex=False).str.strip()
        )
        frame.dropna(inplace=True)

    y_train = train.pop("income").map({"<=50K": 0, ">50K": 1})
    y_test = test.pop("income").map({"<=50K": 0, ">50K": 1})

    train = train.loc[y_train.notna()]
    test = test.loc[y_test.notna()]
    y_train = y_train.dropna().astype(np.float32)
    y_test = y_test.dropna().astype(np.float32)

    x_train_df = pd.get_dummies(train)
    x_test_df = pd.get_dummies(test).reindex(columns=x_train_df.columns, fill_value=0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_df).astype(np.float32)
    x_test = scaler.transform(x_test_df).astype(np.float32)

    return (
        torch.tensor(x_train), torch.tensor(y_train.to_numpy()).view(-1, 1),
        torch.tensor(x_test), torch.tensor(y_test.to_numpy()).view(-1, 1),
    )


class IncomeNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def make_clients(x, y, count: int):
    splits = np.array_split(np.random.permutation(len(x)), count)
    return [(x[index], y[index]) for index in splits if len(index)]


def train_client(global_model, data, epochs: int, batch_size: int):
    model = copy.deepcopy(global_model)
    loader = DataLoader(TensorDataset(*data), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    model.train()

    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()

    return model.state_dict(), len(data[0])


def add_noise(state, std: float):
    if std <= 0:
        return state
    result = {}
    for key, value in state.items():
        result[key] = value + torch.randn_like(value) * std if torch.is_floating_point(value) else value
    return result


def fed_avg(states, sizes):
    total = sum(sizes)
    averaged = copy.deepcopy(states[0])
    for key in averaged:
        if torch.is_floating_point(averaged[key]):
            averaged[key] = sum(state[key] * (size / total) for state, size in zip(states, sizes))
    return averaged


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    predicted = (model(x) >= 0.5).float()
    return predicted.eq(y).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    if args.clients < 1 or args.rounds < 1:
        parser.error("--clients and --rounds must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x_train, y_train, x_test, y_test = prepare_data(args.train_data, args.test_data)
    clients = make_clients(x_train, y_train, args.clients)
    global_model = IncomeNet(x_train.shape[1])
    history = []

    for round_index in range(args.rounds):
        states, sizes = [], []
        for client in clients:
            state, size = train_client(
                global_model, client, args.local_epochs, args.batch_size
            )
            states.append(add_noise(state, args.noise_std))
            sizes.append(size)

        global_model.load_state_dict(fed_avg(states, sizes))
        score = accuracy(global_model, x_test, y_test)
        history.append(score)
        print(f"Round {round_index + 1}/{args.rounds}: test accuracy = {score:.4f}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(global_model.state_dict(), output / "final_model.pt")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Communication round")
    plt.ylabel("Held-out test accuracy")
    plt.title("Federated Learning Performance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "accuracy_by_round.png", dpi=150)
    plt.close()
    print(f"Saved outputs to: {output.resolve()}")
    print("Note: noise injection here does not provide a formal differential-privacy guarantee.")


if __name__ == "__main__":
    main()
