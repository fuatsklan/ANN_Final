import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

from dataset import MVTecTrainDataset, batch_loader
from model import ConvAutoencoder
from nn_scratch import MSELoss, Adam


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_model(model, path):
    params = [p.copy() for p, g in model.params()]

    with open(path, "wb") as f:
        pickle.dump(params, f)


def load_model(model, path):
    with open(path, "rb") as f:
        saved_params = pickle.load(f)

    for saved, (p, g) in zip(saved_params, model.params()):
        p[...] = saved


def plot_training_history(history, out_path):
    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, marker="o", linewidth=2, color="#2c7fb8")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training reconstruction MSE")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def add_gaussian_noise(x, std):
    if std <= 0:
        return x
    noisy = x + np.random.normal(loc=0.0, scale=std, size=x.shape)
    return np.clip(noisy, 0.0, 1.0)


def checkpoint_filename(category, training_mode):
    if training_mode == "vanilla":
        return f"{category}_scratch_cae.pkl"
    return f"{category}_{training_mode}_scratch_cae.pkl"


def train(args):
    np.random.seed(args.seed)

    dataset = MVTecTrainDataset(
        root_dir=args.data_root,
        category=args.category,
        img_size=args.img_size
    )

    model = ConvAutoencoder()
    loss_fn = MSELoss()
    optimizer = Adam(model.params(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir) / args.category
    figures_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n = 0

        pbar = tqdm(
            batch_loader(dataset, args.batch_size, shuffle=True),
            total=len(dataset) // args.batch_size + 1,
            desc=f"Epoch {epoch}/{args.epochs}"
        )

        for x, target in pbar:
            if args.training_mode == "denoising":
                model_input = add_gaussian_noise(x, args.noise_std)
            else:
                model_input = x

            pred = model.forward(model_input)
            loss = loss_fn.forward(pred, target)

            grad = loss_fn.backward()
            model.backward(grad)

            optimizer.step()

            epoch_loss += loss * x.shape[0]
            n += x.shape[0]

            pbar.set_postfix({"loss": loss})

        epoch_loss /= n
        history.append({"epoch": epoch, "train_loss": float(epoch_loss)})

        print(f"Epoch {epoch}: train loss = {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = output_dir / checkpoint_filename(args.category, args.training_mode)
            save_model(model, save_path)
            print(f"Saved model to {save_path}")

    csv_path = history_dir / f"{args.category}_training_history.csv"
    json_path = history_dir / f"{args.category}_training_history.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(history)

    with open(json_path, "w") as f:
        json.dump(
            {
                "category": args.category,
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "training_mode": args.training_mode,
                "noise_std": args.noise_std,
                "seed": args.seed,
                "best_loss": float(best_loss),
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"Training history saved to {csv_path} and {json_path}")

    loss_plot_path = figures_dir / "00_training_loss.png"
    plot_training_history(history, loss_plot_path)
    print(f"Training loss plot saved to {loss_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--category", type=str, default="carpet")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "checkpoints"),
    )
    parser.add_argument(
        "--history_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "metrics"),
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "figures"),
    )

    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--training_mode",
        choices=["vanilla", "denoising"],
        default="vanilla",
        help="vanilla reconstructs clean input; denoising reconstructs clean target from noisy input",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.10,
        help="Gaussian noise standard deviation for denoising training",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
