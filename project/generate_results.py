import time
import minitorch
from run_torch import TorchTrain
from run_fast_tensor import FastTrain

HIDDEN = 10
LEARNING_RATE = 0.5
EPOCHS = 500
DATASET_SIZE = 250  # Number of points per dataset

# Markdown header
markdown_lines = [
    "# Training Results for Neural Network\n",
    f"- Hidden size: `{HIDDEN}`",
    f"- Learning rate: `{LEARNING_RATE}`",
    f"- Epochs: `{EPOCHS}`",
    f"- Dataset size: `{DATASET_SIZE}`",
    "",
]


def run_training(model_class, label):
    """Runs training for a given model class and returns markdown output."""
    result_lines = [f"# {label} Results\n"]

    for name, dataset_fn in minitorch.datasets.items():
        print(f"\n== [{label}] Training on dataset: {name} ==")
        dataset = dataset_fn(DATASET_SIZE)
        model = model_class(HIDDEN)

        epoch_losses = []
        epoch_accuracies = []

        def log_fn(epoch, total_loss, correct, _):
            acc = 100.0 * correct / DATASET_SIZE
            epoch_losses.append(total_loss)
            epoch_accuracies.append(acc)
            if epoch % 100 == 0 or epoch == EPOCHS:
                print(
                    f"[{label}] Epoch {epoch}: loss={total_loss:.4f}, accuracy={acc:.2f}%"
                )

        start_time = time.time()
        model.train(dataset, LEARNING_RATE, EPOCHS, log_fn)
        total_time = time.time() - start_time
        time_per_epoch = total_time / EPOCHS

        # Markdown section for this dataset
        result_lines += [
            f"## Dataset: `{name}`",
            f"- Time per epoch: **{time_per_epoch:.4f}** seconds",
            f"- Final loss: **{epoch_losses[-1]:.4f}**",
            f"- Final accuracy: **{epoch_accuracies[-1]:.2f}%**",
            "",
            "### Loss and Accuracy per Epoch",
            "",
            "| Epoch | Loss | Accuracy (%) |",
            "|-------|------|---------------|",
        ]
        for epoch in range(len(epoch_losses)):
            result_lines.append(
                f"| {epoch+1} | {epoch_losses[epoch]:.4f} | {epoch_accuracies[epoch]:.2f} |"
            )

        result_lines.append("")  # Blank line between datasets

    return result_lines


# Run for both TorchTrain and FastTrain
markdown_lines += run_training(TorchTrain, "TorchTrain (Slow)")
markdown_lines += run_training(FastTrain, "FastTrain (Fast Tensor Backend)")

# Save to README
with open("README.md", "a") as f:
    f.write("\n".join(markdown_lines))
