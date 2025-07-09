import time
import minitorch
from run_torch import TorchTrain

HIDDEN = 10
LEARNING_RATE = 0.5
EPOCHS = 500
DATASET_SIZE = 250  # Number of points per dataset

# Markdown header
markdown_lines = [
    "# Training Results for Neural Network with Tensor Backend\n",
    f"- Hidden size: `{HIDDEN}`",
    f"- Learning rate: `{LEARNING_RATE}`",
    f"- Epochs: `{EPOCHS}`",
    f"- Dataset size: `{DATASET_SIZE}`",
    "",
]

for name, dataset_fn in minitorch.datasets.items():
    print(f"\n== Training on dataset: {name} ==")
    dataset = dataset_fn(DATASET_SIZE)
    model = TorchTrain(HIDDEN)

    epoch_losses = []
    epoch_accuracies = []

    def log_fn(epoch, total_loss, correct, _):
        acc = 100.0 * correct / DATASET_SIZE
        epoch_losses.append(total_loss)
        epoch_accuracies.append(acc)
        if epoch % 100 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch}: loss={total_loss:.4f}, accuracy={acc:.2f}%")

    start_time = time.time()
    model.train(dataset, LEARNING_RATE, EPOCHS, log_fn)
    total_time = time.time() - start_time
    time_per_epoch = total_time / EPOCHS

    # Markdown section for this dataset
    markdown_lines += [
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
        markdown_lines.append(
            f"| {epoch+1} | {epoch_losses[epoch]:.4f} | {epoch_accuracies[epoch]:.2f} |"
        )

    markdown_lines.append("")  # Blank line between datasets

# Save results
with open("README.md", "a") as f:
    f.write("\n".join(markdown_lines))
