from mnist import MNIST

import minitorch

mndata = MNIST("project/data/")
images, labels = mndata.load_training()

BACKEND = minitorch.TensorBackend(minitorch.FastOps)
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input):
        conv = minitorch.conv2d(input, self.weights.value)
        return conv + self.bias.value


class Network(minitorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self):
        super().__init__()

        # For vis
        self.mid = None
        self.out = None
        self.conv1 = Conv2d(1, 4, 3, 3)
        self.conv2 = Conv2d(4, 8, 3, 3)
        self.layer1 = Linear(392, 64)
        self.layer2 = Linear(64, 10)

    def forward(self, x):
        self.mid = self.conv1.forward(x).relu()
        self.out = self.conv2.forward(self.mid).relu()
        pool = minitorch.avgpool2d(self.out, (4, 4))
        pool = pool.view(BATCH, 392)
        h = self.layer1.forward(pool).relu()
        if self.training:
            h = minitorch.dropout(h, 0.25)
        return minitorch.logsoftmax(self.layer2.forward(h), dim=1)


def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


best_val = 0.0


def default_log_fn(epoch, total_loss, correct, total, losses, model):
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")


def file_log_fn_factory(
    log_path="mnist.txt",
    train_size=None,
    val_size=None,
    learning_rate=None,
    batch_size=None,
    max_epochs=None,
):
    log_file = open(log_path, "w")

    # Write training configuration
    log_file.write("# MNIST Classification Training Log\n")
    log_file.write(f"Train size: {train_size}\n")
    log_file.write(f"Validation size: {val_size}\n")
    log_file.write(f"Learning rate: {learning_rate}\n")
    log_file.write(f"Batch size: {batch_size}\n")
    log_file.write(f"Max epochs: {max_epochs}\n")
    log_file.write("-" * 40 + "\n")

    def log_fn(epoch, total_loss, correct, total, losses, model):
        global best_val
        best_val = max(best_val, correct / total)
        msg = (
            f"Epoch {epoch}, Loss: {total_loss:.4f}, "
            f"Validation Accuracy: {correct/total:.2%}\n"
            f"Best Validation Accuracy: {best_val:.2%}\n"
        )

        print(msg.strip())
        log_file.write(msg)
        log_file.flush()

    return log_fn


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=BACKEND))

    def train(
        self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):
                if n_training_samples - example_num <= BATCH:
                    continue
                y = minitorch.tensor(
                    y_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == BACKEND
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                # Update
                optim.step()

                if batch_num % 5 == 0:
                    model.eval()
                    # Evaluate on 5 held-out batches

                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = minitorch.tensor(
                            y_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        x = minitorch.tensor(
                            X_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(epoch, total_loss, correct, BATCH, losses, model)

                    total_loss = 0.0
                    model.train()


if __name__ == "__main__":
    train_size = 5000
    val_size = 500
    learning_rate = 0.01
    max_epochs = 2
    batch_size = BATCH  # 16

    data_train, data_val = (
        make_mnist(0, train_size),
        make_mnist(10000, 10000 + val_size),
    )

    log_fn = file_log_fn_factory(
        log_path="mnist.txt",
        train_size=train_size,
        val_size=val_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
    )

    ImageTrain().train(
        data_train,
        data_val,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        log_fn=log_fn,
    )
