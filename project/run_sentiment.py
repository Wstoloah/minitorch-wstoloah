import random

import embeddings

import minitorch
from datasets import load_dataset

BACKEND = minitorch.TensorBackend(minitorch.FastOps)


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


class Conv1d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width)
        self.bias = RParam(1, out_channels, 1)

    def forward(self, input):
        conv = minitorch.conv1d(input, self.weights.value)
        return conv + self.bias.value


class CNNSentimentKim(minitorch.Module):
    """
    Implement a CNN for Sentiment classification based on Y. Kim 2014.

    This model should implement the following procedure:

    1. Apply a 1d convolution with input_channels=embedding_dim
        feature_map_size=100 output channels and [3, 4, 5]-sized kernels
        followed by a non-linear activation function (the paper uses tanh, we apply a ReLu)
    2. Apply max-over-time across each feature map
    3. Apply a Linear to size C (number of classes) followed by a ReLU and Dropout with rate 25%
    4. Apply a sigmoid over the class dimension.
    """

    def __init__(
        self,
        feature_map_size=100,
        embedding_size=50,
        filter_sizes=[3, 4, 5],
        dropout=0.25,
    ):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.embeddomg_size = embedding_size
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.num_filters = len(filter_sizes)
        self.flatten_len = self.num_filters * feature_map_size

        self.conv_layers = [
            Conv1d(embedding_size, feature_map_size, filter_size)
            for filter_size in filter_sizes
        ]

        self.concat_tensors = [
            minitorch.tensor(
                [1 if i == filter_idx else 0 for i in range(self.num_filters)],
                backend=BACKEND,
            ).view(1, 1, self.num_filters)
            for filter_idx in range(self.num_filters)
        ]

        self.linear_layer = Linear(self.num_filters * feature_map_size, 1)

    def forward(self, embeddings):
        """
        embeddings tensor: [batch x sentence length x embedding dim]
        """
        batch, s_length, emb_dim = embeddings.shape
        emb_permute: minitorch.Tensor = embeddings.permute(0, 2, 1)
        mid_layers = [conv1d_layer(emb_permute) for conv1d_layer in self.conv_layers]
        mid_max_layers = [minitorch.max(mid_layer, 2) for mid_layer in mid_layers]
        concat_layer = [c * m for c, m in zip(self.concat_tensors, mid_max_layers)]
        sum_layer = concat_layer[0] + concat_layer[1] + concat_layer[2]

        end_layer = self.linear_layer(sum_layer.view(batch, self.flatten_len)).relu()
        out_layer = minitorch.dropout(end_layer, self.dropout, not self.training)

        return out_layer.sigmoid()


# Evaluation helper methods
def get_predictions_array(y_true, model_output):
    predictions_array = []
    for j, logit in enumerate(model_output.to_numpy()):
        true_label = y_true[j]
        if logit > 0.5:
            predicted_label = 1.0
        else:
            predicted_label = 0
        predictions_array.append((true_label, predicted_label, logit))
    return predictions_array


def get_accuracy(predictions_array):
    correct = 0
    for y_true, y_pred, logit in predictions_array:
        if y_true == y_pred:
            correct += 1
    return correct / len(predictions_array)


best_val = 0.0


def default_log_fn(
    epoch,
    train_loss,
    losses,
    train_predictions,
    train_accuracy,
    validation_predictions,
    validation_accuracy,
):
    global best_val
    best_val = (
        best_val if best_val > validation_accuracy[-1] else validation_accuracy[-1]
    )
    print(f"Epoch {epoch}, loss {train_loss}, train accuracy: {train_accuracy[-1]:.2%}")
    if len(validation_predictions) > 0:
        print(f"Validation accuracy: {validation_accuracy[-1]:.2%}")
        print(f"Best Valid accuracy: {best_val:.2%}")


class SentenceSentimentTrain:
    def __init__(self, model):
        self.model = model

    def train(
        self,
        data_train,
        learning_rate,
        batch_size=10,
        max_epochs=500,
        data_val=None,
        log_fn=default_log_fn,
    ):
        model = self.model
        (X_train, y_train) = data_train
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        train_accuracy = []
        validation_accuracy = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, batch_size)
            ):
                y = minitorch.tensor(
                    y_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x)
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -(prob.log() / y.shape[0]).sum()
                loss.view(1).backward()

                # Save train predictions
                train_predictions += get_predictions_array(y, out)
                total_loss += loss[0]

                # Update
                optim.step()

            # Evaluate on validation set at the end of the epoch
            validation_predictions = []
            if data_val is not None:
                (X_val, y_val) = data_val
                model.eval()
                y = minitorch.tensor(
                    y_val,
                    backend=BACKEND,
                )
                x = minitorch.tensor(
                    X_val,
                    backend=BACKEND,
                )
                out = model.forward(x)
                validation_predictions += get_predictions_array(y, out)
                validation_accuracy.append(get_accuracy(validation_predictions))
                model.train()

            train_accuracy.append(get_accuracy(train_predictions))
            losses.append(total_loss)
            log_fn(
                epoch,
                total_loss,
                losses,
                train_predictions,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )
            total_loss = 0.0


def encode_sentences(
    dataset, N, max_sentence_len, embeddings_lookup, unk_embedding, unks
):
    Xs = []
    ys = []
    for sentence in dataset["sentence"][:N]:
        # pad with 0s to max sentence length in order to enable batching
        # TODO: move padding to training code
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = [0] * embeddings_lookup.d_emb
            if w in embeddings_lookup:
                sentence_embedding[i][:] = embeddings_lookup.emb(w)
            else:
                # use random embedding for unks
                unks.add(w)
                sentence_embedding[i][:] = unk_embedding
        Xs.append(sentence_embedding)

    # load labels
    ys = dataset["label"][:N]
    return Xs, ys


def encode_sentiment_data(dataset, pretrained_embeddings, N_train, N_val=0):
    #  Determine max sentence length for padding
    max_sentence_len = 0
    for sentence in dataset["train"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(sentence.split()))

    for sentence in dataset["validation"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(sentence.split()))

    unks = set()
    unk_embedding = [
        0.1 * (random.random() - 0.5) for i in range(pretrained_embeddings.d_emb)
    ]
    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"missing pre-trained embedding for {len(unks)} unknown words")

    return (X_train, y_train), (X_val, y_val)


if __name__ == "__main__":
    train_size = 450
    validation_size = 100
    learning_rate = 1e-6
    batch_size = 16
    max_epochs = 50

    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        load_dataset("glue", "sst2"),
        embeddings.GloveEmbedding("wikipedia_gigaword", d_emb=50, show_progress=True),
        train_size,
        validation_size,
    )
    model_trainer = SentenceSentimentTrain(
        CNNSentimentKim(feature_map_size=100, filter_sizes=[3, 4, 5], dropout=0.25)
    )
    model_trainer.train(
        (X_train, y_train),
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        data_val=(X_val, y_val),
    )
