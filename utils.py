import matplotlib.pyplot as plt


def plot_loss_curves(history):
    """
    Plot training & validation loss and metrics from a Keras History object.
    """

    loss = history.history.get("loss")
    val_loss = history.history.get("val_loss")

    # Handle accuracy naming differences
    acc = history.history.get("accuracy") or history.history.get("acc")
    val_acc = history.history.get("val_accuracy") or history.history.get("val_acc")

    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # ---- Loss plot ----
    ax[0].plot(epochs, loss, label="Training Loss")
    if val_loss is not None:
        ax[0].plot(epochs, val_loss, label="Validation Loss")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)

    # ---- Accuracy plot ----
    if acc is not None:
        ax[1].plot(epochs, acc, label="Training Accuracy")
    if val_acc is not None:
        ax[1].plot(epochs, val_acc, label="Validation Accuracy")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()