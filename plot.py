import matplotlib.pyplot as plt
from train import training_loss, testing_loss, num_epochs


def plot_training(epochs=num_epochs, training_losses=training_loss, testing_losses=testing_loss):
    plt.plot(epochs, training_losses, label="Train loss")
    plt.plot(epochs, testing_losses, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend();
    plt.show()