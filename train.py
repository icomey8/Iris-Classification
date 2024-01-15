import build
import torch
from build import X_train, X_test, y_train, y_test



training_loss = []
testing_loss = []
num_epochs = []
acc = []

def train_test_model(model_1, loss_fn, optimizer_AD, accuracy, epochs):
    for epoch in range(epochs):
        model_1.train()
        train_preds = model_1.forward(X_train)

        train_loss = loss_fn(train_preds, y_train)
        train_acc = accuracy(train_preds, y_train)
        optimizer_AD.zero_grad()
        train_loss.backward()
        optimizer_AD.step()

        model_1.eval()
        with torch.inference_mode():
            test_preds = model_1(X_test)
            test_loss = loss_fn(test_preds, y_test)
            test_acc = accuracy(test_preds, y_test)


        if epoch % 100 == 0:
            training_loss.append(train_loss.detach().numpy())
            testing_loss.append(test_loss.detach().numpy())
            num_epochs.append(epoch)
            acc.append(test_acc.detach().numpy())
            print(f"Epoch: {epoch} | Loss: {train_loss:.3f} | Acc: {train_acc:.3f} | Test Loss: {test_loss:.3f}| Test Acc: {test_acc:.3f}")

