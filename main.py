import build, train, plot
import time


print("\nHello, this is a script that uses the Iris flower dataset to train and classify the flower type of the last 20% of the dataset.")
print("\nThese flowers will be identified as Iris-Setosa, Iris-Versicolour, or Iris-Virginica.")
input("\nEnter any key to begin.\n")

print("\n")

model_1 = build.IrisModelv1(4, 10, 3)
loss_fn = build.nn.CrossEntropyLoss()
optimizer_AD = build.torch.optim.Adam(params=model_1.parameters(), lr=0.1)
accuracy = build.Accuracy(task="multiclass", num_classes=3)

print("loading...\n\n")
time.sleep(0.75)

train.train_test_model(model_1, loss_fn, optimizer_AD, accuracy, 1000)

plot.plot_training()