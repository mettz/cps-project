import signal

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import ToTensor

from cnn import CNN

signal.signal(signal.SIGINT, signal.SIG_DFL)

train_data = datasets.MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True,
)
test_data = datasets.MNIST(root="data", train=False, transform=ToTensor())

print(train_data)
print(test_data)
print(train_data.data.size())
print(train_data.targets.size())

plt.imshow(train_data.data[0], cmap="gray")
plt.title("%i" % train_data.targets[0])
plt.show()

figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

loaders = {
    "train": torch.utils.data.DataLoader(
        train_data, batch_size=100, shuffle=True, num_workers=1
    ),
    "test": torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=True, num_workers=1
    ),
}

model = CNN()
print(model)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

from torch.autograd import Variable

num_epochs = 10


def train(num_epochs, cnn, loaders):
    cnn.train()

    # Train the model
    total_step = len(loaders["train"])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders["train"]):
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y
            output = model(b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )
                pass

        pass

    pass


train(num_epochs, model, loaders)


def test():
    # Test the model
    model.eval()
    with torch.no_grad():
        for images, labels in loaders["test"]:
            test_output, last_layer = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
    print("Test Accuracy of the model on the 10000 test images: %.2f" % accuracy)
    pass


test()

sample = next(iter(loaders["test"]))
imgs, lbls = sample
actual_number = lbls[:10].numpy()
test_output, last_layer = model(imgs[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(f"Prediction number: {pred_y}")
print(f"Actual number: {actual_number}")
