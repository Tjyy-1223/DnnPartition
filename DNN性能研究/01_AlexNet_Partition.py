import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),nn.BatchNorm2d(96),
    nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),nn.BatchNorm2d(256),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 1000),
    # nn.Softmax(dim = 1)
)


if __name__ == '__main__':
    # X = torch.randn(1, 1, 224, 224)
    # for layer in net:
    #     X=layer(X)
    #     print(layer.__class__.__name__,'output shape:\t',X.shape)

    batch_size = 1
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    loss = nn.CrossEntropyLoss()
    for x, y in train_iter:
        print(x.shape)
        print(y.shape)
        # print(X.shape)
        print(loss(net(x), y))
        # print(loss(net(X),y))
        break

