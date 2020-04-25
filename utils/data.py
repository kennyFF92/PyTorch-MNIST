import torch
import torchvision
import torchvision.transforms as T


def get_MNIST_stats(data_path="./data"):
    mnist_train = torchvision.datasets.MNIST(data_path,
                                             transform=T.ToTensor(),
                                             train=True,
                                             download=True)
    mnist_mean = mnist_train.data.float().mean()
    mnist_std = mnist_train.data.float().std()
    return mnist_mean.item(), mnist_std.item()


def load_MNIST(batch_size,
               input_size=[28, 28],
               normalize_data=True,
               val_perc=0.0,
               data_path="./data",
               torch_device=torch.device('cpu')):
    if torch_device.type == "cpu":
        pin_memory = False
        num_workers = 0
    else:
        pin_memory = True
        num_workers = 1

    # Data normalization
    if normalize_data:
        mnist_mean, mnist_std = get_MNIST_stats()
        transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize((mnist_mean, ), (mnist_std, ))
        ])
    else:
        transform = T.Compose([T.Resize(input_size), T.ToTensor()])

    # Datasets
    mnist_train = torchvision.datasets.MNIST(data_path,
                                             transform=transform,
                                             train=True,
                                             download=True)
    if val_perc > 0.0:
        train_len = int(len(mnist_train) * (1 - val_perc))
        val_len = len(mnist_train) - train_len
        train_set, val_set = torch.utils.data.random_split(
            mnist_train, [train_len, val_len])
    else:
        train_set = mnist_train

    test_set = torchvision.datasets.MNIST(data_path,
                                          transform=transform,
                                          train=False,
                                          download=True)

    # Loaders
    def get_loader(dataset, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_set_loader = get_loader(train_set, shuffle=True)
    test_set_loader = get_loader(test_set)

    if val_perc > 0.0:
        val_set_loader = get_loader(val_set)
        return train_set_loader, val_set_loader, test_set_loader
    else:
        return train_set_loader, test_set_loader
