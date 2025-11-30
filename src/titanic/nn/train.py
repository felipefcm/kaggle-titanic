from titanic.nn.nn import TitanicNN
from titanic.dataset import load_titanic_data
from titanic.nn.input import get_torch_dataset
from torch.utils.data import DataLoader
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import binary_cross_entropy

num_epochs = 800
batch_size = 10

trainset_passengers = load_titanic_data("./dataset/train.csv")
trainset = get_torch_dataset(trainset_passengers[:700])
evalset = get_torch_dataset(trainset_passengers[700:])

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
evalloader = DataLoader(evalset, batch_size=batch_size, shuffle=False)

input_size = trainset[0][0].shape.numel()
nn = TitanicNN(input_size)

optimiser = SGD(nn.parameters(), lr=0.1)
lr_scheduler = StepLR(optimiser, step_size=100, gamma=0.9)

epoch_losses: list[float] = []

for epoch in range(num_epochs):
    nn.train()

    for input, expected in trainloader:
        output = nn(input)
        zeroes = Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)

        # print(output, zeroes)
        loss = binary_cross_entropy(output, expected)
        # loss = binary_cross_entropy(output, zeroes)
        epoch_losses.append(loss.item())

        loss.backward()
        optimiser.step()
        nn.zero_grad()

    lr_scheduler.step()
    print(
        "Epoch",
        epoch,
        "loss:",
        sum(epoch_losses) / len(epoch_losses),
        "lr:",
        lr_scheduler.get_lr(),
    )
    epoch_losses = []
