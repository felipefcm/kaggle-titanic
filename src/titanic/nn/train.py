from titanic.nn.nn import TitanicNN
from titanic.dataset import load_titanic_data
from titanic.nn.input import get_torch_dataset, prepare_input
from torch.utils.data import DataLoader
import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import binary_cross_entropy

num_epochs = 400
batch_size = 5

trainset_passengers, stats = load_titanic_data("./dataset/train.csv")

train_raw = get_torch_dataset(trainset_passengers[:700])
eval_raw = get_torch_dataset(trainset_passengers[700:])

trainset = prepare_input(train_raw, stats)
evalset = prepare_input(eval_raw, stats)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(evalset, batch_size=batch_size, shuffle=False)

input_size = trainset[0][0].shape.numel()
nn = TitanicNN(input_size)

optimiser = Adam(nn.parameters(), lr=0.1)
lr_scheduler = StepLR(optimiser, step_size=100, gamma=0.9)

epoch_losses: list[float] = []

for epoch in range(num_epochs):
    nn.train()

    for input, expected in train_loader:
        output = nn(input)
        loss = binary_cross_entropy(output, expected)

        epoch_losses.append(loss.item())

        loss.backward()
        optimiser.step()
        nn.zero_grad()

    lr_scheduler.step()

    nn.eval()
    with torch.no_grad():
        correct: int = 0
        for input, expected in eval_loader:
            output = nn(input)
            correct += ((output >= 0.5) == (expected == 1.0)).sum().item()

        accuracy = correct / len(evalset)
        print("Evaluation accuracy after epoch", epoch, ":", accuracy)

    print(
        "Epoch",
        epoch,
        "loss:",
        sum(epoch_losses) / len(epoch_losses),
        "lr:",
        lr_scheduler.get_last_lr(),
    )
    epoch_losses = []
