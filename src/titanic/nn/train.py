from datetime import datetime

from titanic.nn.nn import TitanicNN
from titanic.dataset import load_titanic_data, process_passengers
from titanic.nn.input import get_torch_dataset
from titanic.nn.tb_logger import TBLogger

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import binary_cross_entropy


num_epochs = 300
batch_size = 10

trainset_passengers = load_titanic_data("./dataset/train.csv")

train_processed, stats = process_passengers(trainset_passengers[:700])
eval_processed, _ = process_passengers(trainset_passengers[700:])
print(f"Datasets loaded: train={len(train_processed)} eval={len(eval_processed)}")

trainset = get_torch_dataset(train_processed)
evalset = get_torch_dataset(eval_processed)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(evalset, batch_size=batch_size, shuffle=False)

input_size = trainset[0][0].shape.numel()
nn = TitanicNN(input_size)

optimiser = Adam(nn.parameters(), lr=0.1)
lr_scheduler = StepLR(optimiser, step_size=100, gamma=0.9)

logger = TBLogger(
    run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}", log_dir="runs"
)

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

    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    epoch_losses = []

    logger.log_epoch(
        epoch, lr=lr_scheduler.get_last_lr(), loss=avg_loss, accuracy=accuracy
    )

    print(
        "Epoch",
        epoch,
        "loss:",
        avg_loss,
        "lr:",
        lr_scheduler.get_last_lr(),
        "evaluation accuracy:",
        accuracy,
    )

logger.close()
