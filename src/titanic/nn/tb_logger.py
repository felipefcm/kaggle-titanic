from typing import Mapping, Optional

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, run_name: str, log_dir="runs"):
        path = f"{log_dir}/{run_name}"
        self._writer = SummaryWriter(log_dir=path)

    def log_epoch(
        self,
        epoch: int,
        lr: Optional[list[float]] = None,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        metrics: Optional[Mapping[str, float]] = None,
    ) -> None:
        if loss is not None:
            self._writer.add_scalar("Loss/train", float(loss), epoch)
        if accuracy is not None:
            self._writer.add_scalar("Accuracy/eval", float(accuracy), epoch)
        if lr is not None:
            lr_value = float(lr[0]) if isinstance(lr, (list, tuple)) else float(lr)
            self._writer.add_scalar("LR", lr_value, epoch)
        if metrics:
            for name, value in metrics.items():
                self._writer.add_scalar(f"Metrics/{name}", float(value), epoch)

    def close(self) -> None:
        self._writer.close()
