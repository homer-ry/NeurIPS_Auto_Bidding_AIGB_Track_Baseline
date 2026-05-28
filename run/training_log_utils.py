import math
import os

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class BatchMeanMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, batch_size):
        self.total += float(value) * int(batch_size)
        self.count += int(batch_size)

    @property
    def mean(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise ValueError(f"Expected a boolean value, got {value!r}")


def format_loss_for_log(value, precision=6):
    value = float(value)
    if not math.isfinite(value):
        return str(value)

    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value:.4e} (log1p={math.log1p(abs_value):.{precision}f})"
    if abs_value >= 10_000:
        return f"{value:.4e}"
    return f"{value:.{precision}f}"


def create_summary_writer(log_dir):
    if SummaryWriter is None or not log_dir:
        return None
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)


def add_scalar(writer, name, value, step):
    if writer is not None:
        writer.add_scalar(name, float(value), step)
