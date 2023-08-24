# # <font style="color:blue">TensorBoard Visualizer Class</font>

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pytz import timezone
import pytz

class TensorBoardVisualizer():
    def __init__(self):
        date = datetime.now(tz=pytz.utc)
        date = date.astimezone(timezone('US/Pacific'))
        pst_time = date.strftime("%B_%d_%Y_%I:%M%p")
        log_dir = f"/workspace/PaddleOCR/runs/{pst_time}"
        self._writer = SummaryWriter(log_dir=log_dir)


    def update_charts(self, lr, train_acc, train_loss, valid_acc, valid_loss, inf_loss, epoch):
        self._writer.add_scalar(
            tag=f"Data/LR", 
            scalar_value=lr, 
            global_step=epoch
        )
        self._writer.add_scalar(
            tag=f"Data/Count_Infinite_Loss", 
            scalar_value=inf_loss, 
            global_step=epoch
        )
        self._writer.add_scalar(
            tag=f"Train/Accuracy", 
            scalar_value=train_acc, 
            global_step=epoch
        )
        self._writer.add_scalar(
            tag=f"Train/Loss", 
            scalar_value=train_loss, 
            global_step=epoch
        )
        self._writer.add_scalar(
            tag=f"Valid/Accuracy", 
            scalar_value=valid_acc, 
            global_step=epoch
        )
        self._writer.add_scalar(
            tag=f"Valid/Loss", 
            scalar_value=valid_loss, 
            global_step=epoch
        )

    def close_tensorboard(self):
        self._writer.close()
