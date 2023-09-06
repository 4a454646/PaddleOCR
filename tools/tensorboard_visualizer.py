# # <font style="color:blue">TensorBoard Visualizer Class</font>

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pytz import timezone
import pytz

class TensorBoardVisualizer():
    def __init__(self, log_dir):
        self._writer = SummaryWriter(log_dir=log_dir)


    def update_charts(self, lr, train_acc, train_loss, valid_acc, valid_loss, epoch):
        self._writer.add_scalar(
            tag=f"Data/LR", 
            scalar_value=lr, 
            global_step=epoch
        )
        if train_loss == 0:
            self._writer.add_scalar(
                tag=f"Batch/Train_Accuracy", 
                scalar_value=train_acc, 
                global_step=epoch
            )
            self._writer.add_scalar(
                tag=f"Batch/Valid_Accuracy", 
                scalar_value=valid_acc, 
                global_step=epoch
            )
        else:
            self._writer.add_scalar(
                tag=f"Dataset/Train_Accuracy", 
                scalar_value=train_acc, 
                global_step=epoch
            )
            self._writer.add_scalar(
                tag=f"Dataset/Valid_Accuracy", 
                scalar_value=valid_acc, 
                global_step=epoch
            )
            self._writer.add_scalar(
                tag=f"Dataset/Train_Loss", 
                scalar_value=train_loss, 
                global_step=epoch
            )
            self._writer.add_scalar(
                tag=f"Dataset/Valid_Loss", 
                scalar_value=valid_loss, 
                global_step=epoch
            )

    def close_tensorboard(self):
        self._writer.close()
