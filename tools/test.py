import yaml

# Define the input and output file paths
input_file = "/workspace/PaddleOCR/configs/rec/v4.yml"
output_file = "/workspace/PaddleOCR/configs/rec/v4_overwritten.yml"

# Load the YAML file
with open(input_file, "r") as f:
    config = yaml.safe_load(f)

# Add the new option to the RecAug item in the transforms list
for transform in config["Train"]["dataset"]["transforms"]:
    if "RecAug" in transform:
        transform["RecAug"]["hello_world"] = 9999

# Write the updated config to a new file
with open(output_file, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

import train
import program

from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp
HP_HELLO_WORLD = hp.HParam('hello_world', hp.IntInterval(0, 999))

writer = SummaryWriter(log_dir="./trials/hparams")
hp.hparams_config(
    hparams=[HP_HELLO_WORLD],
    metrics=[hp.Metric('accuracy', display_name='Accuracy')],
)

# for i in range(5):
#     hello_world = HP_HELLO_WORLD.domain.sample_uniform()
#     print(f"Hello World: {hello_world}")
#     print(type(hello_world))
#     writer.add_hparams(
#         {'hello_world': hello_world},
#         {
#             'accuracy': 10 * float(hello_world),
#         },
#     )

if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    valid_acc = train.main(config, device, logger, vdl_writer)