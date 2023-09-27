import yaml
import datetime
import time

from torch.utils.tensorboard.writer import SummaryWriter
from tensorboard.plugins.hparams import api as hp

import train
import program


LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.00005, 0.005))
AFFINE_SCALE_DIFF = hp.HParam('affine_scale_diff', hp.RealInterval(0.01, 0.3))
AFFINE_TRANSLATE_PERCENT = hp.HParam('affine_translate_percent', hp.RealInterval(0.01, 0.3))
AFFINE_ROTATION = hp.HParam('affine_rotation', hp.IntInterval(0, 20))
AFFINE_SHEAR = hp.HParam('affine_shear', hp.IntInterval(0, 20))
AFFINE_PROB = hp.HParam('affine_prob', hp.RealInterval(0.0, 1.0))
CHANNELSHUFFLE_PROB = hp.HParam('channelshuffle_prob', hp.RealInterval(0.0, 1.0))
ADD_AMOUNT = hp.HParam('add_amount', hp.IntInterval(0, 100))
ADD_GAUSSIAN_AMOUNT = hp.HParam('add_gaussian_amount', hp.RealInterval(0.01, 0.5))
MULTIPLY_AMOUNT = hp.HParam('multiply_amount', hp.RealInterval(0.01, 0.5))
DROPOUT_PERCENT = hp.HParam('dropout_percent', hp.RealInterval(0.01, 0.25))
C_DROPOUT_PERCENT = hp.HParam('c_dropout_percent', hp.RealInterval(0.01, 0.25))
C_DROPOUT_SIZE_PERCENT = hp.HParam('c_dropout_size_percent', hp.RealInterval(0.01, 0.25))
INVERT_PROB = hp.HParam('invert_prob', hp.RealInterval(0.01, 1.0))
JPEG_COMPRESSION = hp.HParam('jpeg_compression', hp.RealInterval(0.01, 0.8))
GAUSSIAN_SIGMA = hp.HParam('gaussian_sigma', hp.IntInterval(1, 7))
MOTIONBLUR_KERNEL = hp.HParam('motionblur_kernel', hp.IntInterval(3, 10))
HS_MULTIPLIER = hp.HParam('hs_multiplier', hp.RealInterval(0.0, 1.0))
SATURATION_REMOVER = hp.HParam('saturation_remover', hp.RealInterval(0.0, 1.0))
COLOR_TEMP_SHIFT = hp.HParam('color_temp_shift', hp.IntInterval(1000, 40000))
CONTRAST_GAMMA = hp.HParam('contrast_gamma', hp.RealInterval(0.0, 1.0))
CLOUD_SNOW_PROB = hp.HParam('cloud_snow_prob', hp.RealInterval(0.0, 0.1))

def create_hparams():
    input_file = "/workspace/PaddleOCR/configs/rec/v4.yml"
    output_file = "/workspace/PaddleOCR/configs/rec/v4_overwritten.yml"

    with open(input_file, "r") as f: config = yaml.safe_load(f)
    config["Optimizer"]["lr"]["learning_rate"] = round(LEARNING_RATE.domain.sample_uniform(), 5)
    for transform in config["Train"]["dataset"]["transforms"]:
        if "RecAug" in transform:
            hparams = {
                "learning_rate": config["Optimizer"]["lr"]["learning_rate"],
                "affine_scale_diff": round(AFFINE_SCALE_DIFF.domain.sample_uniform(), 2),
                "affine_translate_percent": round(AFFINE_TRANSLATE_PERCENT.domain.sample_uniform(), 2),
                "affine_rotation": AFFINE_ROTATION.domain.sample_uniform(),
                "affine_shear": AFFINE_SHEAR.domain.sample_uniform(),
                "affine_prob": round(AFFINE_PROB.domain.sample_uniform(), 2),
                "channelshuffle_prob": round(CHANNELSHUFFLE_PROB.domain.sample_uniform(), 2),
                "add_amount": ADD_AMOUNT.domain.sample_uniform(),
                "add_gaussian_amount": round(ADD_GAUSSIAN_AMOUNT.domain.sample_uniform(), 2),
                "multiply_amount": round(MULTIPLY_AMOUNT.domain.sample_uniform(), 2),
                "dropout_percent": round(DROPOUT_PERCENT.domain.sample_uniform(), 2),
                "c_dropout_percent": round(C_DROPOUT_PERCENT.domain.sample_uniform(), 2),
                "c_dropout_size_percent": round(C_DROPOUT_SIZE_PERCENT.domain.sample_uniform(), 2),
                "invert_prob": round(INVERT_PROB.domain.sample_uniform(), 2),
                "jpeg_compression": round(JPEG_COMPRESSION.domain.sample_uniform(), 2),
                "gaussian_sigma": GAUSSIAN_SIGMA.domain.sample_uniform(),
                "motionblur_kernel": MOTIONBLUR_KERNEL.domain.sample_uniform(),
                "hs_multiplier": round(HS_MULTIPLIER.domain.sample_uniform(), 2),
                "saturation_remover": round(SATURATION_REMOVER.domain.sample_uniform(), 2),
                "color_temp_shift": COLOR_TEMP_SHIFT.domain.sample_uniform(),
                "contrast_gamma": round(CONTRAST_GAMMA.domain.sample_uniform(), 2),
                "cloud_snow_prob": round(CLOUD_SNOW_PROB.domain.sample_uniform(), 2),
            }
            for param_name in hparams:
                transform["RecAug"][param_name] = hparams[param_name]

            with open(output_file, "w") as f: yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            return hparams

if __name__ == '__main__':
    for i in range(100):
        print(f"\n==================\nBEGINNING TRIAL {i+1}\n==================\n")
        hparams = create_hparams()
        time.sleep(3)
        config, device, logger, vdl_writer = program.preprocess(is_train=True)
        train_acc, valid_acc = train.main(config, device, logger, vdl_writer)
        # use datetime to create a unique identifier for each run
        run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=f"./trials/hparams/{run_name}")
        writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={
                'train_acc': train_acc,
                'valid_acc': valid_acc
            }
        )