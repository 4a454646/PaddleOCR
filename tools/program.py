# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import platform
import yaml
import time
import datetime
import paddle
import paddle.distributed as dist
from tqdm import tqdm
import cv2
import numpy as np
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from ppocr.utils.stats import TrainingStats
from ppocr.utils.save_load import save_model
from ppocr.utils.utility import print_dict, AverageMeter
from ppocr.utils.logging import get_logger
from ppocr.utils.loggers import VDLLogger, WandbLogger, Loggers
from ppocr.utils import profiler
from ppocr.data import build_dataloader
from ppocr.losses import build_loss
import torch


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format ' \
                 '\"key1=value1;key2=value2;key3=value3\".'
        )

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config


def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config


def check_device(use_gpu, use_xpu=False, use_npu=False, use_mlu=False):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config {} cannot be set as true while your paddle " \
          "is not compiled with {} ! \nPlease try: \n" \
          "\t1. Install paddlepaddle to run model on {} \n" \
          "\t2. Set {} as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and use_xpu:
            print("use_xpu and use_gpu can not both be ture.")
        if use_gpu and not paddle.is_compiled_with_cuda():
            print(err.format("use_gpu", "cuda", "gpu", "use_gpu"))
            sys.exit(1)
        if use_xpu and not paddle.device.is_compiled_with_xpu():
            print(err.format("use_xpu", "xpu", "xpu", "use_xpu"))
            sys.exit(1)
        if use_npu:
            if int(paddle.version.major) != 0 and int(
                    paddle.version.major) <= 2 and int(
                        paddle.version.minor) <= 4:
                if not paddle.device.is_compiled_with_npu():
                    print(err.format("use_npu", "npu", "npu", "use_npu"))
                    sys.exit(1)
            # is_compiled_with_npu() has been updated after paddle-2.4
            else:
                if not paddle.device.is_compiled_with_custom_device("npu"):
                    print(err.format("use_npu", "npu", "npu", "use_npu"))
                    sys.exit(1)
        if use_mlu and not paddle.device.is_compiled_with_mlu():
            print(err.format("use_mlu", "mlu", "mlu", "use_mlu"))
            sys.exit(1)
    except Exception as e:
        pass


def to_float32(preds):
    if isinstance(preds, dict):
        for k in preds:
            if isinstance(preds[k], dict) or isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, list):
        for k in range(len(preds)):
            if isinstance(preds[k], dict):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, paddle.Tensor):
        preds = preds.astype(paddle.float32)
    return preds


def train(config,
          train_dataloader,
          valid_dataloader,
          device,
          model,
          loss_class,
          optimizer,
          lr_scheduler,
          post_process_class,
          eval_class,
          pre_best_model_dict,
          logger,
          log_writer=None,
          scaler=None,
          amp_level='O2',
          amp_custom_black_list=[],
          visualizer=None):

    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']
    profiler_options = config['profiler_options']

    global_step = 0
    if 'global_step' in pre_best_model_dict:
        global_step = pre_best_model_dict['global_step']
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        if len(valid_dataloader) == 0:
            logger.info(
                'No Images in eval dataset, evaluation during training ' \
                'will be disabled'
            )
            start_eval_step = 1e111

    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    best_model_dict.update(pre_best_model_dict)
    train_stats = TrainingStats(log_smooth_window, ['lr'])
    model_average = False
    model.train()

    use_srn = config['Architecture']['algorithm'] == "SRN"
    extra_input_models = [
        "SRN", "NRTR", "SAR", "SEED", "SVTR", "SPIN", "VisionLAN",
        "RobustScanner", "RFL", 'DRRG'
    ]
    extra_input = False
    if config['Architecture']['algorithm'] == 'Distillation':
        for key in config['Architecture']["Models"]:
            extra_input = extra_input or config['Architecture']['Models'][key][
                'algorithm'] in extra_input_models
    else:
        extra_input = config['Architecture']['algorithm'] in extra_input_models
    try:
        model_type = config['Architecture']['model_type']
    except:
        model_type = None

    algorithm = config['Architecture']['algorithm']
    ctc_loss = build_loss(config['CTCLoss'])

    start_epoch = best_model_dict[
        'start_epoch'] if 'start_epoch' in best_model_dict else 1


    max_iter = len(train_dataloader) - 1 if platform.system(
    ) == "Windows" else len(train_dataloader)


    for epoch in range(start_epoch, epoch_num + 1):
        print(f"starting epoch {epoch}")
        reader_start = time.time()
        eta_meter = AverageMeter()

        if train_dataloader.dataset.need_reset:
            train_dataloader = build_dataloader(config, 'Train', device, logger, seed=epoch)
            max_iter = len(train_dataloader)

        pbar = tqdm(total=len(train_dataloader), desc='training', position=0, leave=True)
        for idx, batch in enumerate(train_dataloader):
            profiler.add_profiler_step(profiler_options)
            if idx >= max_iter:
                break
            lr = optimizer.get_lr()
            images = batch[0]

            preds = model(images, data=batch[1:])
            loss = loss_class(preds, batch)['loss']
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            pbar.update(1)


        # eval
        if dist.get_rank() == 0:
            calc_loss = epoch % 10 == 0
            valid_metrics = eval_with(model, valid_dataloader, post_process_class, eval_class, calc_loss, loss_class, epoch, model_type, extra_input=extra_input, scaler=scaler, amp_level=amp_level, amp_custom_black_list=amp_custom_black_list)

            train_metrics = eval_with(model, train_dataloader, post_process_class, eval_class, calc_loss, loss_class, epoch, model_type, extra_input=extra_input, scaler=scaler, amp_level=amp_level, amp_custom_black_list=amp_custom_black_list)

            if calc_loss:
                visualizer.update_charts(
                    lr=optimizer.get_lr(),
                    train_acc=train_metrics['acc'],
                    train_loss=train_metrics['loss'],
                    valid_acc=valid_metrics['acc'],
                    valid_loss=valid_metrics['loss'],
                    epoch=epoch
                )
            else:
                visualizer.update_charts(
                    lr=optimizer.get_lr(),
                    train_acc=train_metrics['acc'],
                    train_loss=0,
                    valid_acc=valid_metrics['acc'],
                    valid_loss=0,
                    epoch=epoch
                )

            print(f"lr: {optimizer.get_lr()}, train_acc: {train_metrics['acc']}, train_loss: {train_metrics['loss']}, valid_acc: {valid_metrics['acc']}, valid_loss: {valid_metrics['loss']}")

            if valid_metrics[main_indicator] >= best_model_dict[main_indicator]:
                print("saving new best model!")
                best_model_dict.update(valid_metrics)
                best_model_dict['best_epoch'] = epoch
                save_model(model, optimizer, save_model_dir, logger, config, is_best=True, prefix='best_accuracy', best_model_dict=best_model_dict, epoch=epoch, global_step=global_step)

            calc_time_remaining(epoch_num, epoch, reader_start, eta_meter)

            optimizer.clear_grad()



        if dist.get_rank() == 0:
            save_model(model, optimizer, save_model_dir, logger, config, is_best=False, prefix='latest', best_model_dict=best_model_dict, epoch=epoch, global_step=global_step)
            if epoch > 0 and epoch % save_epoch_step == 0:
                save_model(model, optimizer, save_model_dir, logger, config, is_best=False, prefix='iter_epoch_{}'.format(epoch), best_model_dict=best_model_dict, epoch=epoch, global_step=global_step)

    return

def calc_time_remaining(epoch_num, epoch, reader_start, eta_meter):
    train_epoch_time = time.time() - reader_start
    eta_meter.update(train_epoch_time)
    reader_start = time.time()
    print(f"previous epoch took {int(train_epoch_time)} seconds to train")
    eta_seconds = eta_meter.avg * (epoch_num - epoch)
    hours, remainder = divmod(eta_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_remaining = f"{int(hours)}h {int(minutes)}m {int(seconds)}s remaining"
    print(f"estimated time remaining: {time_remaining}")



def eval_with(model,
         dataloader,
         post_process_class,
         eval_class,
         calc_loss,
         loss_class,
         epoch,
         model_type=None,
         extra_input=False,
         scaler=None,
         amp_level='O2',
         amp_custom_black_list=[],
         amp_custom_white_list=[],
         amp_dtype='float16'):
    if not calc_loss:
        model.eval()
    with torch.no_grad():
        if calc_loss: 
            total = len(dataloader)
        else: total = 5
        pbar = tqdm(total=total, desc=f"evaluating with{'' if calc_loss else 'out'} loss", position=0, leave=True)
        for idx, batch in enumerate(dataloader):
            if idx > 5:
                if calc_loss: pass
                else: continue
            images = batch[0]
            preds = model(images, data=batch[1:])

            batch_numpy = []
            for item in batch:
                if isinstance(item, paddle.Tensor):
                    batch_numpy.append(item.numpy())
                else:
                    batch_numpy.append(item)

            if calc_loss:
                loss = loss_class(preds, batch)['loss']
                loss.backward()
                post_result = post_process_class(preds['ctc'], batch_numpy[1])
            else: 
                post_result = post_process_class(preds, batch_numpy[1])
                loss = 0
            
            eval_class(post_result, batch_numpy)

            pbar.update(1)
        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    
    if type(loss) == int:
        metric['loss'] = loss
    else:
        metric['loss'] = loss.numpy()[0]
    return metric


def update_center(char_center, post_result, preds):
    result, label = post_result
    feats, logits = preds
    logits = paddle.argmax(logits, axis=-1)
    feats = feats.numpy()
    logits = logits.numpy()

    for idx_sample in range(len(label)):
        if result[idx_sample][0] == label[idx_sample][0]:
            feat = feats[idx_sample]
            logit = logits[idx_sample]
            for idx_time in range(len(logit)):
                index = logit[idx_time]
                if index in char_center.keys():
                    char_center[index][0] = (
                        char_center[index][0] * char_center[index][1] +
                        feat[idx_time]) / (char_center[index][1] + 1)
                    char_center[index][1] += 1
                else:
                    char_center[index] = [feat[idx_time], 1]
    return char_center


def get_center(model, eval_dataloader, post_process_class):
    pbar = tqdm(total=len(eval_dataloader), desc='get center:')
    max_iter = len(eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(eval_dataloader)
    char_center = dict()
    for idx, batch in enumerate(eval_dataloader):
        if idx >= max_iter:
            break
        images = batch[0]
        start = time.time()
        preds = model(images)

        batch = [item.numpy() for item in batch]
        # Obtain usable results from post-processing methods
        post_result = post_process_class(preds, batch[1])

        #update char_center
        char_center = update_center(char_center, post_result, preds)
        pbar.update(1)

    pbar.close()
    for key in char_center.keys():
        char_center[key] = char_center[key][0]
    return char_center


def preprocess(is_train=False):
    FLAGS = ArgsParser().parse_args()
    profiler_options = FLAGS.profiler_options
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    profile_dic = {"profiler_options": FLAGS.profiler_options}
    config = merge_config(config, profile_dic)

    if is_train:
        # save_config
        save_model_dir = config['Global']['save_model_dir']
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, 'config.yml'), 'w') as f:
            yaml.dump(
                dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = '{}/train.log'.format(save_model_dir)
    else:
        log_file = None
    logger = get_logger(log_file=log_file)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global'].get('use_gpu', False)
    use_xpu = config['Global'].get('use_xpu', False)
    use_npu = config['Global'].get('use_npu', False)
    use_mlu = config['Global'].get('use_mlu', False)

    alg = config['Architecture']['algorithm']
    assert alg in [
        'EAST', 'DB', 'SAST', 'Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN',
        'CLS', 'PGNet', 'Distillation', 'NRTR', 'TableAttn', 'SAR', 'PSE',
        'SEED', 'SDMGR', 'LayoutXLM', 'LayoutLM', 'LayoutLMv2', 'PREN', 'FCE',
        'SVTR', 'ViTSTR', 'ABINet', 'DB++', 'TableMaster', 'SPIN', 'VisionLAN',
        'Gestalt', 'SLANet', 'RobustScanner', 'CT', 'RFL', 'DRRG', 'CAN',
        'Telescope'
    ]

    if use_xpu:
        device = 'xpu:{0}'.format(os.getenv('FLAGS_selected_xpus', 0))
    elif use_npu:
        device = 'npu:{0}'.format(os.getenv('FLAGS_selected_npus', 0))
    elif use_mlu:
        device = 'mlu:{0}'.format(os.getenv('FLAGS_selected_mlus', 0))
    else:
        device = 'gpu:{}'.format(dist.ParallelEnv()
                                 .dev_id) if use_gpu else 'cpu'
    check_device(use_gpu, use_xpu, use_npu, use_mlu)

    device = paddle.set_device(device)

    config['Global']['distributed'] = dist.get_world_size() != 1

    loggers = []

    if 'use_visualdl' in config['Global'] and config['Global']['use_visualdl']:
        save_model_dir = config['Global']['save_model_dir']
        vdl_writer_path = '{}/vdl/'.format(save_model_dir)
        log_writer = VDLLogger(vdl_writer_path)
        loggers.append(log_writer)
    if ('use_wandb' in config['Global'] and
            config['Global']['use_wandb']) or 'wandb' in config:
        save_dir = config['Global']['save_model_dir']
        wandb_writer_path = "{}/wandb".format(save_dir)
        if "wandb" in config:
            wandb_params = config['wandb']
        else:
            wandb_params = dict()
        wandb_params.update({'save_dir': save_dir})
        log_writer = WandbLogger(**wandb_params, config=config)
        loggers.append(log_writer)

    else:
        log_writer = None
    print_dict(config, logger)

    if loggers:
        log_writer = Loggers(loggers)
    else:
        log_writer = None

    logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            device))
    return config, device, logger, log_writer
