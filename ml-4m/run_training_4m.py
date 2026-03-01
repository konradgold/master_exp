# Copyright 2024 EPFL and Apple Inc.
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
import datetime
import json
import math
import os
import resource
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from tokenizers import Tokenizer

import fourm.utils as utils
from fourm.data import build_mixture_dataloader, get_train_dataloader, get_val_dataloader, setup_sampling_mod_info
from fourm.models import fm
from fourm.data.modality_info import MODALITY_INFO
from fourm.utils import NativeScalerWithGradNormCount as NativeScaler
from fourm.utils import create_model
from fourm.utils.optim_factory import create_optimizer


def setup_modality_info(cfg: DictConfig):
    # Global modality info
    modality_info = {mod: MODALITY_INFO[mod] for mod in cfg.all_domains}
    
    # Max tokens
    for mod in modality_info:
        image_size, patch_size = modality_info[mod].get('input_size', cfg.input_size), modality_info[mod].get('patch_size', cfg.patch_size)
        num_patches = (image_size // patch_size) ** 2
        if modality_info[mod]['type'] == 'img':
            modality_info[mod]['max_tokens'] = num_patches

    return modality_info


def setup_data(cfg: DictConfig):
    # Set number of tokens for the sampling
    if cfg.min_input_tokens is None:
        cfg.min_input_tokens = cfg.num_input_tokens
    if cfg.min_target_tokens is None:
        cfg.min_target_tokens = cfg.num_target_tokens

    # Load text tokenizer
    text_tokenizer = Tokenizer.from_file(cfg.text_tokenizer_path)
    
    print(f"Loading data config from: {cfg.data_config}")
    with open(cfg.data_config, "r") as f:
        data_config = yaml.safe_load(f)
    
    # Train
    train_config = data_config['train']['datasets']

    # All input and output domains from potentially multiple datasets
    cfg.in_domains = sorted(set.union(*[set(dataset_cfg['in_domains'].split('-')) for dataset_cfg in train_config.values()]))
    cfg.out_domains = sorted(set.union(*[set(dataset_cfg['out_domains'].split('-')) for dataset_cfg in train_config.values()]))
    cfg.all_domains = sorted(list(set(cfg.in_domains) | set(cfg.out_domains)))

    # Set up shared modality info
    modality_info = setup_modality_info(cfg)

    # Initialize (multiple) train loaders
    # Each train loader needs to be split by node if there are multiple
    if any([dataset_cfg['data_path'].startswith('s3') for dataset_cfg in train_config.values()]):
        utils.s3_utils.override_wds_s3_tar_loading(cfg.s3_data_endpoint, cfg.s3_multipart_threshold_mb, cfg.s3_multipart_chunksize_mb, cfg.s3_max_io_queue)
    num_trainsets = len(train_config)
    train_iters = []
    shards_per_dataset = [] # For computing max number of workers
    for dataset_name, dataset_cfg in train_config.items():
        print(f'Setting up dataset {dataset_name} / train')
        dataset_mod_info, sampling_weights = setup_sampling_mod_info(dataset_cfg, modality_info)
        dataset_batch_size = None #cfg.batch_size if num_trainsets == 1 else None
        epoch_size = None #cfg.epoch_size if num_trainsets == 1 else None
        dataiter = get_train_dataloader(
            dataset_config=dataset_cfg, modality_info=dataset_mod_info, 
            sampling_weights=sampling_weights, text_tokenizer=text_tokenizer, input_size=cfg.input_size, 
            num_input_tokens=cfg.num_input_tokens, num_target_tokens=cfg.num_target_tokens,
            min_input_tokens=cfg.min_input_tokens, min_target_tokens=cfg.min_target_tokens,
            num_tasks=cfg.num_tasks, num_workers=cfg.num_workers, dataset_batch_size=dataset_batch_size,
            epoch_size=epoch_size
        )
        train_iters.append(dataiter)
        if hasattr(dataiter, 'n_shards'):
            shards_per_dataset.append(dataiter.n_shards)

    num_workers = min(min(shards_per_dataset), cfg.num_workers) if shards_per_dataset else cfg.num_workers

    # When there are multiple train loaders, create a wrapper to sample from all of them
    weights = data_config['train'].get('weights', [1.0] * num_trainsets) # Default is equal weighting
    epoch_size = cfg.epoch_size
    data_loader_train = build_mixture_dataloader(
        data_iters=train_iters, weights=weights, modality_info=modality_info, 
        batch_size=cfg.batch_size, num_workers=num_workers, 
        epoch_size=epoch_size, num_gpus=cfg.num_tasks
    )
    num_training_steps_per_epoch = epoch_size // (cfg.batch_size * cfg.num_tasks)

    # Val
    if 'val' in data_config:
        val_config = data_config['val']['datasets']

        data_loaders_val, data_loaders_fixed_eval = {}, {}
        for dataset_name, dataset_cfg in val_config.items():

            dataset_mod_info, sampling_weights = setup_sampling_mod_info(train_config[dataset_name], modality_info)

            data_loaders_val[dataset_name] = get_val_dataloader(
                dataset_config=dataset_cfg, dataset_name=dataset_name, train_configs=train_config,
                modality_info=dataset_mod_info, sampling_weights=sampling_weights, text_tokenizer=text_tokenizer,
                input_size=cfg.input_size, num_input_tokens=cfg.num_input_tokens, num_target_tokens=cfg.num_target_tokens,
                min_input_tokens=cfg.min_input_tokens, min_target_tokens=cfg.min_target_tokens, fixed_eval=False, 
                fixed_eval_input_tokens=cfg.fixed_eval_input_tokens, fixed_eval_target_tokens=cfg.fixed_eval_target_tokens,
                dist_eval=cfg.dist_eval, num_tasks=cfg.num_tasks, num_workers=cfg.num_workers,
                batch_size=int(1.5*cfg.batch_size), pin_mem=cfg.pin_mem,
            )
            if cfg.fixed_eval:
                data_loaders_fixed_eval[dataset_name] = get_val_dataloader(
                    dataset_config=dataset_cfg, dataset_name=dataset_name, train_configs=train_config,
                    modality_info=dataset_mod_info, sampling_weights=sampling_weights, text_tokenizer=text_tokenizer,
                    input_size=cfg.input_size, num_input_tokens=cfg.num_input_tokens, num_target_tokens=cfg.num_target_tokens,
                    min_input_tokens=cfg.min_input_tokens, min_target_tokens=cfg.min_target_tokens, fixed_eval=True, 
                    fixed_eval_input_tokens=cfg.fixed_eval_input_tokens, fixed_eval_target_tokens=cfg.fixed_eval_target_tokens,
                    dist_eval=cfg.dist_eval, num_tasks=cfg.num_tasks, num_workers=cfg.num_workers,
                    batch_size=int(1.5*cfg.batch_size), pin_mem=cfg.pin_mem,
                )
  
        data_loaders_fixed_eval = data_loaders_fixed_eval if data_loaders_fixed_eval else None

    else:
        data_loaders_val, data_loaders_fixed_eval = None, None

    return modality_info, data_loader_train, num_training_steps_per_epoch, data_loaders_val, data_loaders_fixed_eval


def get_model(cfg: DictConfig, modality_info):
    """Creates and returns model from arguments
    """
    print(f"Creating model: {cfg.model} for modalities {list(modality_info.keys())}")

    encoder_embeddings = {}
    for mod in cfg.in_domains:
        info = modality_info[mod]
        if info.get("encoder_embedding", None) is not None:
            if info["type"] == "img":
                image_size, patch_size = info.get('input_size', cfg.input_size), info.get('patch_size', cfg.patch_size)
                encoder_embeddings[mod] = info["encoder_embedding"](patch_size=patch_size, image_size=image_size)
            else:
                encoder_embeddings[mod] = info["encoder_embedding"]()

    decoder_embeddings = {}
    for mod in cfg.out_domains:
        info = modality_info[mod]
        if info.get("decoder_embedding", None) is not None:
            if info["type"] == "img":
                image_size, patch_size = info.get('input_size', cfg.input_size), info.get('patch_size', cfg.patch_size)
                decoder_embeddings[mod] = info["decoder_embedding"](patch_size=patch_size, image_size=image_size)
            else:
                decoder_embeddings[mod] = info["decoder_embedding"]()

    model = create_model(
        cfg.model,
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        num_register_tokens=cfg.num_register_tokens,
    )

    return model


@hydra.main(config_path="../conf", config_name="config_4m", version_base=None)
def main(cfg: DictConfig):
    # Convert to regular dict for compatibility with existing code
    cfg = OmegaConf.to_container(cfg, resolve=True) # type: ignore
    cfg = DictConfig(cfg)
    ## Distributed init
    utils.init_distributed_mode(cfg)
    device = torch.device(cfg.device)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (cfg.rlimit, rlimit[1]))
    
    utils.setup_run_name(cfg)
    utils.setup_s3_args(cfg)

    if cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    seed = cfg.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if not cfg.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    if cfg.dtype in ['float16', 'fp16']:
        dtype = torch.float16
    elif cfg.dtype in ['bfloat16', 'bf16']:
        dtype = torch.bfloat16
    elif cfg.dtype in ['float32', 'fp32']:
        dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {cfg.dtype}")
    
    # Distributed training variables
    num_tasks = utils.get_world_size()
    cfg.num_tasks = num_tasks
    global_rank = utils.get_rank()
    
    ## Data
    modality_info, data_loader_train, num_training_steps_per_epoch, data_loaders_val, data_loaders_fixed_eval = setup_data(cfg)

    ## Model
    model = get_model(cfg, modality_info)

    # Logger
    if global_rank == 0 and cfg.log_wandb:
        log_writer = utils.WandbLogger(cfg)
    else:
        log_writer = None

    ## Training phases / epochs
    if cfg.epochs < 0:
        if cfg.total_tokens < 0:
            print("Epochs and total tokens are both set to negative values, stopping training.")
            exit(1)
        else:
            train_dataset_size = cfg.epoch_size # or len(dataset_train)
            cfg.epochs = math.ceil(cfg.total_tokens * 1e9 / ((cfg.num_input_tokens + cfg.num_target_tokens) * train_dataset_size))
            print(f"Total tokens: {cfg.total_tokens}B")
            print(f"Setting the number of epochs accordingly to {cfg.epochs}")
    elif cfg.total_tokens > 0:
        print("Epochs and total tokens are both non-negative, stopping training.")
        exit(1)

    # Warmup
    if cfg.warmup_epochs < 0 and cfg.warmup_steps < 0:
        if cfg.warmup_tokens < 0:
            print("Warmup epochs, steps and total tokens all set to negative values, stopping training.")
            exit(1)
        else:
            cfg.warmup_steps = math.ceil(cfg.warmup_tokens * 1e9 / ((cfg.num_input_tokens + cfg.num_target_tokens) * cfg.batch_size * utils.get_world_size()))

    # Cooldown
    if cfg.cooldown_epochs < 0 and cfg.cooldown_steps < 0:
        if cfg.cooldown_tokens < 0 and cfg.scheduler in ['inverse_sqrt']:
            print("Cooldown epochs, steps and total tokens all set to negative values, stopping training.")
            exit(1)
        else:
            cfg.cooldown_steps = math.ceil(cfg.cooldown_tokens * 1e9 / ((cfg.num_input_tokens + cfg.num_target_tokens) * cfg.batch_size * utils.get_world_size()))
    
    # Frozen
    if cfg.frozen_model_epochs <= 0:
        if cfg.frozen_model_tokens > 0:
            train_dataset_size = cfg.epoch_size # or len(dataset_train)
            cfg.frozen_model_epochs = math.ceil(cfg.frozen_model_tokens * 1e9 / ((cfg.num_input_tokens + cfg.num_target_tokens) * train_dataset_size))
        else:
            print("No frozen models during training.")
    else:
        if cfg.frozen_model_tokens > 0:
            print("Frozen_model_epochs and frozen_model_tokens are both non-negative, stopping training.")
            exit(1)

    print(OmegaConf.to_yaml(cfg))

    ## Starting from pre-trained model
    if cfg.finetune:
        if cfg.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.finetune, map_location='cpu')
        else:
            checkpoint = torch.load(cfg.finetune, map_location='cpu')

        # Remove pos_emb
        # TODO: In the future, find a way to not have to store the pos_embs here
        checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if ".pos_emb" not in k}

        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model = %s" % str(model_without_ddp))
    print(f"Number of params: {n_parameters / 1e6} M")
    
    batch_size_no_accum = cfg.batch_size * utils.get_world_size()
    total_batch_size = cfg.batch_size * cfg.accum_iter * utils.get_world_size()
    cfg.lr = cfg.blr * total_batch_size / 256
    cfg.min_lr = cfg.min_blr * total_batch_size / 256
    if cfg.frozen_model_blr > 0:
        cfg.frozen_model_lr = cfg.frozen_model_blr * total_batch_size / 256
    else:
        cfg.frozen_model_lr = cfg.blr * total_batch_size / 256

    print("LR = %.8f" % cfg.lr)
    print("Min LR = %.8f" % cfg.min_lr)
    print("Total (effective) batch size = %d" % total_batch_size)
    print("Accumulate grad iterations = %d" % cfg.accum_iter)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (batch_size_no_accum * num_training_steps_per_epoch))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=cfg.find_unused_params)
    model_without_ddp = model.module

    optimizer = create_optimizer(cfg, model_without_ddp)
    loss_scaler = NativeScaler(enabled=dtype == torch.float16)

    ## LR and WD schedules
    if cfg.weight_decay_end is None:
        cfg.weight_decay_end = cfg.weight_decay

    if cfg.frozen_model_epochs > 0:
        frozen_lr_schedule_values = utils.constant_scheduler(cfg.frozen_model_lr, cfg.frozen_model_epochs, num_training_steps_per_epoch)
        frozen_wd_schedule_values = utils.constant_scheduler(cfg.weight_decay, cfg.frozen_model_epochs, num_training_steps_per_epoch)
        main_schedule_epochs = cfg.epochs - cfg.frozen_model_epochs
    else:
        frozen_lr_schedule_values = np.array([]) 
        frozen_wd_schedule_values = np.array([])
        main_schedule_epochs = cfg.epochs   
    if cfg.scheduler == 'cosine':
        main_lr_schedule_values = utils.cosine_scheduler(
            cfg.lr, cfg.min_lr, main_schedule_epochs, num_training_steps_per_epoch, 
            warmup_epochs=cfg.warmup_epochs, warmup_steps=cfg.warmup_steps
        )
        wd_schedule_values = utils.cosine_scheduler(
            cfg.weight_decay, cfg.weight_decay_end, main_schedule_epochs, num_training_steps_per_epoch
        )
    elif 'inverse_sqrt' in cfg.scheduler:
        try:
            timescale = int(cfg.scheduler.split('-')[-1])
        except:
            timescale = 10_000
        main_lr_schedule_values = utils.inverse_sqrt_scheduler(
            cfg.lr, cfg.min_lr, main_schedule_epochs, num_training_steps_per_epoch, 
            warmup_epochs=cfg.warmup_epochs, warmup_steps=cfg.warmup_steps,
            cooldown_epochs=cfg.cooldown_epochs, cooldown_steps=cfg.cooldown_steps,
            timescale=timescale
        )
        wd_schedule_values = utils.inverse_sqrt_scheduler(
            cfg.weight_decay, cfg.weight_decay_end, main_schedule_epochs, num_training_steps_per_epoch,
            cooldown_epochs=cfg.cooldown_epochs, cooldown_steps=cfg.cooldown_steps,
            timescale=timescale
        )
    else:
        raise NotImplementedError(f"Scheduler {cfg.scheduler} not implemented.")
    
    lr_schedule_values = np.concatenate((frozen_lr_schedule_values, main_lr_schedule_values))
    wd_schedule_values = np.concatenate((frozen_wd_schedule_values, wd_schedule_values))
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # Auto-load from checkpoint
    utils.auto_load_model(
        args=cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    ## Eval (on trained model)
    if cfg.eval:
        if data_loaders_val is not None:
            for dataset_name, data_loader_val in data_loaders_val.items():
                prefix = '[Eval] ' if not dataset_name else f'[Eval ({dataset_name})] '
                eval_stats = evaluate(model, data_loader_val, device,
                                    num_input_tokens=cfg.num_input_tokens,
                                    num_target_tokens=cfg.num_target_tokens,
                                    all_domains=cfg.all_domains, dtype=dtype,
                                    prefix=prefix, loss_type=cfg.loss_type)

                print("Eval Stats:" if not dataset_name else f"Eval Stats ({dataset_name}):")
                print(eval_stats)
                print()


        if data_loaders_fixed_eval is not None:
            for dataset_name, data_loader_fixed_eval in data_loaders_fixed_eval.items():
                prefix = '[Fixed Eval] ' if not dataset_name else f'[Fixed Eval ({dataset_name})] '
                fixed_eval_stats = evaluate(model, data_loader_fixed_eval, device,
                                            num_input_tokens=cfg.fixed_eval_input_tokens,
                                            num_target_tokens=cfg.fixed_eval_target_tokens,
                                            all_domains=cfg.all_domains, dtype=dtype,
                                            prefix=prefix, loss_type=cfg.loss_type)
                print("Fixed Eval Stats:" if not dataset_name else f"Fixed Eval Stats ({dataset_name}):")
                print(fixed_eval_stats)
                print()

        exit(0)

    ## Training
    print(f"Start training for {cfg.epochs} epochs")
    start_time = time.time()
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            frozen_model_epochs=cfg.frozen_model_epochs,
            loss_scaler=loss_scaler,
            accum_iter=cfg.accum_iter,
            max_norm=cfg.clip_grad,
            max_skip_norm=cfg.skip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_input_tokens=cfg.num_input_tokens,
            num_target_tokens=cfg.num_target_tokens,
            all_domains=cfg.all_domains,
            dtype=dtype,
            loader_len=num_training_steps_per_epoch,
            output_dir=cfg.output_dir,
            compute_grad_norm=cfg.compute_grad_norm,
            loss_type=cfg.loss_type,
            total_batch_size=total_batch_size,
        )
        if cfg.output_dir:
            if (epoch + 1) % cfg.save_ckpt_freq == 0 or epoch + 1 == cfg.epochs:
                utils.save_model(
                    args=cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                if epoch + 1 == cfg.epochs:
                    use_s3 = len(cfg.s3_save_dir) > 0
                    utils.save_model(
                        args=cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, ckpt_name='final', use_s3=use_s3)
                    

        log_stats = {**{k: v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'input_tokens_seen_b': (epoch + 1) * num_training_steps_per_epoch * (total_batch_size / cfg.accum_iter) * cfg.num_input_tokens / 1e9,
                     'target_tokens_seen_b': (epoch + 1) * num_training_steps_per_epoch * (total_batch_size / cfg.accum_iter) * cfg.num_target_tokens / 1e9,
                     'total_tokens_seen_b': (epoch + 1) * num_training_steps_per_epoch * (total_batch_size / cfg.accum_iter) * (cfg.num_input_tokens + cfg.num_target_tokens) / 1e9,
                    }

        if data_loaders_val is not None and ((epoch + 1) % cfg.eval_freq == 0 or epoch + 1 == cfg.epochs):
            for dataset_name, data_loader_val in data_loaders_val.items():
                prefix = '[Eval] ' if not dataset_name else f'[Eval ({dataset_name})] '
                eval_stats = evaluate(model, data_loader_val, device, num_input_tokens=cfg.num_input_tokens, num_target_tokens=cfg.num_target_tokens,
                                    all_domains=cfg.all_domains, dtype=dtype, prefix=prefix, loss_type=cfg.loss_type)
                extra_stats = {**{k: v for k, v in eval_stats.items()}}
                log_stats.update(extra_stats)

        if data_loaders_fixed_eval is not None and ((epoch + 1) % cfg.eval_freq == 0 or epoch + 1 == cfg.epochs):
            for dataset_name, data_loader_fixed_eval in data_loaders_fixed_eval.items():
                prefix = '[Fixed Eval] ' if not dataset_name else f'[Fixed Eval ({dataset_name})] '
                fixed_eval_stats = evaluate(model, data_loader_fixed_eval, device, num_input_tokens=cfg.fixed_eval_input_tokens, num_target_tokens=cfg.fixed_eval_target_tokens,
                                            all_domains=cfg.all_domains, dtype=dtype, prefix=prefix, loss_type=cfg.loss_type)
                extra_stats = {**{k: v for k, v in fixed_eval_stats.items()}}
                log_stats.update(extra_stats)

        if log_writer is not None:
            log_writer.update(log_stats)

        if cfg.output_dir and utils.is_main_process():
            with open(os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    num_input_tokens: int, num_target_tokens: int, loss_type: str, device: torch.device, epoch: int, frozen_model_epochs: int, 
                    loss_scaler, accum_iter, max_norm: float = None, max_skip_norm: float = None, log_writer=None,
                    lr_scheduler=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    all_domains: List[str] = [], dtype: torch.dtype = torch.float16, loader_len: Optional[int] = None,
                    output_dir=None, compute_grad_norm=True, total_batch_size=None):
    
    model.train()
    if frozen_model_epochs > 0 and epoch < frozen_model_epochs:
        if cfg.frozen_embedding_domain is None:
            model.module.freeze_shared_params()
        
        else:
            model.module.freeze_params_except_specific_embeddings(cfg.frozen_embedding_domain)
    else:
        model.module.unfreeze_all()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, x in enumerate(metric_logger.log_every(data_loader, print_freq, iter_len=loader_len, header=header)):
        # Assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        update_grad = (step + 1) % accum_iter == 0

        if step % accum_iter == 0:
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

        mod_dict = {
            modality: {k: v.to(device, non_blocking=True) for k, v in d.items()}
            for modality, d in x.items()
            if modality in all_domains
        }

        # Only sync if we update grad (for accum_iter)
        # See https://muellerzr.github.io/blog/gradient_accumulation.html
        with nullcontext() if update_grad else model.no_sync():

            with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
                loss, mod_loss = model(mod_dict, num_encoder_tokens=num_input_tokens, num_decoder_tokens=num_target_tokens, loss_type=loss_type)

                loss_value = loss.item()
                mod_loss_values = {f'{mod}_loss': l.item() for mod, l in mod_loss.items()}

            if not math.isfinite(loss_value):
                torch.save(mod_dict, os.path.join(output_dir, "debug_mod_dict.pt"))
                print(f"Loss is {loss_value}, stopping training", file=sys.stderr)
                sys.exit(1)

            loss = loss / accum_iter
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, skip_grad=max_skip_norm,
                                    parameters=model.parameters(), compute_grad_norm=compute_grad_norm, 
                                    update_grad=update_grad)
            if update_grad:
                optimizer.zero_grad()

            if dtype == torch.float16:
                loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**mod_loss_values)
        if dtype == torch.float16:
            metric_logger.update(loss_scale=loss_scale_value) 
        min_lr = 1.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.update(mod_loss_values)

            if total_batch_size is not None:
                log_writer.update(
                    {
                        'input_tokens_seen_b': it * (total_batch_size / accum_iter) * num_input_tokens / 1e9,
                        'target_tokens_seen_b': it * (total_batch_size /accum_iter) * num_target_tokens / 1e9,
                        'total_tokens_seen_b': it * (total_batch_size / accum_iter) * (num_input_tokens + num_target_tokens) / 1e9,
                    }
                )

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    torch.cuda.empty_cache()

    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, num_input_tokens, num_target_tokens, loss_type,
             all_domains: List[str], dtype: torch.dtype = torch.float16, prefix="[Eval] "):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = prefix

    # switch to evaluation mode
    model.eval()

    print_freq = 10
    iter_len = len(data_loader) if hasattr(data_loader, '__len__') else -1 # Dealing with iterable datasets

    for x in metric_logger.log_every(data_loader, print_freq, iter_len=iter_len, header=header):

        mod_dict = {
            modality: {k: v.to(device, non_blocking=True) for k, v in d.items()}
            for modality, d in x.items()
            if modality in all_domains
        }

        with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
            loss, mod_loss = model(mod_dict, num_encoder_tokens=num_input_tokens, num_decoder_tokens=num_target_tokens, loss_type=loss_type)

            loss_value = loss.item()
            mod_loss_values = {f'{mod}_loss': l.item() for mod, l in mod_loss.items()}

        metric_logger.update(loss=loss_value)
        metric_logger.update(**mod_loss_values)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Eval averaged stats:", metric_logger)
    torch.cuda.empty_cache()
    
    return {prefix + k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':

    

    main()

