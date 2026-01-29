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
import os
from typing import Any
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import datetime
import copy
from pathlib import Path
import yaml
import json
import warnings

# PyTorch & friends
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torchvision.transforms.functional as TF 

# Metrics 
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal import CLIPScore

# Tokenizers (text & image modalities)
from tokenizers import Tokenizer
from fourm.vq.vqvae import VQVAE, DiVAE, VQControlNet

# 4M
from fourm.utils import load_safetensors
from fourm.models.fm import FM
from fourm.data.modality_info import MODALITY_INFO
from fourm.models.generate import GenerationSampler

# Local
import fourm.utils as utils
from fourm.data.modality_info import MODALITY_INFO, MODALITY_TRANSFORMS
from fourm.models.generate import build_chained_generation_schedules, init_empty_target_modality, init_full_input_modality
from fourm.data.masking import UnifiedMasking
from fourm.data.modality_transforms import UnifiedDataTransform, CropSettingsTransform
from fourm.data.multimodal_dataset_folder import MultiModalDatasetFolder
from fourm.data import PreTokenizedImageAugmenter, RandomCropImageAugmenter
from fourm.data.dataset_utils import SubsampleDatasetWrapper
from fourm.utils.generation_datasets import PartiPromptsDataset, EmptyDataset
from fourm.utils.generation import batch_to_device
from fourm.utils.plotting_utils import decode_dict, plot_conds_and_targets, save_conds_and_targets, denormalize

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

torch.set_grad_enabled(False)


# Configuration now handled via Hydra - see conf/config_generation.yaml


def truncate_caption_for_clip(caption, clip_tokenizer, max_tokens=60):
    seq_trunc = clip_tokenizer.encode(caption)
    seq_trunc = seq_trunc[:max_tokens-1] + [seq_trunc[-1]]
    cap_trunc = clip_tokenizer.decode(seq_trunc)
    return caption[:len(cap_trunc)]

def string_to_list(input_string, dtype=float, delim='-'):
    """
    Convert a string separated by hyphens into a list of a given data type, 
    replacing invalid values with None.
    
    Args:
        input_string (str): The input string to convert.
        dtype (type): The target data type for conversion. Default is float.
        delim (str): The delimiter used to separate values in the string. Default is '-'.

    Returns:
        list: A list of values in the specified data type or None for invalid values.
    """
    if input_string is None:
        return [None]
    
    if isinstance(input_string, float) or isinstance(input_string, int):
        return [input_string]
    
    def try_cast(item, dtype):
        try:
            return dtype(item)
        except ValueError:
            return None

    return [try_cast(item, dtype) for item in input_string.split(delim)]

def repeat_if_necessary(lst, n):
        return lst * n if len(lst) == 1 else lst        

def load_model(model_id, model_class, device):
    if model_id is None:
        model = None
    elif model_id.endswith('.safetensors'):
        ckpt, config = load_safetensors(model_id)
        model = model_class(config=config)
        model.load_state_dict(ckpt)
    else:
        model = model_class.from_pretrained(model_id)
    return model.eval().to(device)

def load_tokenizers(cfg, device):
    toks = {}

    # RGB tokenizer
    if cfg.rgb_tok_id:
        toks['tok_rgb'] = load_model(cfg.rgb_tok_id, DiVAE, device)

    # Optional RGB ControlNet
    if cfg.controlnet_id:
        toks['controlnet'] = load_model(cfg.controlnet_id, VQControlNet, device)

    # Depth tokenizer
    if cfg.depth_tok_id:
        toks['tok_depth'] = load_model(cfg.depth_tok_id, DiVAE, device)

    # Normal tokenizer
    if cfg.normal_tok_id:
        toks['tok_normal'] = load_model(cfg.normal_tok_id, DiVAE, device)

    # Edges tokenizer
    if cfg.edges_tok_id:
        toks['tok_canny_edge'] = load_model(cfg.edges_tok_id, DiVAE, device)
        toks['tok_sam_edge'] = toks['tok_canny_edge']

    # Semseg tokenizer
    if cfg.semseg_tok_id:
        toks['tok_semseg'] = load_model(cfg.semseg_tok_id, VQVAE, device)

    # CLIP tokenizer
    if cfg.clip_tok_id:
        toks['tok_clip'] = load_model(cfg.clip_tok_id, VQVAE, device)

    # DINOv2 tokenizer
    if cfg.dinov2_tok_id:
        toks['tok_dinov2'] = load_model(cfg.dinov2_tok_id, VQVAE, device)

    # ImageBind tokenizer
    if cfg.imagebind_tok_id:
        toks['tok_imagebind'] = load_model(cfg.imagebind_tok_id, VQVAE, device)

    # DINOv2 global tokenizer
    if cfg.dinov2_glob_tok_id:
        toks['tok_dinov2_global'] = load_model(cfg.dinov2_glob_tok_id, VQVAE, device)

    # ImageBind global tokenizer
    if cfg.imagebind_glob_tok_id:
        toks['tok_imagebind_global'] = load_model(cfg.imagebind_glob_tok_id, VQVAE, device)

    # SAM instances
    if cfg.sam_instance_tok_id:
        toks['sam_instance'] = load_model(cfg.sam_instance_tok_id, VQVAE, device)

    # Human poses
    if cfg.human_poses_tok_id:
        toks['tok_pose'] = load_model(cfg.human_poses_tok_id, VQVAE, device)

    return toks

def get_dataset(cfg, text_tokenizer):
    
    # For unconditional generation
    if len(cfg.cond_domains) == 0:
        cfg.loaded_domains = cfg.cond_domains
        dataset = EmptyDataset(dataset_size=cfg.num_samples)
    
    # For caption->X generation using Parti Prompts
    elif cfg.data_path == 'parti_prompts':
        llm_embedder = None
        cfg.loaded_domains = cfg.cond_domains
        cfg.parti_prompts_t5_embs = None

        dataset = PartiPromptsDataset(text_tokenizer, max_length=128, parti_prompts_t5_embs=cfg.parti_prompts_t5_embs, llm_embedder=llm_embedder)
    
     # Otherwise, construct CC12M/IN1K-like pre-tokenized dataset
    else:
        # Also load RGB (for det augmentation and FID calculation)
        cfg.loaded_domains = sorted(list(set(cfg.cond_domains) | set(['rgb'])))

        modality_transforms = MODALITY_TRANSFORMS

        modality_info = {mod: MODALITY_INFO[mod] for mod in cfg.loaded_domains}
        # Max tokens
        for k in modality_info:
            num_patches = (cfg.image_size // cfg.patch_size) ** 2
            if modality_info[k]['type'] == 'img':
                modality_info[k]['max_tokens'] = num_patches
        # Dirichlet concentration parameter (Alpha)
        for k in modality_info:
            modality_info[k]["input_alphas"] = [0.]
            modality_info[k]["target_alphas"] = [0.]
            modality_info[k]["keep"] = ['all']

        if 'tok' not in '-'.join(cfg.loaded_domains):
            image_augmenter = RandomCropImageAugmenter(
                target_size=cfg.image_size, hflip=False, 
                crop_scale=(1.0,1.0), crop_ratio=(1.0,1.0)
            )
        else:
            image_augmenter = PreTokenizedImageAugmenter(target_size=cfg.image_size, no_aug=True)
            modality_transforms["crop_settings"] = CropSettingsTransform()
            cfg.loaded_domains.append("crop_settings")

        transform = transforms.Compose([
            UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
            UnifiedMasking(
                modality_info=modality_info, text_tokenizer=text_tokenizer,
                input_tokens_range=512, target_tokens_range=512
            ),
        ])

        modality_paths = {mod: modality_info[mod]['path'] for mod in modality_info if modality_info[mod].get('path', None) is not None}
        
        dataset = MultiModalDatasetFolder(
            cfg.data_path, cfg.loaded_domains, modality_paths=modality_paths,  
            modality_transforms=modality_transforms, transform=transform
        )
    
    # Subsample dataset if needed
    dataset = SubsampleDatasetWrapper(dataset, dataset_size=cfg.num_samples, seed=0, return_orig_idx=True)

    return dataset


def create_superres_input(out_dict, sr_cond_domains, sr_target_domains, sr_tokens_per_target, text_tokenizer, device):
    superres_sample = {}

    # Low-res condition and generated targets become condition for super resolution
    for domain in sr_cond_domains:
        superres_sample[domain] = out_dict[domain]

    # Initialize input modalities
    for cond_mod in sr_cond_domains:
        superres_sample = init_full_input_modality(superres_sample, MODALITY_INFO, cond_mod, device, eos_id=text_tokenizer.token_to_id("[EOS]"))
        
    # Initialize target modalities
    for target_mod, ntoks in zip(sr_target_domains, sr_tokens_per_target):
        superres_sample = init_empty_target_modality(superres_sample, MODALITY_INFO, target_mod, 1, ntoks, device)
        
    return superres_sample


@hydra.main(config_path="../conf", config_name="config_generation", version_base=None)
def main(cfg: DictConfig):

    utils.setup_run_name(cfg)

    if cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    utils.init_distributed_mode(cfg)

    device = torch.device(cfg.device)

    # Fix the seed for reproducibility
    cfg.seed = cfg.seed + utils.get_rank()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # random.seed(cfg.seed)

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
    
    if cfg.data_name == 'auto':
        cfg.data_name = Path(cfg.data_config_path).stem
    if cfg.name == 'auto':
        cfg.name = Path(cfg.gen_config_path).stem
    if cfg.sr_name == 'auto':
        cfg.sr_name = Path(cfg.sr_config_path).stem

    # Output directory
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.data_name, f'{cfg.name}--{cfg.sr_name}' if cfg.sr_name else cfg.name)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare args
    delim = '-'

    # Generation parameters
    cfg.cond_domains = sorted(list(string_to_list(cfg.cond_domains, dtype=str, delim=delim)))
    cfg.target_domains = string_to_list(cfg.target_domains, dtype=str, delim=delim)
    cfg.all_domains = sorted(list(set(cfg.cond_domains) | set(cfg.target_domains)))
    cfg.loaded_domains = sorted(list(set(cfg.cond_domains) | set(['rgb'])))
    n_targets = len(cfg.target_domains)
    cfg.tokens_per_target = repeat_if_necessary(string_to_list(cfg.tokens_per_target, dtype=int, delim=delim), n_targets)
    cfg.autoregression_schemes = repeat_if_necessary(string_to_list(cfg.autoregression_schemes, dtype=str, delim=delim), n_targets)
    cfg.decoding_steps = repeat_if_necessary(string_to_list(cfg.decoding_steps, dtype=int, delim=delim), n_targets)
    cfg.token_decoding_schedules = repeat_if_necessary(string_to_list(cfg.token_decoding_schedules, dtype=str, delim=delim), n_targets)
    cfg.temps = repeat_if_necessary(string_to_list(cfg.temps, dtype=float, delim=delim), n_targets)
    cfg.temp_schedules = repeat_if_necessary(string_to_list(cfg.temp_schedules, dtype=str, delim=delim), n_targets)
    cfg.cfg_scales = repeat_if_necessary(string_to_list(cfg.cfg_scales, dtype=float, delim=delim), n_targets)
    cfg.cfg_schedules = repeat_if_necessary(string_to_list(cfg.cfg_schedules, dtype=str, delim=delim), n_targets)

    # Super-resolution parameters
    if cfg.sr_cond_domains is None:
        cfg.sr_cond_domains = cfg.cond_domains + cfg.target_domains
    else:
        cfg.sr_cond_domains = sorted(list(string_to_list(cfg.sr_cond_domains, dtype=str, delim=delim)))
    cfg.sr_target_domains = string_to_list(cfg.sr_target_domains, dtype=str, delim=delim)
    cfg.sr_all_domains = sorted(list(set(cfg.sr_cond_domains) | set(cfg.sr_target_domains)))
    sr_n_targets = len(cfg.sr_target_domains)
    cfg.sr_tokens_per_target = repeat_if_necessary(string_to_list(cfg.sr_tokens_per_target, dtype=int, delim=delim), sr_n_targets)
    cfg.sr_autoregression_schemes = repeat_if_necessary(string_to_list(cfg.sr_autoregression_schemes, dtype=str, delim=delim), sr_n_targets)
    cfg.sr_decoding_steps = repeat_if_necessary(string_to_list(cfg.sr_decoding_steps, dtype=int, delim=delim), sr_n_targets)
    cfg.sr_token_decoding_schedules = repeat_if_necessary(string_to_list(cfg.sr_token_decoding_schedules, dtype=str, delim=delim), sr_n_targets)
    cfg.sr_temps = repeat_if_necessary(string_to_list(cfg.sr_temps, dtype=float, delim=delim), sr_n_targets)
    cfg.sr_temp_schedules = repeat_if_necessary(string_to_list(cfg.sr_temp_schedules, dtype=str, delim=delim), sr_n_targets)
    cfg.sr_cfg_scales = repeat_if_necessary(string_to_list(cfg.sr_cfg_scales, dtype=float, delim=delim), sr_n_targets)
    cfg.sr_cfg_schedules = repeat_if_necessary(string_to_list(cfg.sr_cfg_schedules, dtype=str, delim=delim), sr_n_targets)

    # Load text tokenizer
    text_tokenizer = Tokenizer.from_file(cfg.text_tok_path)

    # Load image tokenizers
    tokenizers = load_tokenizers(cfg, device)

    # Load model & define sampler
    model = load_model(cfg.model, FM, device)
    gen_sampler= GenerationSampler(model)

    # Load super-resolution model if so specified
    model_sr = load_model(cfg.sr_model, FM, device)
    gen_sampler_sr = GenerationSampler(model_sr) if model_sr is not None else None    

    # Get dataset
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    dataset = get_dataset(args, text_tokenizer)
    if cfg.dist_gen:
        if len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
        data_sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        data_sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=data_sampler,
        batch_size=1, num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem, drop_last=False,
    )

    # Logging
    if global_rank == 0 and cfg.log_wandb:
        # Edit run name and add tags
        cfg.wandb_tags = [cfg.data_name, cfg.name, cfg.wandb_run_name]
        if cfg.sr_name:
            cfg.wandb_tags.append(cfg.sr_name)
        cfg.wandb_run_name = f"{cfg.name}--{cfg.sr_name}--{cfg.data_name}--{cfg.wandb_run_name}"
        log_writer = utils.WandbLogger(cfg)
        log_writer.set_step(0)
    else:
        log_writer = None

    print('\nArguments:')
    print(cfg)
    print('')

    print('Starting generation...')
    start_time = time.time()

    # Measure generation statistics & save samples
    gen_stats = generate(gen_sampler, gen_sampler_sr, tokenizers, text_tokenizer, data_loader, device, dtype, cfg)

    if log_writer is not None:
        log_writer.update(gen_stats)

    if cfg.output_dir and utils.is_main_process():
        with open(os.path.join(cfg.output_dir, "log_eval.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(gen_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Done! Total generation time {} on device {}'.format(total_time_str, device))
    torch.cuda.empty_cache()


@torch.no_grad()
def generate(gen_sampler, gen_sampler_sr, tokenizers, text_tokenizer, data_loader, device, dtype, cfg):

    # Set up generation schedule
    schedule = build_chained_generation_schedules(
        cond_domains=cfg.cond_domains, 
        target_domains=cfg.target_domains,
        tokens_per_target=cfg.tokens_per_target,
        autoregression_schemes=cfg.autoregression_schemes, 
        decoding_steps=cfg.decoding_steps, 
        token_decoding_schedules=cfg.token_decoding_schedules,
        temps =cfg.temps,
        temp_schedules=cfg.temp_schedules,
        cfg_scales=cfg.cfg_scales, 
        cfg_schedules=cfg.cfg_schedules,
        cfg_grow_conditioning=cfg.cfg_grow_conditioning, 
    )

    # Set up super resolution schedule
    sr_schedule = build_chained_generation_schedules(
        cond_domains=cfg.sr_cond_domains, 
        target_domains=cfg.sr_target_domains,
        tokens_per_target=cfg.sr_tokens_per_target,
        autoregression_schemes=cfg.sr_autoregression_schemes, 
        decoding_steps=cfg.sr_decoding_steps, 
        token_decoding_schedules=cfg.sr_token_decoding_schedules,
        temps =cfg.sr_temps,
        temp_schedules=cfg.sr_temp_schedules,
        cfg_scales=cfg.sr_cfg_scales, 
        cfg_schedules=cfg.sr_cfg_schedules,
        cfg_grow_conditioning=cfg.sr_cfg_grow_conditioning, 
    ) if gen_sampler_sr is not None else None

    # Set up metric loggers
    fid_metric, inception_metric, clip_metric = None, None, None
    if 'tok_rgb' in cfg.target_domains:
        inception_metric = InceptionScore(
            feature='logits_unbiased', splits=10, normalize=False,
            sync_on_compute=True
        ).to(device)
        if 'rgb' in cfg.loaded_domains:
            fid_metric = FrechetInceptionDistance(
                feature=2048, reset_real_features=True, 
                normalize=False, sync_on_compute=True
            ).to(device)
        if 'caption' in cfg.cond_domains:
            clip_metric = CLIPScore(
                model_name_or_path="openai/clip-vit-large-patch14", 
                sync_on_compute=True
            ).to(device)

    # For super resolution as well (if it is performed)
    fid_metric_sr, inception_metric_sr, clip_metric_sr = None, None, None
    if gen_sampler_sr is not None and 'tok_rgb@448' in cfg.sr_target_domains:
        inception_metric_sr = InceptionScore(
            feature='logits_unbiased', splits=10, normalize=False,
            sync_on_compute=True
        ).to(device)
        if 'rgb' in cfg.loaded_domains:
            fid_metric_sr = FrechetInceptionDistance(
                feature=2048, reset_real_features=True, 
                normalize=False, sync_on_compute=True
            ).to(device)
        if 'caption' in cfg.cond_domains:
            clip_metric_sr = CLIPScore(
                model_name_or_path="openai/clip-vit-large-patch14", 
                sync_on_compute=True
            ).to(device)


    metric_logger = utils.MetricLogger(delimiter="  ")

    logged_images_count = 0

    for sample, sample_idx in metric_logger.log_every(data_loader, print_freq=1, header='Generation:'):
        sample_idx = sample_idx[0].item()

        # Sample to device
        sample = batch_to_device(sample, device, domains=cfg.loaded_domains)

        # Update FID metric with a sample from the real distribution
        if fid_metric is not None or fid_metric_sr is not None:
            rgb_real = (255 * denormalize(sample['rgb']['tensor'])).to(torch.uint8)
            rgb_real = TF.resize(rgb_real, size=cfg.image_size_metrics)
            if fid_metric is not None:
                fid_metric.update(rgb_real, real=True)
            if fid_metric_sr is not None:
                fid_metric_sr.update(rgb_real, real=True)

        # Remove RGB if it is not used as an input (just loaded to make det dataloading happy and for metrics)
        for domain in cfg.loaded_domains:
            if domain not in cfg.cond_domains and domain in sample:
                del sample[domain]

        # Initialize input modalities
        for cond_mod in cfg.cond_domains:
            sample = init_full_input_modality(sample, MODALITY_INFO, cond_mod, device, eos_id=text_tokenizer.token_to_id("[EOS]"))

        # Initialize target modalities
        for target_mod, ntoks in zip(cfg.target_domains, cfg.tokens_per_target):
            sample = init_empty_target_modality(sample, MODALITY_INFO, target_mod, 1, ntoks, device)
            
        
        dec_dicts = []
        dec_dicts_sr = []

        # Draw several samples using the same conditioning
        for i in range(cfg.num_variations):
            with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
                out_dict = gen_sampler.generate(
                    sample, schedule, text_tokenizer=text_tokenizer, verbose=False, 
                    seed=utils.generate_seed(cfg.seed, sample_idx, i),
                    top_p=cfg.top_p, top_k=cfg.top_k
                )

            # Decode tokens into images/text
            dec_dict = decode_dict(
                out_dict, tokenizers, text_tokenizer, 
                image_size=cfg.image_size, patch_size=cfg.patch_size, 
                decoding_steps=cfg.detokenizer_steps, 
                activate_controlnet=cfg.activate_controlnet,
                controlnet_guidance_scale=cfg.controlnet_guidance_scale,
                controlnet_cond_scale=cfg.controlnet_cond_scale,
            )
            dec_dicts.append(dec_dict)

            # Update metrics
            if inception_metric is not None:
                rgb_pred = TF.to_tensor(255 * dec_dict['tok_rgb']).to(dtype=torch.uint8, device=device).unsqueeze(0)
                rgb_pred = TF.resize(rgb_pred, size=cfg.image_size_metrics)
                inception_metric.update(rgb_pred)
                if fid_metric is not None:
                    fid_metric.update(rgb_pred, real=False)
                if clip_metric is not None:
                    caption_trunc = truncate_caption_for_clip(dec_dict['caption'][0], clip_metric.processor.tokenizer)
                    clip_metric.update(rgb_pred, caption_trunc)

            # Super-resolution
            if gen_sampler_sr is not None:
                with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
                    sample_sr = create_superres_input(
                        out_dict, cfg.sr_cond_domains, cfg.sr_target_domains, 
                        cfg.sr_tokens_per_target, text_tokenizer, device
                    )
                    out_dict_sr = gen_sampler_sr.generate(
                        sample_sr, sr_schedule, text_tokenizer=text_tokenizer, verbose=False, 
                        seed=utils.generate_seed(cfg.seed, sample_idx, i),
                        top_p=cfg.sr_top_p, top_k=cfg.sr_top_k,
                    )

                # Decode tokens into images/text
                dec_dict_sr = decode_dict(
                    out_dict_sr, tokenizers, text_tokenizer, 
                    image_size=448, patch_size=cfg.patch_size, 
                    decoding_steps=cfg.detokenizer_steps,
                    activate_controlnet=cfg.activate_controlnet,
                    controlnet_guidance_scale=cfg.controlnet_guidance_scale,
                    controlnet_cond_scale=cfg.controlnet_cond_scale,
                )
                dec_dicts_sr.append(dec_dict_sr)

                # Update superres metrics
                if inception_metric_sr is not None:
                    rgb_pred = TF.to_tensor(255 * dec_dict_sr['tok_rgb@448']).to(dtype=torch.uint8, device=device).unsqueeze(0)
                    rgb_pred = TF.resize(rgb_pred, size=cfg.image_size_metrics)
                    inception_metric_sr.update(rgb_pred)
                    if fid_metric_sr is not None:
                        fid_metric_sr.update(rgb_pred, real=False)
                    if clip_metric_sr is not None:
                        caption_trunc = truncate_caption_for_clip(dec_dict['caption'][0], clip_metric_sr.processor.tokenizer)
                        clip_metric_sr.update(rgb_pred, caption_trunc)
            

        # Save all-in-one plot
        if cfg.num_log_images == 'all' or (utils.is_main_process() and logged_images_count < int(cfg.num_log_images)):
            plot_conds_and_targets(
                cfg.cond_domains, cfg.target_domains, dec_dicts, 
                save_path=os.path.join(cfg.output_dir, 'plots', f'{sample_idx:06d}.jpg')
            )
            for sr_idx, sr_dec_dict in enumerate(dec_dicts_sr):
                plot_conds_and_targets(
                    cfg.sr_cond_domains, cfg.sr_target_domains, [sr_dec_dict], 
                    save_path=os.path.join(cfg.output_dir, 'plots', f'{sample_idx:06d}_sr{sr_idx}.jpg')
                )
            logged_images_count += 1

        # Save each modality separately
        if cfg.save_all_outputs:
            save_conds_and_targets(
                cfg.cond_domains, cfg.target_domains, dec_dicts, 
                save_dir=cfg.output_dir, sample_idx=sample_idx
            )

    # Compute and log metrics
    results = {}

    if inception_metric is not None:
        inception_mean, inception_std = inception_metric.compute()
        results['inception_mean'] = inception_mean.item()
        results['inception_std'] = inception_std.item()
    if fid_metric is not None:
        fid = fid_metric.compute().item()
        results['fid'] = fid
    if clip_metric is not None:
        clip_score = clip_metric.compute().item()
        results['clip_score'] = clip_score

    if inception_metric_sr is not None:
        inception_mean_sr, inception_std_sr = inception_metric_sr.compute()
        results['inception_mean_sr'] = inception_mean_sr.item()
        results['inception_std_sr'] = inception_std_sr.item()
    if fid_metric_sr is not None:
        fid_sr = fid_metric_sr.compute().item()
        results['fid_sr'] = fid_sr
    if clip_metric_sr is not None:
        clip_score_sr = clip_metric_sr.compute().item()
        results['clip_score_sr'] = clip_score_sr

    metric_logger.update(**results)
    # Gather the stats from all processes (they should already be the same since we sync the torcheval metrics after every step)
    metric_logger.synchronize_between_processes()
    print("Generation results:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()
