# TODO: Copy from tadaconv, replace globally.

#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Config class for loading and keeping track of the configs."""

import os
import yaml
import json
import copy
import argparse
import logging

logger = logging.getLogger(__name__)

class Config(object):
    """
    Global config object. 
    It automatically loads from a hierarchy of config files and turns the keys to the 
    class attributes. 
    """
    def __init__(self, load=True, cfg_dict=None, cfg_level=None):
        """
        Args: 
            load (bool): whether or not yaml is needed to be loaded.
            cfg_dict (dict): dictionary of configs to be updated into the attributes
            cfg_level (int): indicating the depth level of the config
        """
        self._level = "cfg" + ("." + cfg_level if cfg_level is not None else "")
        if load:
            self.args = self._parse_args()
            print("Loading config from {}.".format(self.args.cfg_file))
            self.need_initialization = True
            cfg_base = self._initialize_cfg()
            cfg_dict = self._load_yaml(self.args)
            cfg_dict = self._merge_cfg_from_base(cfg_base, cfg_dict)
            self.cfg_dict = cfg_dict

        self._update_dict(cfg_dict)
        if load:
            ckp.make_checkpoint_dir(self.OUTPUT_DIR)

    def _parse_args(self):
        """
        Wrapper for argument parser. 
        """
        parser = argparse.ArgumentParser(
            description="Argparser for configuring [code base name to think of] codebase"
        )
        parser.add_argument(
            "--cfg",
            dest="cfg_file",
            help="Path to the configuration file",
            default=None
        )
        parser.add_argument(
            "--init_method",
            help="Initialization method, includes TCP or shared file-system",
            default="tcp://localhost:9999",
            type=str,
        )
        parser.add_argument(
            "opts",
            help="other configurations",
            default=None,
            nargs=argparse.REMAINDER
        )
        return parser.parse_args()

    def _path_join(self, path_list):
        """
        Join a list of paths.
        Args:
            path_list (list): list of paths.
        """
        path = ""
        for p in path_list:
            path+= p + '/'
        return path[:-1]

    def _initialize_cfg(self):
        """
        When loading config for the first time, base config is required to be read.
        """
        if self.need_initialization:
            self.need_initialization = False
            if os.path.exists('./configs/pool/base.yaml'):
                with open("./configs/pool/base.yaml", 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
            else:
                # for compatibility to the cluster
                with open("./DAMO-Action/configs/pool/base.yaml", 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg
    
    def _load_yaml(self, args, file_name=""):
        """
        Load the specified yaml file.
        Args:
            args: parsed args by `self._parse_args`.
            file_name (str): the file name to be read from if specified.
        """
        assert args.cfg_file is not None
        if not file_name == "": # reading from base file
            with open(file_name, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        else: # reading from top file
            with open(args.cfg_file, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
                file_name = args.cfg_file

        if "_BASE_RUN" not in cfg.keys() and "_BASE_MODEL" not in cfg.keys() and "_BASE" not in cfg.keys():
            # return cfg if the base file is being accessed
            return cfg

        if "_BASE" in cfg.keys():
            # load the base file of the current config file
            if cfg["_BASE"][1] == '.':
                prev_count = cfg["_BASE"].count('..')
                cfg_base_file = self._path_join(file_name.split('/')[:(-1-cfg["_BASE"].count('..'))] + cfg["_BASE"].split('/')[prev_count:])
            else:
                cfg_base_file = cfg["_BASE"].replace(
                    "./", 
                    args.cfg_file.replace(args.cfg_file.split('/')[-1], "")
                )
            cfg_base = self._load_yaml(args, cfg_base_file)
            cfg = self._merge_cfg_from_base(cfg_base, cfg)
        else:
            # load the base run and the base model file of the current config file
            if "_BASE_RUN" in cfg.keys():
                if cfg["_BASE_RUN"][1] == '.':
                    prev_count = cfg["_BASE_RUN"].count('..')
                    cfg_base_file = self._path_join(file_name.split('/')[:(-1-prev_count)] + cfg["_BASE_RUN"].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg["_BASE_RUN"].replace(
                        "./", 
                        args.cfg_file.replace(args.cfg_file.split('/')[-1], "")
                    )
                cfg_base = self._load_yaml(args, cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base, cfg, preserve_base=True)
            if "_BASE_MODEL" in cfg.keys():
                if cfg["_BASE_MODEL"][1] == '.':
                    prev_count = cfg["_BASE_MODEL"].count('..')
                    cfg_base_file = self._path_join(file_name.split('/')[:(-1-cfg["_BASE_MODEL"].count('..'))] + cfg["_BASE_MODEL"].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg["_BASE_MODEL"].replace(
                        "./", 
                        args.cfg_file.replace(args.cfg_file.split('/')[-1], "")
                    )
                cfg_base = self._load_yaml(args, cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base, cfg)
        cfg = self._merge_cfg_from_command(args, cfg)
        return cfg
    
    def _merge_cfg_from_base(self, cfg_base, cfg_new, preserve_base=False):
        """
        Replace the attributes in the base config by the values in the coming config, 
        unless preserve base is set to True.
        Args:
            cfg_base (dict): the base config.
            cfg_new (dict): the coming config to be merged with the base config.
            preserve_base (bool): if true, the keys and the values in the cfg_new will 
                not replace the keys and the values in the cfg_base, if they exist in 
                cfg_base. When the keys and the values are not present in the cfg_base,
                then they are filled into the cfg_base.
        """
        for k,v in cfg_new.items():
            if k in cfg_base.keys():
                if isinstance(v, dict):
                    self._merge_cfg_from_base(cfg_base[k], v)
                else:
                    cfg_base[k] = v
            else:
                if "BASE" not in k or preserve_base:
                    cfg_base[k] = v
        return cfg_base

    def _merge_cfg_from_command(self, args, cfg):
        """
        Merge cfg from command. Currently only support depth of four. 
        E.g. VIDEO.BACKBONE.BRANCH.XXXX. is an attribute with depth of four.
        Args:
            args : the command in which the overriding attributes are set.
            cfg (dict): the loaded cfg from files.
        """
        assert len(args.opts) % 2 == 0, 'Override list {} has odd length: {}.'.format(
            args.opts, len(args.opts)
        )
        keys = args.opts[0::2]
        vals = args.opts[1::2]

        # maximum supported depth 3
        for idx, key in enumerate(keys):
            key_split = key.split('.')
            assert len(key_split) <= 4, 'Key depth error. \nMaximum depth: 3\n Get depth: {}'.format(
                len(key_split)
            )
            assert key_split[0] in cfg.keys(), 'Non-existant key: {}.'.format(
                key_split[0]
            )
            if len(key_split) == 2:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
            elif len(key_split) == 3:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[2] in cfg[key_split[0]][key_split[1]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
            elif len(key_split) == 4:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[2] in cfg[key_split[0]][key_split[1]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[3] in cfg[key_split[0]][key_split[1]][key_split[2]].keys(), 'Non-existant key: {}.'.format(
                    key
                )


            if len(key_split) == 1:
                cfg[key_split[0]] = vals[idx]
            elif len(key_split) == 2:
                cfg[key_split[0]][key_split[1]] = vals[idx]
            elif len(key_split) == 3:
                cfg[key_split[0]][key_split[1]][key_split[2]] = vals[idx]
            elif len(key_split) == 4:
                cfg[key_split[0]][key_split[1]][key_split[2]][key_split[3]] = vals[idx]
            
        return cfg
    
    def _update_dict(self, cfg_dict):
        """
        Set the dict to be attributes of the config recurrently.
        Args:
            cfg_dict (dict): the dictionary to be set as the attribute of the current 
                config class.
        """
        def recur(key, elem):
            if type(elem) is dict:
                return key, Config(load=False, cfg_dict=elem, cfg_level=key)
            else:
                if type(elem) is str and elem[1:3]=="e-":
                    elem = float(elem)
                return key, elem
        
        dic = dict(recur(k, v) for k, v in cfg_dict.items())
        self.__dict__.update(dic)
    
    def get_args(self):
        """
        Returns the read arguments.
        """
        return self.args
    
    def __repr__(self):
        return "{}\n".format(self.dump())
            
    def dump(self):
        return json.dumps(self.cfg_dict, indent=2)

    def deep_copy(self):
        return copy.deepcopy(self)
    
if __name__ == '__main__':
    # debug
    cfg = Config(load=True)
    print(cfg.DATA)
    

def get_args(args=None):
    config_parser = parser = argparse.ArgumentParser(description='Generation Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                       help='YAML config file specifying default arguments')
    parser.add_argument('-dc', '--data_config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying validation data specific arguments')
    parser.add_argument('-gc', '--gen_config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying generation specific arguments')
    parser.add_argument('-src', '--sr_config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying super resolution specific arguments')

    parser = argparse.ArgumentParser('FourM generation script', add_help=False)    

    parser.add_argument('--run_name', type=str, default='auto')

    
    # Generation parameters
    parser.add_argument('--cond_domains', default='caption-det', type=str,
                        help='Conditioning domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--target_domains', default='tok_clip-tok_normal-tok_rgb', type=str,
                        help='Target domain names, separated by hyphen. (default: %(default)s)')
    parser.add_argument('--tokens_per_target', default='196-196-196', type=str,
                        help='Number of tokens for each target modality. (default: %(default)s)')
    parser.add_argument('--autoregression_schemes', default='maskgit-maskgit-maskgit', type=str,
                        help='Scheme of autoregressive generation for each target modality. "maskgit", "roar" or "autoregressive" (default: %(default)s)')
    parser.add_argument('--decoding_steps', default='25-25-25', type=str,
                        help='Number of decoding steps for each target modality. (default: %(default)s)')
    parser.add_argument('--token_decoding_schedules', default='cosine-cosine-cosine', type=str,
                        help='Token decoding schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--temps', default='5.0-1.0-1.0', type=str,
                        help='Starting temperature for each target modality. (default: %(default)s)')
    parser.add_argument('--temp_schedules', default='linear-linear-linear', type=str,
                        help='Temperature schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--cfg_scales', default='4.0-4.0-4.0', type=str,
                        help='Classifier-free guidance scales for each target modality. (default: %(default)s)')
    parser.add_argument('--cfg_schedules', default='constant-constant-constant', type=str,
                        help='Classifier-free guidance schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--cfg_grow_conditioning', action='store_true',
                        help='After every completed modality, add them to classifier-free guidance conditioning.')
    parser.add_argument('--no_cfg_grow_conditioning', action='store_false', dest='cfg_grow_conditioning',
                        help='Perform classifier-free guidance only on initial conditioning.')
    parser.set_defaults(cfg_grow_conditioning=True)
    parser.add_argument('--top_p', default=0.0, type=float,
                        help='top_p > 0.0: Keep the top tokens with cumulative probability >= top_p (a.k.a. nucleus filtering) (default: %(default)s)')
    parser.add_argument('--top_k', default=0.0, type=float,
                        help='top_k > 0: Keep only top k tokens with highest probability (a.k.a. top-k filtering) (default: %(default)s)')

    # Super resolution parameters
    parser.add_argument('--sr_cond_domains', default=None, type=str,
                        help='SuperRes: Conditioning domain names, separated by hyphen. If none, all base conditions and targets are used. (default: %(default)s)')
    parser.add_argument('--sr_target_domains', default='tok_clip@448-tok_rgb@448', type=str,
                        help='SuperRes: Target domain names, separated by hyphen. (default: %(default)s)')
    parser.add_argument('--sr_tokens_per_target', default='784', type=str,
                        help='SuperRes: Number of tokens for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_autoregression_schemes', default='maskgit', type=str,
                        help='SuperRes: Scheme of autoregressive generation for each target modality. "maskgit", "roar" or "autoregressive" (default: %(default)s)')
    parser.add_argument('--sr_decoding_steps', default='8', type=str,
                        help='SuperRes: Number of decoding steps for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_token_decoding_schedules', default='cosine', type=str,
                        help='SuperRes: Token decoding schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_temps', default='1.0', type=str,
                        help='SuperRes: Starting temperature for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_temp_schedules', default='linear', type=str,
                        help='SuperRes: Temperature schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_cfg_scales', default='4.0', type=str,
                        help='SuperRes: Classifier-free guidance scales for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_cfg_schedules', default='constant', type=str,
                        help='SuperRes: Classifier-free guidance schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_cfg_grow_conditioning', action='store_true',
                        help='SuperRes: After every completed modality, add them to classifier-free guidance conditioning.')
    parser.add_argument('--sr_no_cfg_grow_conditioning', action='store_false', dest='sr_cfg_grow_conditioning',
                        help='SuperRes: Perform classifier-free guidance only on initial conditioning.')
    parser.set_defaults(sr_cfg_grow_conditioning=True)
    parser.add_argument('--sr_top_p', default=0.0, type=float,
                        help='SuperRes: top_p > 0.0: Keep the top tokens with cumulative probability >= top_p (a.k.a. nucleus filtering) (default: %(default)s)')
    parser.add_argument('--sr_top_k', default=0.0, type=float,
                        help='SuperRes: top_k > 0: Keep only top k tokens with highest probability (a.k.a. top-k filtering) (default: %(default)s)')
    
    # Script parameters
    parser.add_argument('--num_samples', default=None,
                        help='Maximum number of samples to draw from the dataloader. (default: %(default)s)')
    parser.add_argument('--num_variations', default=1, type=int,
                        help='Number of variations to generate from each sample. (default: %(default)s)')
    parser.add_argument('--seed', default=0, type=int, help='Random seed ')
    
    # Tokenizer settings
    parser.add_argument('--detokenizer_steps', default=25, type=int,
                        help='Number of DDPM/DDIM steps for decoding with diffusion-based tokenizers. (default: %(default)s)')
    parser.add_argument('--rgb_tok_id', default=None, type=str,
                        help='RGB tokenizer ID (default: %(default)s)')
    parser.add_argument('--depth_tok_id', default=None, type=str,
                        help='Depth tokenizer ID (default: %(default)s)')
    parser.add_argument('--normal_tok_id', default=None, type=str,
                        help='Normal tokenizer ID (default: %(default)s)')
    parser.add_argument('--edges_tok_id', default=None, type=str,
                        help='Edges tokenizer ID (default: %(default)s)')
    parser.add_argument('--semseg_tok_id', default=None, type=str,
                        help='Semseg tokenizer ID (default: %(default)s)')
    parser.add_argument('--clip_tok_id', default=None, type=str,
                        help='CLIP tokenizer ID (default: %(default)s)')
    parser.add_argument('--dinov2_tok_id', default=None, type=str,
                        help='DINOv2 tokenizer ID (default: %(default)s)')
    parser.add_argument('--imagebind_tok_id', default=None, type=str,
                        help='ImageBind tokenizer ID (default: %(default)s)')
    parser.add_argument('--dinov2_glob_tok_id', default=None, type=str,
                        help='DINOv2 global tokenizer ID (default: %(default)s)')
    parser.add_argument('--imagebind_glob_tok_id', default=None, type=str,
                        help='ImageBind global tokenizer ID (default: %(default)s)')
    parser.add_argument('--sam_instance_tok_id', default=None, type=str,
                        help='SAM instance tokenizer ID (default: %(default)s)')
    parser.add_argument('--human_poses_tok_id', default=None, type=str,
                        help='Human poses tokenizer ID (default: %(default)s)')
    parser.add_argument('--text_tok_path', default='fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json', type=str,
                        help='Text tokenizer path (default: %(default)s)')
    
    # ControlNet parameters
    parser.add_argument('--activate_controlnet', action='store_true',
                        help='When enabled, RGB detokenizer will be replaced by RGB ControlNet.')
    parser.add_argument('--no_activate_controlnet', action='store_false', dest='activate_controlnet')
    parser.set_defaults(activate_controlnet=False)
    parser.add_argument('--controlnet_id', default=None, type=str,
                        help='RGB ControlNet ID (default: %(default)s)')
    parser.add_argument('--controlnet_guidance_scale', default=2.5, type=float,
                        help='RGB ControlNet guidance scale (default: %(default)s)')
    parser.add_argument('--controlnet_cond_scale', default=0.8, type=float,
                        help='RGB ControlNet conditioning scale (default: %(default)s)')
    
    # Model parameters
    parser.add_argument('--model', default=None, type=str, metavar='MODEL',
                        help='4M model: Hugging Face Hub ID, or path to local safetensors checkpoint (default: %(default)s)')
    parser.add_argument('--sr_model', default=None, type=str, metavar='MODEL',
                        help='Superres model: Hugging Face Hub ID, or path to local safetensors checkpoint (default: %(default)s)')
    parser.add_argument('--image_size', default=224, type=int,
                        help='Image size. (default: %(default)s)')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Base patch size for image-like modalities (default: %(default)s)')
    
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float16', 'bfloat16', 'float32', 'bf16', 'fp16', 'fp32'],
                        help='Data type (default: %(default)s')
    # Data
    parser.add_argument('--data_path', default='/mnt/datasets/cc12_multitask_224/val', 
                        help='Path to dataset (default: %(default)s)')
    parser.add_argument('--data_name', default='', type=str,
                        help='Name of dataset, used for wandb and output folder. (default: %(default)s)')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--parti_prompts_t5_embs', default=None, type=str,
                        help="(Optional) path to pre-computed T5 embeddings for PartiPrompts (in .npz format)")
    
    # Misc.
    parser.add_argument('--s3_endpoint', default='', type=str, help='S3 endpoint URL')
    parser.add_argument('--s3_path', default='', type=str, help='S3 path to model')
    parser.add_argument('--image_size_metrics', default=256, type=int,
                        help='Image size for computing FID, Inception, and CLIP metrics. (default: %(default)s)')
    parser.add_argument('--name', default='', type=str,
                        help='wandb and folder name (default: %(default)s)')
    parser.add_argument('--sr_name', default='', type=str,
                        help='SR wandb and folder name (default: %(default)s)')
    parser.add_argument('--output_dir', default='',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--num_log_images', default=100,
                        help='Number of images to log (default: %(default)s)')
    parser.add_argument('--save_all_outputs', action='store_true',
                        help='Save all conditioning and target modalities for all drawn samples as individual files.')
    parser.add_argument('--no_save_all_outputs', action='store_false', dest='save_all_outputs',
                        help='Do not save any outputs.')
    parser.set_defaults(save_all_outputs=False)
    
    # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='Log training and validation metrics to wandb')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='Project name on wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='User or team name on wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='Run name on wandb')
    parser.add_argument('--wandb_mode', default='online', type=str,
                        help='Wandb mode')
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # GPU / Distributed parameters
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    parser.add_argument('--dist_gen', action='store_true', default=False,
                        help='Enabling distributed generation')
    parser.add_argument('--no_dist_gen', action='store_false', dest='dist_gen',
                        help='Disabling distributed generation')
    parser.set_defaults(dist_gen=True)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Parse config file if there is one
    args_config, remaining = config_parser.parse_known_args(args)

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    if args_config.data_config:
        with open(args_config.data_config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    if args_config.gen_config:
        with open(args_config.gen_config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    if args_config.sr_config:
        with open(args_config.sr_config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    #The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Add the config paths if given
    args.config_path = args_config.config
    args.data_config_path = args_config.data_config
    args.gen_config_path = args_config.gen_config
    args.sr_config_path = args_config.sr_config
    
    return args