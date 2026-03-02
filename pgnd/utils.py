from typing import Union, Dict, Optional
from pathlib import Path
from omegaconf import DictConfig
import sys
import shutil
import numpy as np
import wandb
import warp as wp


Tape = wp.Tape

class CondTape(object):
    def __init__(self, tape: Optional[Tape], cond: bool = True) -> None:
        self.tape = tape
        self.cond = cond

    def __enter__(self):
        if self.tape is not None and self.cond:
            self.tape.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tape is not None and self.cond:
            self.tape.__exit__(exc_type, exc_value, traceback)


def cfg2dict(cfg: DictConfig) -> Dict:
    """
    Recursively convert OmegaConf to vanilla dict
    :param cfg:
    :return:
    """
    cfg_dict = {}
    for k, v in cfg.items():
        if type(v) == DictConfig:
            cfg_dict[k] = cfg2dict(v)
        else:
            cfg_dict[k] = v
    return cfg_dict


class Logger:

    def __init__(self, cfg, project='deformable_dynamics', entity=None):
        # Extract object_name and ep_idx from config to match ParticleFormer's wandb format
        import os
        mode = getattr(cfg.train, 'mode', None)
        object_name = getattr(cfg.train, 'object_name', None)
        if object_name is None:
            object_name = os.path.basename(str(cfg.train.source_dataset_name)) if cfg.train.source_dataset_name else ""
        ep_idx = cfg.train.training_start_episode
        
        # If entity is None, wandb will use the default logged-in user's entity
        if entity is None:
            wandb.init(project=project, name=cfg.train.name)
        else:
            wandb.init(project=project, entity=entity, name=cfg.train.name)
        
        # Build config matching ParticleFormer's format
        config_dict = cfg2dict(cfg)
        config_dict['method'] = 'PGND'
        config_dict['object_name'] = object_name
        config_dict['ep_idx'] = ep_idx
        config_dict['train_episodes'] = list(range(cfg.train.training_start_episode, cfg.train.training_end_episode))
        config_dict['test_episodes'] = list(range(cfg.train.eval_start_episode, cfg.train.eval_end_episode))
        if mode is None:
            # Backward compatible inference when mode isn't explicitly configured.
            is_multi = (cfg.train.training_start_episode != cfg.train.eval_start_episode or
                        cfg.train.training_end_episode != cfg.train.eval_end_episode)
            mode = 'multi-episode' if is_multi else 'episode'
        config_dict['mode'] = mode
        config_dict['train_objects'] = list(getattr(cfg.train, 'train_objects', []))
        config_dict['test_objects'] = list(getattr(cfg.train, 'test_objects', []))
        wandb.config.update(config_dict, allow_val_change=True)
    
    def add_scalar(self, tag, scalar, step=None):
        wandb.log({tag: scalar}, step=step)

    def add_image(self, tag, img, step=None, scale=True):
        if scale:
            img = (img - img.min()) / (img.max() - img.min())
        wandb.log({tag: wandb.Image(img)}, step=step)

    def add_video(self, tag, video, step=None):
        wandb.log({tag: wandb.Video(video)}, step=step)


def mkdir(path: Path, resume=False, overwrite=False) -> None:
    while True:
        if overwrite:
            if path.is_dir():
                print('overwriting directory ({})'.format(path))
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
            return
        elif resume:
            print('resuming directory ({})'.format(path))
            path.mkdir(parents=True, exist_ok=True)
            return
        else:
            if path.exists():
                # If not a TTY (non-interactive), default to overwrite
                if not sys.stdin.isatty():
                    print('Non-interactive environment detected, defaulting to overwrite for ({})'.format(path))
                    overwrite = True
                    continue
                
                try:
                    feedback = input('target directory ({}) already exists, overwrite? [Y/r/n] '.format(path))
                    ret = feedback.casefold() if feedback else 'y'
                except EOFError:
                    print('EOF detected, defaulting to overwrite for ({})'.format(path))
                    ret = 'y'
            else:
                ret = 'y'
                
            if ret == 'n':
                sys.exit(0)
            elif ret == 'r':
                resume = True
            elif ret == 'y' or ret == '':
                overwrite = True


def get_root(path: Union[str, Path], name: str = '.root') -> Path:
    root = Path(path).resolve()
    while not (root / name).is_file():
        root = root.parent
    return root
