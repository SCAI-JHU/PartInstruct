import os
import sys
import pathlib
import hydra
import copy
import random
import json
import time
import re
from omegaconf import OmegaConf
from PartInstruct.baselines.policy.diffusion_policy import RobodiffUnetImagePolicy
from PartInstruct.baselines.training.base_workspace import BaseWorkspace
from hydra.core.global_hydra import GlobalHydra

os.environ['HYDRA_FULL_ERROR'] = '1'
OmegaConf.register_new_resolver("eval", eval, replace=True)

class Evaluator(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        output_dir = cfg.output_dir
        super().__init__(cfg, output_dir=output_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.dataset_meta_path = os.path.join(str(cfg.data_root), str(cfg.meta_path))
        print("##### Using dataset meta file from:", self.dataset_meta_path)
        
        self.model: RobodiffUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: RobodiffUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
    
    def roll_out(self, cfg):
        for obj_class in self.dataset_meta.keys():
            if self.specified_obj_classes and obj_class not in self.specified_obj_classes:
                continue 
            for split in self.dataset_meta[obj_class].keys():
                if self.specified_splits and split not in self.specified_splits:
                    continue
                task_types = self._get_task_types(obj_class, split)
                if not task_types:
                    continue
                
                env_runner = hydra.utils.instantiate(
                    cfg.task.env_runner,
                    output_dir=cfg.output_dir, obj_class=obj_class, split=split, task_types=task_types, epoch=self.epoch)

                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()
                policy.cuda()
                env_runner.run(policy)

    def _get_task_types(self, obj_class, split):
        if self.specified_task_type:
            try:
                self.dataset_meta[obj_class][split][str(self.specified_task_type)]
                return [self.specified_task_type] * self.n_envs 
            except KeyError:
                return None
        else:
            task_types = list(self.dataset_meta[obj_class][split].keys()) 
            if '11' in task_types:
                task_types.remove('11')
            k = len(task_types)
            if k < self.n_envs:
                random.seed(42)
                additional_samples = random.choices(task_types, k=self.n_envs-k)
                task_types.extend(additional_samples)
            return task_types

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        assert os.path.exists(self.dataset_meta_path)
        with open(self.dataset_meta_path, 'r') as file:
            self.dataset_meta = json.load(file)
        
        self.n_envs = cfg.task.env_runner.n_envs
        self.specified_obj_classes = cfg.obj_classes
        self.specified_task_type = cfg.task_type
        self.specified_splits = cfg.split
        self.epoch = -1
        
        if cfg.rollout_mode == 'runtime':
            self._run_runtime(cfg)
        elif cfg.rollout_mode == 'top_k':
            self._run_top_k(cfg)
        elif cfg.rollout_mode == 'specific_ckpt':
            self._run_specific_ckpt(cfg)

    def _run_runtime(self, cfg):
        self.epoch = cfg.epoch
        previous_epoch = self.epoch - 100

        while self.epoch <= cfg.max_epoch:
            ckpt_path = os.path.join(cfg.output_dir, 'checkpoints', str(self.epoch), 'latest.ckpt')
            print(f"Attempting to resume from checkpoint {ckpt_path}")
            
            checkpoint_ready = False
            while not checkpoint_ready:
                if os.path.exists(ckpt_path):
                    print(f"Resuming from checkpoint {ckpt_path}")
                    try:
                        self.load_checkpoint(path=ckpt_path)
                        checkpoint_ready = True
                    except Exception as e:
                        print(f"Failed to load checkpoint {ckpt_path}. Error: {e}. Waiting...")
                        time.sleep(10)
                else:
                    print(f"Checkpoint {ckpt_path} does not exist. Waiting...")
                    time.sleep(10)
            
            self.roll_out(cfg)
            previous_epoch = self.epoch
            self.epoch += cfg.training.checkpoint_every

    def _run_top_k(self, cfg):
        print("Running in top_k mode")
        ckpt_folder = os.path.join(cfg.output_dir, 'checkpoints')
        for ckpt_path in os.listdir(ckpt_folder):
            full_path = os.path.join(ckpt_folder, ckpt_path)
            if os.path.isfile(full_path):
                match = re.search(r'epoch=(\d+)', str(ckpt_path))
                self.load_checkpoint(path=full_path)
                self.epoch = int(match.group(1))
                print(f"Resuming from checkpoint {ckpt_path}")
                self.roll_out(cfg)

    def _run_specific_ckpt(self, cfg):
        ckpt_path = cfg.ckpt_path
        self.epoch = cfg.epoch
        self.load_checkpoint(path=ckpt_path)
        print(f"Resuming from checkpoint {ckpt_path}")
        self.roll_out(cfg)

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = Evaluator(cfg)
    workspace.run()

if __name__ == "__main__":
    main()