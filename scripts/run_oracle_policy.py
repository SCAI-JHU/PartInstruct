from pathlib import Path
import collections
from typing import List, Dict, Union, Optional, Literal
from numpy.typing import ArrayLike

import numpy as np
import pybullet
import math
import os
import gc
import time
import requests
import json
import argparse

from PartInstruct.PartGym.env.backend.planner.vgn.vgn_detection import VGN
from PartInstruct.PartGym.env.backend.planner.panda_arm import PandaArm
from PartInstruct.PartGym.env.backend.utils.grasp import Grasp
from PartInstruct.PartGym.env.backend.utils.perception import *
from PartInstruct.PartGym.env.backend.utils.vision_utils import * 
from PartInstruct.PartGym.env.backend.utils.transform import *
import PartInstruct.PartGym.env.backend.bullet_sim as bullet_sim
from PartInstruct.PartGym.env.backend.bullet_sim import Body
from omegaconf import OmegaConf
from PartInstruct.PartGym.env.bullet_env import BulletEnv
from PartInstruct.PartGym.env.backend.planner.bullet_planner import OracleChecker, BulletPlanner
from PartInstruct.PartGym.env.backend.utils.semantic_parser import SpatialSampler, SemanticParser
from typing import Union, List, Dict, Tuple, Optional, Literal

# Setup paths and meta
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
planner_directory = os.path.join(root_directory, "PartInstruct", "PartGym")
config_path = os.path.join(planner_directory, "config", "config_oracle.yaml")
config = OmegaConf.load(config_path)
config.data_root = os.path.join(root_directory, "data")
data_root = config.data_root
meta_path = config.meta_path

meta_path = os.path.join(data_root, meta_path)
with open(meta_path, 'r') as file:
    episode_data = json.load(file)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="oracle", help="Path to the directory to save the generated demos")
parser.add_argument("--splits", nargs='+', default=['train'], help="A list of data splits to generate demos for")
parser.add_argument("--obj_classes", nargs='+', default=[], help="Object classes")
parser.add_argument("--task_types", nargs='+', default=[], help="Task types")
parser.add_argument("--num_envs", type=int, default=16, help="Num envs to run per evaluation setting")
parser.add_argument("--time_out_multiplier", type=int, default=2, help="The starting episode ID")
parser.add_argument("--seed", type=int, default=40, help="Seed of RNG")

args = parser.parse_args()

seed = args.seed
rng = np.random.default_rng(seed)
output_dir = os.path.join(data_root, "..", "outputs", args.output_dir)
os.makedirs(output_dir, exist_ok=True)

class RobotAgent:
    def __init__(self, env: BulletEnv, planner: BulletPlanner):
        self.env = env
        self.planner = planner
        self.action_history = []
        self.available_parts = [name for name in list(self.env.parser.part_pcds_t0.keys()) if (not name==self.env.obj_class) and (not name=='middle')]
        
    def grasp_obj(self, part_grasp=""):
        ret = self.planner.grasp_obj(self.env.obj_class, part_grasp=part_grasp)
        if ret:
            self.action_history.append(f"Grasp {self.env.obj_class} by part: {part_grasp}.")
        else:
            self.action_history.append(f"Failed attempt to grasp {self.env.obj_class} by part: {part_grasp}.")

    def rotate_obj(self, part_rotate, dir_rotate, grasping=True):
        ret = self.planner.rotate_obj(part_rotate, dir_rotate, grasping=grasping)
        if ret:
            self.action_history.append(f"Rotate {self.env.obj_class}'s {part_rotate} to face {dir_rotate}.")
        else:
            self.action_history.append(f"Failed attempt to rotate {self.env.obj_class}'s {part_rotate} to face {dir_rotate}.")

    def touch_obj(self, part_touch=""):
        ret = self.planner.touch_obj(self.env.obj_class, part_touch=part_touch)
        if ret:
            self.action_history.append(f"Touch {self.env.obj_class} at part: {part_touch}.")
        else:
            self.action_history.append(f"Failed attempt to touch {self.env.obj_class} at part: {part_touch}.")

    def move_gripper(self, dir_move, distance=config.translate_distance, grasping=False, touching=False, put_down=False):
        ret = self.planner.move_gripper(dir_move, distance, grasping=grasping, touching=touching, put_down=put_down)
        if ret:
            self.action_history.append(f"Move gripper with dir_move='{dir_move}', distance={distance}.")
        else:
            self.action_history.append(f"Failed attempt to move gripper with dir_move='{dir_move}', distance={distance}.")

    def release_obj(self):
        ret = self.planner.release_obj()
        if ret:
            self.action_history.append(f"Release {self.env.obj_class}.")
        else:
            self.action_history.append(f"Fail to release {self.env.obj_class}.")

# To track success/failure of each episode
if args.splits:
    results_tracking = {split: [] for split in args.splits}
    trace_tracking = {split: [] for split in args.splits}
else:
    results_tracking = {split: [] for split in ['test1', 'test2', 'test3', 'test4', 'test5']}
    trace_tracking = {split: [] for split in ['test1', 'test2', 'test3', 'test4', 'test5']}

keys = list(episode_data.keys())
rng.shuffle(keys)

for obj_class in keys:
    if args.obj_classes and obj_class not in args.obj_classes:
        continue 
    for split in episode_data[obj_class].keys():
        if args.splits and split not in args.splits:
            continue
        for task_type in list(episode_data[obj_class][split].keys()) :
            if args.task_types and task_type not in args.task_types:
                continue

            for i in range(args.num_envs):
                # Env setups
                env = BulletEnv(
                    config_path=config_path, gui=False, record=True, evaluation=True, skill_mode=False,
                    obj_class=obj_class, split=split, task_type=task_type, 
                    track_samples=False,
                )
                env.seed()
                env.reset()
                
                planner = BulletPlanner(env, generation_mode=True)
                checker = OracleChecker(env)

                # Instantiate the robot agent.
                agent = RobotAgent(env, planner)

                # Define a task description.
                task_description = env.task_instruction

                # Execute the next action
                done = False
                time_out = False
                done_skills = []

                for action in env.chain_params:
                    skill_name = action["skill_name"]
                    skill_params = action["params"]
                    checker.reset_skill(skill_name, skill_params)
                    getattr(agent, f"{skill_name}")(**skill_params)
                    print("skill_name", skill_name)
                    print("check", checker.is_skill_done(planner.parser))
                    if not checker.is_skill_done(planner.parser):
                        break
                    done_skills.append(skill_name)
                
                done_all_skills = (len(done_skills)==len(action))

                done = env._check_if_done_task() or done_all_skills

                # Save
                ep_dir = os.path.join(output_dir, f"{split}_{task_type}_{obj_class}_{i}")
                os.makedirs(ep_dir, exist_ok=True)

                video_path = os.path.join(ep_dir, "video.mp4")
                state_path = os.path.join(ep_dir, "state.json")
                state = {
                    "demo_ep_id": env.ep_id,
                    "task_instruction": task_description,
                    "demo_skill_instructions": env.skill_instructions,
                    "success": done,
                    "steps": env.num_steps,
                    "obj_class": obj_class,
                    "task_type": task_type,
                    "sub_id": i,
                    "trace": agent.action_history
                }

                env.save_renders(video_path, video_only=True)
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=4)

                results_tracking[split].append({
                    "demo_ep_id": env.ep_id,
                    "success": done,
                    "steps": env.num_steps,
                    "obj_class": obj_class,
                    "task_type": task_type,
                    "sub_id": i,
                })
                trace_tracking[split].append({
                    "demo_ep_id": env.ep_id,
                    "task_instruction": task_description,
                    "demo_skill_instructions": env.skill_instructions,
                    "success": done,
                    "steps": env.num_steps,
                    "obj_class": obj_class,
                    "task_type": task_type,
                    "sub_id": i,
                    "trace": agent.action_history
                })

results_tracking_file_path = os.path.join(output_dir, "results_tracking.json")
trace_tracking_file_path = os.path.join(output_dir, "trace_tracking.json")
with open(results_tracking_file_path, 'w') as f:
    json.dump(results_tracking, f, indent=4)
with open(trace_tracking_file_path, 'w') as f:
    json.dump(trace_tracking, f, indent=4)