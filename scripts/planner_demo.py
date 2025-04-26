from pathlib import Path
import time
import collections
from typing import List, Dict, Union, Optional, Literal
from numpy.typing import ArrayLike

import numpy as np
import pybullet
import math
import os
import time

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

State = collections.namedtuple("State", ["tsdf", "pc"])

root_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "partgym")
config_path = os.path.join(root_directory, "config", "config.yaml")
## Open and read the YAML file
config = OmegaConf.load(config_path)
data_root = config.data_root

# Initialize the output directory and other parameters
output_root = os.path.join(data_root, "..", "output", "planner_demo")

obj_scale = 0.1
env = BulletEnv(config_path=config_path, gui=True, record=True)
checker = OracleChecker(env)
obj_class = "mug"
obj_id = "8594"
chain_params = [
    {
        "skill_name": "grasp_obj",
        "params": {
            "part_grasp": "left",
            "region_on_part": ""
        }
    },
    {
        "skill_name": "release_obj",
        "params": {}
    }
]

env.reset(obj_class=obj_class, obj_id=obj_id, chain_params=chain_params, obj_scale=obj_scale)
checker.reset_skill(chain_params[0]["skill_name"], chain_params[0]["params"])
planner = BulletPlanner(env, generation_mode=True)

# Determine the paths for saving the demo
ep_dir = os.path.join(output_root, f"episode_0")
os.makedirs(ep_dir, exist_ok=True)
video_path = os.path.join(ep_dir, "demo.mp4")
state_path = os.path.join(ep_dir, "states.json")

# Execute the skill chain
success = True

for skill in chain_params:
    skill_name = skill["skill_name"]
    params = skill["params"]
    checker.reset_skill(skill_name, params)
    print(f"Skill {skill_name} execution started")
    if skill_name == "grasp_obj":
        planner.grasp_obj(obj=obj_class, **params)
    elif skill_name == "rotate_obj":
        planner.rotate_obj(**params)
    elif skill_name == "move_gripper":
        planner.move_gripper(**params, distance=config.translate_distance)
    elif skill_name == "touch_obj":
        planner.touch_obj(obj=obj_class, **params)
    elif skill_name == "release_obj":
        planner.release_obj()
    skill_success = checker.is_skill_done()
    ## DEBUG
    print("checker", skill_success)
    print("env", env.done_cur_skill)
    success_string = "succeeded" if skill_success else "failed"
    print(f"Skill {skill_name} execution {success_string}!")
    if not skill_success:
        success = False
        break
    # print("****************************************")
    # print((env.robot.get_tcp_pose()*env.robot.T_tcp_gripper).translation)
    # print((env.robot.get_tcp_pose()*env.robot.T_tcp_gripper).rotation.as_quat())

# Save the renders and states
env.save_renders(video_path)
env.save_states(state_path)