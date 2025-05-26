import os
import copy
import json
import shutil
import cv2
import numpy as np
import torch
import open3d as o3d
import pybullet
import gym
from gym import spaces
from torch import nn
from concurrent.futures import ThreadPoolExecutor
from omegaconf import OmegaConf
import pytorch3d.ops as torch3d_ops     
import base64

from PartInstruct.PartGym.env.backend.planner.panda_arm import PandaArm
from PartInstruct.PartGym.env.backend.planner.bullet_planner import OracleChecker
from PartInstruct.PartGym.env.backend.utils.semantic_parser import SemanticParser, mapping_rotate, mapping_translate
from PartInstruct.PartGym.env.backend.utils.language_encoder import T5Encoder
import PartInstruct.PartGym.env.backend.bullet_sim as bullet_sim
from PartInstruct.PartGym.env.backend.bullet_sim import save_image, save_depth
from PartInstruct.PartGym.env.backend.utils.vision_utils import *
from PartInstruct.PartGym.env.backend.utils.transform import *
from PartInstruct.PartGym.env.backend.utils.perception import *
from PartInstruct.PartGym.env.backend.utils.scene_utils import *
from PartInstruct.PartGym.env.backend.utils.vision_utils import *

from PartInstruct.PartGym.env.bullet_env_sam import BulletEnv_SAM
from PartInstruct.PartGym.env.backend.utils.sam_utils import *
from sam2.build_sam import build_sam2_camera_predictor
from PartInstruct.PartGym.env.backend.utils.vlm_utils import GPT_Planner

RESAMPLE_SPATIAL_SKILLS = ["8", "10", "12", "13", "14", "15", "16", "17"]
TWO_STAGE_SKILLS = ["5", "6", "9", "12", "13", "14", "15", "17"]

class BulletEnv_SAM_GPT(BulletEnv_SAM):
    """
    This environment uses the GPT as high-level planner to generate the skill chain and the skill parameters.
    """
    def __init__(self, config_path=None, gui=False, obj_class=None, random_sample=False, 
                 evaluation=False, split='val', task_type=None, record=False, check_tasks=True, track_samples=False, 
                 replica_scene=False, skill_mode=True, debug_output=None):
        super(BulletEnv_SAM_GPT, self).__init__(config_path, gui, obj_class, random_sample, 
                                          evaluation, split, task_type, record, track_samples, 
                                          replica_scene, skill_mode, debug_output)

        self.actual_chain_params = []
        self.gpt_planner = GPT_Planner(self.dataset_meta_path)
        self.checker = OracleChecker(self)
        self.counter = 0
        self.skill_avg_steps = {}
        self.skill_avg_steps["grasp_obj"] = 130
        self.skill_avg_steps["move_gripper"] = 30 # 20
        self.skill_avg_steps["touch_obj"] = 68
        self.skill_avg_steps["release_obj"] = 40
        self.skill_avg_steps["rotate_obj"] = 22
        self.executed_skill_chain = []  
        self.is_skill_done = []
        self.cur_skill_instruction = ""
        self.part_name_error = False
        self.skill_chain_error = False
        self.done_task = False
        self.done_cur_skill = False
        self.cur_skill = ""
        self.action_list = []
        self.reward = 0
        self.skill_timeout = False

    def sample_chain_params(self):
        with open(self.dataset_meta_path, 'r') as file:
            dataset_meta = json.load(file)
        if not self.track_samples:
            self.seed()
        chain_params_list = dataset_meta[self.obj_class][self.split][self.skill_chain]
        episode_info = self.np_random.choice(chain_params_list)
        self.obj_id = episode_info["obj_id"]
        self.ep_id = episode_info["ep_id"]
        self.task_instruction = episode_info["task_instruction"]
        self.skill_instructions = episode_info["skill_instructions"]
        self.actual_chain_params = episode_info["chain_params"]
        print("task_instruction: " + str(self.task_instruction))
        
        self.gpt_skill_chain, object_name, self.gpt_skill_instructions = self.gpt_planner.gpt4_infer_task(self.task_instruction)
        self.chain_params = self.gpt_skill_chain
        self.obj_init_position = episode_info["obj_pose"][4:]
        self.obj_init_position[-1] = 0.2
        self.obj_init_orientation = episode_info["obj_pose"][:4]
        self.obj_scale = episode_info["obj_scale"]

    def save_frame_as_base64(self, frame):
        # Convert from (Channels, Height, Width) -> (Height, Width, Channels) if needed
        if frame.shape[0] == 3 or frame.shape[0] == 4:
            frame = frame.transpose(1, 2, 0)
        # Encode the frame as a base64 string
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def _get_info(self):
        info = {
            "ep_id": self.ep_id,
            "Task Success": self.done_task,
            "part_name_error": self.part_name_error,
            "skill_chain_error": self.skill_chain_error,
            "Skill_timeout": self.skill_timeout,
            "Completion Rate": self.completion_rate,
            "Current Skill": self.cur_skill,
            "GPT Current Skill Success": self.done_cur_skill,
            "Steps": self.num_steps,
            "Object Pose": list(self.current_state["obj_pose"].to_list()),
            "TCP Pose": list(self.current_state["tcp_pose"].to_list()),
            "Joint States": list(self.current_state["joint_states"]),
            "Gripper State": list(self.current_state["gripper_state"]),
            "Action": list(self.action_list),
            "self.reward": self.reward,
            "Instruction": self.task_instruction ,
            "gpt_chain_params": self.executed_skill_chain,
            "actual_chain_params": self.actual_chain_params,
            "is_skill_done": self.is_skill_done
        }
        if self.evaluation:
            info.update({
                "ep_id": self.ep_id,
                "Action": self.action_list,
                "Instruction": self.instruction,
                "Object id": self.obj_id,
                "Task type": self.task_type,
                "Object scale": self.obj_scale,
                "Object init pose": self.obj_init_position,
                "Object init orient": self.obj_init_orientation,
                "obj_class": self.obj_class
            })

        return info


    def step(self, action, gain=0.01, gain_gripper=0.01, gain_near_target=0.01):
        # Execute one time step within the environment
        self.action_list.append(action)
        if self.num_steps == 0:
            self.checker.reset_skill(self.gpt_skill_chain[0]["skill_name"], self.gpt_skill_chain[0]["params"])
        self._take_action(action, gain, gain_gripper, gain_near_target)
        self.num_steps+=1
        observation = self._get_observation()
        self.last_state = self.current_state
        self.current_state = copy.deepcopy(observation)
        position, orientation = self.world.get_body_pose(self.obj)
        self.current_state["obj_pose"] = Transform(Rotation.from_quat(list(orientation)), list(position))

        tcp_pose = self.robot.get_tcp_pose()
        tcp_position = tcp_pose.translation
        tcp_orientation = tcp_pose.rotation.as_quat()
        self.current_state["tcp_pose"] = Transform(Rotation.from_quat(list(tcp_orientation)), list(tcp_position))
        self.reward = 1.0 # Dummy reward
        resample_spatial = False

        try:
            self.done_task = self._check_if_done_task()
        except Exception as e:
            print(f"Error caused by infestible chain params given by GPT: {e}")
            done = True
            info = self._get_info()
            self.skill_chain_error = True
            return observation, self.reward, done, info
        done = self.done_task

        self.done_cur_skill = False
        part_keys = [key for key in self.cur_skill_params if "part" in key]

        # Part name does not exist
        if part_keys != [] and self.cur_skill_params[part_keys[0]] not in self.parser.part_pcds: 
            best_match = self.gpt_planner.find_best_match(self.cur_skill_params[part_keys[0]], self.parser.part_pcds)
            if best_match:
                print(f"Updated '{self.cur_skill_params[part_keys[0]]}' to closest match: '{best_match}'")
                self.cur_skill_params[part_keys[0]] = best_match
            else:
                done = True
                print(f"No suitable match found for '{self.cur_skill_params[part_keys[0]]}'")
                self.cur_skill_params[part_keys[0]] = ""
                self.part_name_error = True
                info = self._get_info()
                return observation, self.reward, done, info

        skill_avg = self.skill_avg_steps[self.cur_skill]

        if self.counter == 0:
            self.first_frame_av = observation["agentview_rgb"]
            self.first_tcp_pose = list(self.current_state["tcp_pose"].to_list())
        cur_skill = copy.deepcopy(self.cur_skill)
        
        if self.counter > skill_avg:
            self.done_cur_skill = self.checker.is_skill_done()
            self.is_skill_done.append(self.done_cur_skill)
    
            if self.done_cur_skill:
                print("Skill " + str(self.cur_skill_idx) + " completed.")
                self.cur_skill_idx+=1
                self.counter = 0

            if self.cur_skill_idx >= len(self.actual_chain_params) * 2:
                done = True
                self.skill_timeout = True
                info = self._get_info()
                return observation, self.reward, done, info

            if not done:
                current_rgb_base64 = self.save_frame_as_base64(self.current_state["agentview_rgb"])
                self.gripper_state = self.robot.get_gripper_state()
                self.last_tcp_pose = list(self.current_state["tcp_pose"].to_list())
                next_skill_params = None
                while next_skill_params == None:
                    next_skill_params, object_name, next_skill_instruction = self.gpt_planner.gpt4_infer_next_skill(
                            self.task_instruction,  # Current task instruction
                            self.executed_skill_chain,  # Already executed skills
                            current_rgb_base64,  # Current RGB image
                            self.gripper_state, self.first_tcp_pose, self.last_tcp_pose
                        )

                self.checker.reset_skill(next_skill_params["skill_name"], next_skill_params["params"])

                self.executed_skill_chain.append({
                    "skill_name": next_skill_params["skill_name"],
                    "params": next_skill_params["params"]
                })
                print(next_skill_params)
                print(next_skill_instruction)
                self.cur_skill_instruction = next_skill_instruction
                self.cur_skill = next_skill_params["skill_name"]
                self.cur_skill_params = next_skill_params["params"]
                part_keys = [key for key in self.cur_skill_params if "part" in key]
                # For the next skill instruction, part keys does not exist
                print(part_keys)
                if part_keys != [] and self.cur_skill_params[part_keys[0]] not in self.parser.part_pcds:
                    best_match = self.gpt_planner.find_best_match(self.cur_skill_params[part_keys[0]], self.parser.part_pcds)
                    if best_match:
                        print(f"Updated '{self.cur_skill_params[part_keys[0]]}' to closest match: '{best_match}'")
                        self.cur_skill_params[part_keys[0]] = best_match
                    else:
                        print(f"No suitable match found for '{self.cur_skill_params[part_keys[0]]}'")
                        self.cur_skill_params[part_keys[0]] = ""
                        done = True
                        info = self._get_info()
                        self.part_name_error = True
                        return observation, self.reward, done, info

        if not self.skill_chain in RESAMPLE_SPATIAL_SKILLS:
            resample_spatial = True
        try:
            self.semantic_grounding(resample_spatial)
        except Exception as e:
            print(f"Error caused by infestible chain params given by GPT: {e}")
            done = True
            self.skill_chain_error

        self.counter += 1

        info = self._get_info()
        if self.record:
            self.state_sequence_buffer.append(info)

        return observation, self.reward, done, info