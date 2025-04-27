from pathlib import Path
import collections
from typing import List, Dict, Union, Optional, Literal
from numpy.typing import ArrayLike

import numpy as np
import pybullet
import math
import os
import time
import base64
import openai
import requests
import json
import cv2
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
config_path = os.path.join(planner_directory, "config", "config_CaP.yaml")
config = OmegaConf.load(config_path)
config.data_root = os.path.join(root_directory, "data")
data_root = config.data_root
meta_path = config.meta_path

meta_path = os.path.join(data_root, meta_path)
with open(meta_path, 'r') as file:
    episode_data = json.load(file)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="CaP", help="Path to the directory to save the generated demos")
parser.add_argument("--splits", nargs='+', default=['train'], help="A list of data splits to generate demos for")
parser.add_argument("--obj_classes", nargs='+', default=[], help="Object classes")
parser.add_argument("--task_types", nargs='+', default=[], help="Task types")
parser.add_argument("--num_envs", type=int, default=16, help="Num envs to run per evaluation setting")
parser.add_argument("--time_out_multiplier", type=int, default=2, help="The starting episode ID")
parser.add_argument("--seed", type=int, default=40, help="Seed of RNG")

args = parser.parse_args()

seed = args.seed
rng = np.random.default_rng(seed)
output_dir = os.path.join(data_root, "..", "output", args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# Retrieve OpenAI API key from an environment variable.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Safe Exec Utilities
def merge_dicts(dicts):
    """Merge a list of dictionaries into one."""
    return {k: v for d in dicts for k, v in d.items()}

def exec_safe(code_str, gvars, lvars):
    """
    Safely execute code_str after checking for banned phrases.
    Raises an AssertionError if a banned phrase is found.
    """
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str, f"Banned phrase found: {phrase}"
    
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([gvars, {'exec': empty_fn, 'eval': empty_fn}])
    exec(code_str, custom_gvars, lvars)

# Vision Utilities
def compute_aabb(point_cloud: np.ndarray):
    """
    Compute the axis-aligned bounding box (AABB) for a given point cloud.
    
    Parameters:
        point_cloud (np.ndarray): Array of shape (N, 3) containing 3D points.
        
    Returns:
        min_point (np.ndarray): The minimum x, y, z coordinates.
        max_point (np.ndarray): The maximum x, y, z coordinates.
        center (np.ndarray): Center of the bounding box.
        extents (np.ndarray): The width, height, and depth of the bounding box.
    """
    min_point = np.min(point_cloud, axis=0)
    max_point = np.max(point_cloud, axis=0)
    center = (min_point + max_point) / 2.0
    extents = max_point - min_point
    return min_point, max_point, center, extents

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

    def rotate_obj(self, part_rotate, dir_rotate):
        ret = self.planner.rotate_obj(part_rotate, dir_rotate)
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

    def move_gripper(self, dir_move, distance=config.translate_distance):
        ret = self.planner.move_gripper(dir_move, distance)
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

    def get_part_bbox(self, part_name):
        if part_name not in self.available_parts:
            message = f"Fail to extract part bbox info, part name {part_name} not available."
        else:
            self.planner.parser.update_part_pcds(resample_spatial=True)
            grounded_pcd = self.planner.parser.part_pcds[part_name]
            min_point, max_point, center, extents = compute_aabb(grounded_pcd)
            message = f"Check bounding box of the part '{part_name}'. min_point: {list(min_point)}, \
            max_point: {list(max_point)}, center: {list(center)}, extents: {list(extents)}."
        self.action_history.append(message)
    
    def get_object_bbox(self):
        self.planner.parser.update_part_pcds()
        grounded_pcd = self.planner.parser.part_pcds[self.env.obj_class]
        min_point, max_point, center, extents = compute_aabb(grounded_pcd)
        message = f"Check bounding box of the object {self.env.obj_class}. min_point: {list(min_point)}, \
        max_point: {list(max_point)}, center: {list(center)}, extents: {list(extents)}."
        self.action_history.append(message)

class LLMPolicy:
    def __init__(self, agent: RobotAgent, openai_api_key: str):
        """
        Initialize with the simulation environment, planner, and OpenAI API key.
        """
        self.agent = agent
        openai.api_key = openai_api_key
        
        self.prompt_template = (
            "You are an expert task planner for a stationary robot agent with a single arm. Your goal is to find "
            "the optimal next action from a pool of API action functions for the robot to perform manipulation tasks "
            "from a given instruction.\n"
            "The task assigned to you requires manipulation with only one object and its parts. \n"
            "The task description may not directly contain the action sequence, sometimes you may need reasoning " \
            "about the relationship between the task and the basic actions or think about the implication and assumptions.\n"
            "You should account for the task progress (implied by the action history), state of the object and parts, "
            "state of the robot and gripper to determine the next optimal action.\n\n"

            "Below is some important information or constraints about the task setup: \n"
            "    - All tasks are using the following direction convention: left: +y, right: -y; top: +z, bottom: -z; " \
            "front: -x, back: +x. To simplify, use these nominal terms only to indicate a direction. For example, " \
            "to express 'upwards', you will use 'top'; to express 'forwards', you use 'front'.\n"
            "    - The robot's tool center point (TCP) pose and the object's pose will be provided. Each pose is a dict "
            "with 'translation' and 'rotation' keys in world frame. The 'translation' value is a 3D array of [x, y, z] "
            "in meters. The 'rotation' value is a 4D array quaternion in [x, y, z, w].\n"
            "    - The robot's gripper state will be provided. It is a Float between {gripper_closed} and {gripper_open}. "
            "A value near {gripper_closed} indicates the gripper is closed; near {gripper_open} means open.\n"
            "    - The robot's gripper has a max opening width of {gripper_max_width} and finger depth of {finger_depth}. "
            "The dimension of the object or part to be grasped should not exceed the constraints.\n" \
            "    - The object will have multiple semantic parts (e.g. cap, handle) and spatial parts (one of "
            "'top', 'bottom', 'left', 'right', 'front', 'back'). The available parts will be provided, you should "
            "only select from the available parts to interact with.\n" \
            "    - You may choose to check the object's or any available part's bounding box by calling API function "
            "agent.get_object_bbox() or agent.get_part_bbox(part_name=<part>) as the next action (see below for detailed "
            "description), in which cases you will see the extracted results shown as the last element of the next step's "
            "action history.\n\n" \

            "Below is the detailed description of different actions that you can use for solving the task: "
            
            "    - agent.get_object_bbox(): Used to extract the current bbox of the object. Results will be shown in the "
            "next step's action history. A bbox has four 3D array components: min_point, max_point, center, extents. "
            "Example: agent.get_object_bbox()\n"

            "    - agent.get_part_bbox(part_name=<part>): Used to extract the current bbox of a part. Results will "
            "be shown in the next step's action history. A bbox has four 3D array components: min_point, max_point, "
            "center, extents. Example: agent.get_part_bbox(part_name='handle') means extracting the bbox of the part "
            "'handle'. Please note that agent.get_part_bbox(part_name=<part>) always uses the latest spatial parts, " \
            "meaning that if the left part is being rotated to face back in the previous step, then if you call "
            "agent.get_part_bbox(part_name='left'), it will return the latest left which corresponds to the previous front.\n"

            "    - agent.grasp_obj(part_grasp=<part>): Used to grasp the object by one of its parts. Direct "
            "invoking without arguments, i.e. agent.grasp_obj(), indicates a general grasp with any parts. Example: "
            "agent.grasp_obj(part_grasp='handle') means grasp the object by its handle.\n"

            "    - agent.rotate_obj(part_rotate=<part>, dir_rotate=<dir>): Used to rotate the object such that "
            "one of its part is facing a given direction. Both arguments required. <dir> can be one of these: "
            "'top', 'bottom', 'left', 'right', 'front', 'back'. Example: agent.rotate_obj(part_rotate='left', "
            "dir_rotate='top') means robot the left part of the object to face upwards.\n"

            "    - agent.touch_obj(part_touch=<part>): Used to touch the object at one of its parts. Directly "
            "invoking without arguments, i.e. agent.touch_obj(), indicates a general touch with any parts. Example: "
            "agent.touch_obj(part_touch='top') means touch the object's top part'.\n"

            "    - agent.move_gripper(dir_move=<dir>, distance=<float>): Used to move the gripper along a "
            "direction for a certain distance. dir_move is required, if not provided distance, a default value will "
            "be used, which is sufficient in most cases, unless you need to fine-tune the robot motion yourself. "
            "Example: agent.move_gripper(dir_move='right') means move the gripper towards right by a default distance. "
            "agent.move_gripper(dir_move='top', distance=0.01) means move the gripper upwards by 0.01 meter.\n"

            "    - agent.release_obj(): Used to release the object that being either grasped or touched, and move away "
            "a distance from the object. Can be treated as a mild version of reset. Example: agent.release_obj().\n\n"

            "Here are some examples of allocating optimal next actions to the robot in a given task:\n"

            "Example 1:\n"
            "Task Description: Touch the cap of the bottle and move it backwards.\n"
            "Action History: []\n"
            "Robot Tool Center Point Pose: {{'rotation': [0.265, 0.959, -0.0868, -0.0455], 'translation': [0.339, 0.567, 0.348] }}\n"
            "Robot Gripper State: 0.039\n"
            "Object State: Object State: {{'rotation': [-0.500, 0.500, 0.500, -0.500], 'translation': [0.0610, -0.0379, 0.127] }}\n"
            "Object Available Parts: ['body', 'cap', 'front', 'back', 'top', 'bottom', 'left', 'right']\n\n"
            "Thought: Based on the empty action history, the task is just started, I'll send out the first optimal action. The task " \
            "description says to 'touch' the cap 'and' 'move', I can first assign a touch action to the robot by using the " \
            "provided API function agent.touch_obj(part_touch=<part>).\n\n"
            "Action: agent.touch_obj(part_touch='cap')\n\n"

            "Example 2:\n"
            "Task Description: Hold the left of the mug and move it upwards.\n"
            "Action History: ['Grasp mug by part: left.']\n"
            "Robot Tool Center Point Pose: {{'rotation': [0.578, -0.567, 0.437, 0.390], 'translation': [-0.00388, 0.0310, 0.150]}}\n"
            "Robot Gripper State: 0.023\n"
            "Object State: Object State: {{'rotation': [-0.509, 0.518, 0.485, -0.485], 'translation': [-0.000344, -0.00928, 0.146]}}\n"
            "Object Available Parts: ['body', 'handle', 'front', 'back', 'top', 'bottom', 'left', 'right']\n\n"
            "Thought: The task description says to 'hold' the handle 'and' 'move'. Based on the action history, also the fact that "
            "gripper state is between 0 and 0.04, the handle has already been grasped, I can assign a move action to the robot by "
            "using the provided API function agent.move_gripper(dir_move=<dir>, distance=<float>) to finish the task. Use dir_move='top' "
            "to indicate an upward motion. Since I don't need to fine-tune the robot position, I'll use the default distance.\n\n"
            "Action: agent.move_gripper(dir_move='top')\n\n"

            "Example 3:\n"
            "Task Description: Lift the knife by its top and move front, then reorient its top to face right.\n"
            "Action History: []\n"
            "Robot Tool Center Point Pose: {{'rotation': [0.265, 0.959, -0.0868, -0.0455], 'translation': [0.339, 0.567, 0.348] }}\n"
            "Robot Gripper State: 0.039\n"
            "Object State: Object State: {{'rotation': [-0.500, 0.500, 0.500, -0.500], 'translation': [0.0530, -0.0409, 0.219] }}\n"
            "Object Available Parts: ['body', 'blade', 'front', 'back', 'top', 'bottom', 'left', 'right']\n\n"
            "Thought: The task description says to 'lift' the top part 'and' 'move', then 'reorient'. Based on the action history, "
            "the task just started. I'll send out the first optimal action. 'Lift' means move upwards but implies a grasp has been performed. " \
            "I can assign a grasp action to the robot first by using the API function agent.grasp_obj(part_grasp=<part>).\n\n"
            "Action: agent.grasp_obj(part_grasp='top')\n\n"

            "Example 4:\n"
            "Task Description: Lift the kettle by its handle and move back, then reorient its front to face right.\n"
            "Action History: ['Grasp kettle by part: handle.', 'Move gripper with dir_move='top', distance=0.12.', "
            "'Move gripper with dir_move='back', distance=0.12.']\n"
            "Robot Tool Center Point Pose: {{'rotation': [0.578, -0.567, 0.436, 0.390], 'translation': [-0.00390, 0.0304, 0.150] }}\n"
            "Robot Gripper State: 0.022\n"
            "Object State: Object State: {{'rotation': [-0.505, 0.518, 0.487, -0.487], 'translation': [5.66e-05, -0.00900, 0.146] }}\n"
            "Object Available Parts: ['body', 'lid', 'handle', 'mouth', 'front', 'back', 'top', 'bottom', 'left', 'right']\n\n"
            "Thought: The task description says to 'lift' the handle 'and' 'move', then 'reorient'. Based on the action history, "
            "the kettle has been picked up and moved accordingly. I'll send out the next action to reorient it. " \
            "I can assign a rotate action to the robot by using the API function agent.rotate_obj(part_rotate=<part>, dir_rotate=<dir>).\n\n"
            "Action: agent.rotate_obj(part_rotate='front', dir_rotate='right')\n\n"

            "Now the task begins.\n"
            "The current state is given below:\n\n"
            "Task Description: {task_description}\n"
            "Action History: {action_history}\n"
            "Robot Tool Center Point Pose: {robot_tcp_pose}\n"
            "Robot Gripper State: {robot_gripper_state}\n"
            "Object State: {object_state}\n"
            "Object Available Parts: {available_parts}\n"
            "Based on the above, output a single line of Python code that calls the appropriate method on the robot "
            "agent API to perform the next action, meaning assign only 1 action to the robot at a time. "
            "Use only available API functions. Return only the code and nothing else.\n"
            # TODO may integrate SAM
        )
    def get_state_info(self):
        """
        Extract robot and object state info from the environment.
        """
        object_state = self.agent.env.obj.get_pose().to_dict()
        robot_tcp_pose = self.agent.env.robot.get_tcp_pose().to_dict()
        robot_gripper_state = self.agent.env.robot.get_gripper_state()
        available_parts = [name for name in list(self.agent.env.parser.part_pcds_t0.keys()) if (not name==self.agent.env.obj_class) and (not name=='middle')]
        return robot_tcp_pose, robot_gripper_state, object_state, available_parts

    def decide_next_action(self, task_description: str) -> str:
        """
        Constructs the prompt, calls the LLM, and returns the generated Python code.
        """
        # Extract state information from the environment.
        robot_tcp_pose, robot_gripper_state, object_state, available_parts = self.get_state_info()
        prompt = self.prompt_template.format(
            robot_tcp_pose=robot_tcp_pose,
            robot_gripper_state=robot_gripper_state,
            object_state=object_state,
            task_description=task_description,
            available_parts=available_parts,
            gripper_max_width=self.agent.env.robot.max_opening_width,
            finger_depth=self.agent.env.robot.finger_depth,
            action_history=self.agent.action_history,
            gripper_open=self.agent.env.robot.GRIPPER_OPEN_JOINT_POS,
            gripper_closed=self.agent.env.robot.GRIPPER_CLOSED_JOINT_POS,
        )
        print("Sending prompt to LLM:\n", prompt)

        user_content = [
            {
                "type": "text",
                "text": prompt
            },
            # {
            #     "type": "image_url",
            #     "image_url": {
            #         "url": f"data:image/jpeg;base64,{image}"
            #     }
            # }
        ]

        # Build the payload.
        payload = {
            "model": "gpt-4o-2024-11-20",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a robotics expert."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        
        # Send the request with retry logic.
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response_json = response.json()
                # Check if the response contains the required fields.
                if 'choices' not in response_json or not response_json['choices']:
                    print(f"Unexpected response: {response_json}")
                    raise ValueError("Error: No choices in the response")
                action_code = response_json['choices'][0]['message']['content'].strip()
                return action_code
            except Exception as e:
                print(f"Attempt {attempt+1} failed with error: {e}")
                if attempt < retries - 1:
                    print("Retrying...")
                else:
                    raise

    def execute_next_action(self, task_description: str):
        """
        Generates the next action code using the LLM and safely executes it.
        """
        action_code = self.decide_next_action(task_description)
        print("Generated action code:", action_code)

        # Remove Markdown code fences if present.
        action_code = action_code.strip()
        if action_code.startswith("```"):
            lines = action_code.splitlines()
            # Remove the first line if it starts with triple backticks.
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove the last line if it ends with triple backticks.
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            action_code = "\n".join(lines)
        
        print("Cleaned action code:", action_code)
        
        # Use a dedicated local dictionary for safe exec.
        local_vars = {}
        # CAUTION: Always validate and sanitize generated code.
        try:
            exec_safe(action_code, globals(), local_vars)
        except AssertionError as ae:
            print("Security check failed:", ae)
        except Exception as e:
            print("Error during execution:", e)

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
        for task_type in list(episode_data[obj_class][split].keys()):
            if args.task_types and task_type not in args.task_types:
                continue

            for i in range(args.num_envs):
                # Env setups
                env = BulletEnv(
                    config_path=config_path, gui=False, record=True, evaluation=True, skill_mode=False,
                    obj_class=obj_class, split=split, task_type=task_type,
                )
                env.reset()
                planner = BulletPlanner(env, generation_mode=True)

                # Instantiate the robot agent.
                agent = RobotAgent(env, planner)

                # Instantiate the LLM policy.
                policy = LLMPolicy(agent, openai_api_key)

                # Define a task description.
                task_description = env.task_instruction

                # Execute the next action as determined by the LLM policy.
                done = False
                time_out = False
                while not done:
                    if len(agent.action_history)>len(env.chain_params)*args.time_out_multiplier:
                        time_out = True
                        break
                    policy.execute_next_action(task_description)
                    # agent.grasp_obj(part_grasp='left')
                    done = env._check_if_done_test_eval()
                print("Final state:")
                print(policy.get_state_info())
                # Save
                ep_dir = os.path.join(output_dir, f"{split}_{task_type}_{obj_class}_{i}")
                os.makedirs(ep_dir, exist_ok=True)

                video_path = os.path.join(ep_dir, "video.mp4")
                state_path = os.path.join(ep_dir, "states.json")
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