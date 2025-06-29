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

from PartInstruct.PartGym.env.backend.planner.panda_arm import PandaArm
from PartInstruct.PartGym.env.backend.utils.semantic_parser import SemanticParser, mapping_rotate, mapping_translate
from PartInstruct.PartGym.env.backend.utils.language_encoder import T5Encoder
import PartInstruct.PartGym.env.backend.bullet_sim as bullet_sim
from PartInstruct.PartGym.env.backend.bullet_sim import save_image, save_depth
from PartInstruct.PartGym.env.backend.utils.vision_utils import *
from PartInstruct.PartGym.env.backend.utils.transform import *
from PartInstruct.PartGym.env.backend.utils.perception import *
from PartInstruct.PartGym.env.backend.utils.scene_utils import *
from PartInstruct.PartGym.env.backend.utils.vision_utils import *

from PartInstruct.PartGym.env.bullet_env import BulletEnv
from PartInstruct.PartGym.env.backend.utils.sam_utils import *
from sam2.build_sam import build_sam2_camera_predictor

RESAMPLE_SPATIAL_SKILLS = ["8", "10", "12", "13", "14", "15", "16", "17"]

class BulletEnv_SAM(BulletEnv):
    """
    BulletEnvSam extends BulletEnv to add SAM (Segment Anything Model) functionality.
    This class inherits all the base functionality from BulletEnv and adds SAM-specific features.
    """

    def __init__(self, config_path=None, config=None, shape_meta=None, gui=False, obj_class=None, random_sample=False, 
                 evaluation=False, split='val', task_type=None, record=False, check_tasks=True, track_samples=False, 
                 replica_scene=False, skill_mode=True, debug_output=None):
        super(BulletEnv_SAM, self).__init__(config_path, config, shape_meta, gui, obj_class, random_sample, 
                                          evaluation, split, task_type, record, track_samples, 
                                          replica_scene, skill_mode, debug_output)
        
        self.sam2_video_predictor = sam2_video_predictor
        self.grounding_success = False
        self.grounding_success_num = 0
        self.iou = []
        self.accuracy = []
        self.check_tasks = check_tasks

    def set_sam(self, image, obj, part):
        """
        Use SAM to segment a part of an object in an image.
        
        Args:
            image: The image to segment
            obj: The object name
            part: The part name to segment
            
        Returns:
            A tuple (success, mask) where success is a boolean and mask is the segmentation mask
        """
        scene_text_input = obj
        part_text_input = f"The {part} of {obj}"
        self.sam2_video_predictor.load_first_frame(image)
        image = Image.fromarray(image)
        sampled_points = phrase_grounding_and_segmentation(
                            image=image,
                            scene_text_input=scene_text_input,
                            part_text_input=part_text_input
                        )
        if not sampled_points:
            return False
        print("sampled_points", sampled_points)
        sampled_points = [(index, np.array([y, x])) for index, (x, y) in sampled_points]
        points = np.array([coord for _, coord in sampled_points], dtype=np.float32)
        if points.shape[0] != 2:
            return False
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1,0], dtype=np.int32)
        
        ann_obj_id = (1)
        _, out_obj_ids, out_mask_logits = self.sam2_video_predictor.add_new_prompt(
            frame_idx=0,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        show_points(points, labels, plt.gca())
        mask_np = show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

        return True, mask_np

    def _get_observation(self):
        
        renders = self.render_all()
        frame_input = renders["agentview"]["rgb"]
        agentview_rgb = frame_input.transpose((2, 0, 1))
        
        if self.num_steps == 0:
            if self.cur_target_part:
                result = self.set_sam(frame_input, self.obj_class, self.cur_target_part)
                if isinstance(result, tuple):
                    self.grounding_success, mask_np = result
                else:
                    self.grounding_success = result
                    mask_np = None
                if self.grounding_success:
                    self.grounding_success_num += 1
        else:
            if self.grounding_success and self.cur_target_part:
                out_obj_ids, out_mask_logits = self.sam2_video_predictor.track(frame_input)
                frame = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
                # plt.figure(figsize=(12, 8))
                # plt.imshow(frame)
                mask_np = show_mask(
                    (out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0]
                )
                if self.debug_output:
                    color_image = visualize_segmentation_mask(mask_np)
                    cv2.imwrite(f"{self.debug_output}/part_mask_frame_{self.num_steps}.png", color_image)
                plt.close()
            

        # Get the current TCP position and orientation
        tcp_pose = self.robot.get_tcp_pose()
        tcp_position = tcp_pose.translation
        tcp_orientation = tcp_pose.rotation.as_quat()
        tcp_position = np.array(tcp_position)  # Convert translation to numpy array
        tcp_orientation = np.array(tcp_orientation)  # Convert quaternion to numpy array

        # Combine the position and orientation into a single array
        tcp_pose_combined = np.concatenate((tcp_orientation, tcp_position))

        # Get the current joint states
        joint_states = self.robot.get_joint_states()
        gripper_state = self.robot.get_gripper_state()

        # Construct the observation dictionary
        observation = {
            "agentview_rgb": agentview_rgb,
            "tcp_pose": tcp_pose_combined,
            "joint_states": np.array(joint_states),
            "gripper_state": np.array(gripper_state).reshape(1,)
        }
        if self.use_pcd:
            agentview_pcd = renders["agentview"]["pcd"]
            print("agentview_pcd", agentview_pcd.shape)
            observation["agentview_pcd"] = agentview_pcd
        if self.use_part_pcd_gt:
            agentview_part_pcd = renders["agentview"]["part_pcd"]
            observation["agentview_part_pcd"] = agentview_part_pcd
        if self.use_part_mask_gt:
            gt_mask = renders["agentview"]["part_mask"]
            observation["agentview_part_mask"] = np.zeros_like(gt_mask)
            if self.grounding_success and self.cur_target_part:
                if self.debug_output:
                    color_image = visualize_segmentation_mask(gt_mask)
                    cv2.imwrite(f"{self.debug_output}/dt_part_mask_frame_{self.num_steps}.png", color_image)
                iou, accuracy = calculate_iou(gt_mask, mask_np)
                if iou > 0:
                    observation["agentview_part_mask"] = mask_np
                self.iou.append(iou)
                self.accuracy.append(accuracy)

        # Get instruction
        if self.use_language:
            assert self.tokenized_instruction
            observation.update(self.tokenized_instruction)

        # Get wrist camera
        if self.use_wrist_camera:
            wrist_rgb = renders["wrist"]["rgb"]
            wrist_rgb = wrist_rgb.transpose((2, 0, 1))
            observation["wrist_rgb"] = wrist_rgb
            if self.use_pcd:
                wrist_pcd = renders["wrist"]["pcd"]
                observation["wrist_pcd"] = wrist_pcd
            if self.use_part_pcd_gt:
                observation["wrist_part_pcd"] = renders["wrist"]["part_pcd"]
            if self.use_part_mask_gt:
                observation["wrist_part_mask"] = renders["wrist"]["part_mask"]

        return observation

    def step(self, action, gain=0.01, gain_gripper=0.01, gain_near_target=0.01):
        # Execute one time step within the environment
        self.action_list.append(action)
        self._take_action(action, gain, gain_gripper, gain_near_target)
        self.num_steps += 1
        observation = self._get_observation()
        self.last_state = self.current_state
        self.current_state = copy.deepcopy(observation)
        position, orientation = self.world.get_body_pose(self.obj)
        self.current_state["obj_pose"] = Transform(Rotation.from_quat(list(orientation)), list(position))

        tcp_pose = self.robot.get_tcp_pose()
        tcp_position = tcp_pose.translation
        tcp_orientation = tcp_pose.rotation.as_quat()
        self.current_state["tcp_pose"] = Transform(Rotation.from_quat(list(tcp_orientation)), list(tcp_position))

        reward = 1.0  # Dummy reward

        resample_spatial = False

        # For Bi-level framework, the skill_mode must be True
        done_cur_skill, done = self._check_if_done_skill()
        self.done_cur_skill = done_cur_skill
        cur_skill = copy.deepcopy(self.cur_skill)
        if done_cur_skill:
            self.cur_skill_idx += 1
            if not done:
                self.cur_skill = self.chain_params[self.cur_skill_idx]["skill_name"]
                self.cur_skill_params = self.chain_params[self.cur_skill_idx]["params"]
                if not self.skill_chain in RESAMPLE_SPATIAL_SKILLS:
                    resample_spatial = True
                self.semantic_grounding(resample_spatial)
                
                if self.cur_target_part:
                    frame_input = observation['agentview_rgb'].transpose((1, 2, 0))
                    result = self.set_sam(frame_input, self.obj_class, self.cur_target_part)
                    if isinstance(result, tuple):
                        self.grounding_success, mask_np = result
                    else:
                        self.grounding_success = result
                        mask_np = None
                    if self.grounding_success:
                        self.grounding_success_num += 1
                    print("Grounding success", self.grounding_success)

        info = {
            "Success": done,
            "Completion Rate": self.completion_rate,
            "Steps": self.num_steps,
            "Object Pose": list(self.current_state["obj_pose"].to_list()),
            "TCP Pose": list(self.current_state["tcp_pose"].to_list()),
            "Joint States": list(self.current_state["joint_states"]),
            "Gripper State": list(self.current_state["gripper_state"]),
            "Action": list(action),
            "chain_params": self.chain_params,
            "Grounding Success": (self.grounding_success_num) / len(self.chain_params),
            "iou": self.iou,
            "accuracy": self.accuracy
        }
        if self.skill_mode:
            info.update({
                "Current Skill": cur_skill,
                "Current Skill Success": done_cur_skill,
                "Completion Rate": self.cur_skill_idx/len(self.chain_params),
            })
        
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
        self.info = info

        if self.record:
            self.state_sequence_buffer.append(info)

        return observation, reward, done, info