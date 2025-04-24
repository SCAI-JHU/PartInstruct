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


lang_encoder = T5Encoder()

class BulletEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 20}

    def __init__(self, config_path="lgm_bc/config/config.yaml", gui=False, obj_class=None, random_sample = False, 
                 evaluation = False, split='val', task_type=None, record=False, track_samples=False, 
                 replica_scene=False, skill_mode=True):

        super(BulletEnv, self).__init__()
        
        self._seed = None
        self.seed()
        
        self.skill_mode = skill_mode
        self.track_samples = track_samples
        self.replica_scene = replica_scene
        self.gui = gui
        self.record = record
        self.sim_hz = 240
        self.control_hz = self.metadata['video.frames_per_second']
        self.world = bullet_sim.BtWorld(gui, self.control_hz)
        self.parser = SemanticParser(self.world)
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.evaluation = evaluation 
        self.random_sample = random_sample 
        self.render_mode = "rgb_array"
        self.renders = {}
        self.render_sequence = []
        self.state_sequence = []
        self.render_sequence_buffer = []
        self.state_sequence_buffer = []
        self.tokenized_instruction = {}
        self.instruction = ''
        self.task_instruction = ''
        self.skill_instructions = None

        self.split = split
        self.obj_class = obj_class
        self.task_type = task_type
        self.cur_target_part=None
        
        # Initialize task-related params
        self.obj: bullet_sim.Body = None
        self.info = {}
        self.table = None
        self.scene = None
        self.obj_scale = 0.1
        self.target_grasp_bbox = None
        self.target_translation = None
        self.target_rotation = None
        self.target_touch_bbox = None
        self.release_new_tcp_pose = None
        self.last_state = None
        self.current_state = None
        self.chain_params = []
        self.skill_chain = None
        if self.skill_mode:
            self.cur_skill = None
            self.cur_skill_idx = None
            self.done_cur_skill = False
            self.cur_skill_params = {}
        self.pre_move_tcp_pose = None
        self.grasp_pose = None
        self.num_steps = 0
        self.previous_steps = 0
        self.completion_rate = 0

        self.use_wrist_camera = False
        self.use_language = False
        self.use_pcd = False
        self.use_part_mask_gt = False
        self.use_part_pcd_gt = False
        self.pcd_size = 1024

        self.action_list = []

        # Configuration
        self.config = OmegaConf.load(config_path)
        self.device = self.config.device

        intrinsic = CameraIntrinsic(self.config.render_width, self.config.render_height, 
                                    self.config.cam_static_intrinsics.fx, self.config.cam_static_intrinsics.fy, 
                                    self.config.cam_static_intrinsics.cx, self.config.cam_static_intrinsics.cy)
        intrinsic.K = bullet_sim._build_intrinsic_from_fov(self.config.render_height, self.config.render_width, fov=60)
        # print("Intrinsic matrix:", intrinsic.K)
        self.record_camera = self.world.add_camera(intrinsic, self.config.cam_near, self.config.cam_far)
        extrinsic = calculate_camera_extrinsics(self.config.cam_static_target, self.config.cam_static_dist, self.config.cam_static_yaw, self.config.cam_static_pitch)
        # extrinsic[2, :]*= -1
        extrinsic = Transform.from_matrix(extrinsic)
        self.record_camera.set_extrinsic(extrinsic)
        self.robot_base_position = self.config.robot_base_position
        self.obj_init_position = self.config.obj_position

        self.data_root = self.config.data_root
        self.dataset_meta_path = os.path.join(self.data_root, self.config.meta_path)
        
        self.urdf_robot = os.path.join(self.data_root, self.config.urdf_robot)
        self.urdf_table = os.path.join(self.data_root, self.config.urdf_table)
        self.urdf_floor = os.path.join(self.data_root, self.config.urdf_floor)
        self.partnet_path = os.path.join(self.data_root, self.config.partnet_path)
        self.objects_directory = os.path.join(self.data_root, self.config.objects_directory) if self.config.objects_directory else None

        self.scene_path = os.path.join(self.data_root, self.config.scene_path)
        self.scene_config_path = os.path.join(self.data_root, self.config.scene_config_path) if self.config.scene_config_path else None

        self.action_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, 0.0]), 
            high=np.array([np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, 1.0]),      
            dtype=np.float32  
        )

        obs_spaces = {}
        for obs_key, obs_meta in self.config.shape_meta['obs'].items():
            if "wrist" in obs_key:
                self.use_wrist_camera = True
            if "instructions" in obs_key:
                self.use_language = True
            if "pcd" in obs_key:
                self.use_pcd = True
                self.pcd_size = obs_meta["shape"][1]
            if "part_mask" in obs_key:
                self.use_part_mask_gt = True
            if "part_pcd" in obs_key:
                self.use_part_pcd_gt = True
            
            shape = tuple(obs_meta['shape'])
            if obs_meta['type'] == 'rgb':
                obs_spaces[obs_key] = spaces.Box(
                    low=0,
                    high=255,
                    shape=shape,
                    dtype=np.uint8
                )
            if obs_meta['type'] == 'mask':
                obs_spaces[obs_key] = spaces.Box(
                    low=0,
                    high=255,
                    shape=shape,
                    dtype=np.uint8
                )
            if obs_meta['type'] == 'low_dim':
                if obs_key == "gripper_state":
                    obs_spaces[obs_key] = spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=shape,
                        dtype=np.float32
                    )
                if obs_key == "tcp_pose":
                    obs_spaces[obs_key] = spaces.Box(
                        low=np.array([-np.inf, -np.inf, -np.inf, -1.0, -1.0, -1.0, -1.0]), 
                        high=np.array([np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, 1.0]),      
                        dtype=np.float32  
                    )
                if obs_key == "joint_states":
                    obs_spaces[obs_key] = spaces.Box(
                        low=-2*np.pi,
                        high=2*np.pi,
                        shape=shape,
                        dtype=np.float32
                    )
            if obs_meta['type'] == 'text':
                obs_spaces["input_ids"] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=shape,
                    dtype=np.int64
                )
                obs_spaces["attention_mask"] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=shape,
                    dtype=np.int64
                )
            if obs_meta['type'] == 'pcd':
                obs_spaces[obs_key] = spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=shape,
                    dtype=np.float32
                )
        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self, obj_class=None, obj_id=None, skill_chain=None, chain_params=None, 
              obj_scale=0.1, obj_position=[0.0, -0.031480762319536344, 0.15], 
              obj_orientation=list(Rotation.identity().as_quat())):

        # Ensure the dataset metadata path exists
        assert os.path.exists(self.dataset_meta_path), "Dataset metadata path does not exist."
        
        # Load dataset metadata
        with open(self.dataset_meta_path, 'r') as file:
            dataset_meta = json.load(file)

        # Handle random sampling mode
        if self.random_sample and not self.evaluation:
            print("Use sampling mode")
            self.obj_class = self.np_random.choice(list(dataset_meta.keys()))
            print("Sampling from object class:", self.obj_class)

            # Extract skill chains
            skill_chain_list = [task_name for task_name, task_values in dataset_meta[self.obj_class][self.split].items() if task_values]
            self.skill_chain = self.np_random.choice(skill_chain_list)
            self.sample_chain_params()  # Get parameters for the selected object

        # Handle evaluation mode
        elif self.evaluation:
            assert all(param is not None for param in [self.task_type, self.split, self.obj_class]), "Task type, split, and object class must be defined."
            print("Use evaluation mode")
            print("Sampling from object class:", self.obj_class)
            self.skill_chain = str(self.task_type)
            print("Sampling from split:", self.split)
            print("Sampling from task:", self.task_type)
            self.sample_chain_params()  # Get parameters for the selected object

        # Handle custom parameters
        elif chain_params:
            self.obj_class, self.obj_id, self.obj_scale = obj_class, obj_id, obj_scale
            self.obj_init_position, self.obj_init_orientation = obj_position, obj_orientation
            self.chain_params, self.skill_chain = chain_params, skill_chain

        # Set object paths
        self.obj_path = os.path.join(self.partnet_path, self.obj_class, self.obj_id)
        self.urdf_obj = Path(os.path.join(self.data_root, self.obj_path, "mobility.urdf"))

        # Determine if two phases are needed
        self.two_phases = False

        if self.skill_chain in ["5", "6", "9", "11", "12", "13", "14", "16"]:
            self.two_phases = True
            self.task_phase = 0
        else:
            self.task_phase = None

        self.num_steps = 0

        # Initialize the PyBullet simulator
        self.world.reset()
        self.table, self.floor = load_grasping_setups(self.world, self.urdf_floor, self.urdf_table)
        self.parser = SemanticParser(self.world)
        self.robot = PandaArm(self.world, self.robot_base_position, urdf_path=self.urdf_robot)
        self.world.set_gravity([0.0, 0.0, -9.81])
        
        # Initialize rendering and state sequences
        self.renders = {}
        self.render_sequence = []
        self.state_sequence = []
        self.render_sequence_buffer = []
        self.state_sequence_buffer = []

        # Configure GUI if enabled
        if self.gui:
            self.world.p.configureDebugVisualizer(self.world.p.COV_ENABLE_GUI, 0)
            self.world.p.configureDebugVisualizer(self.world.p.COV_ENABLE_SHADOWS, 1)
            self.world.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=self.config.cam_static_dist,
                cameraYaw=self.config.cam_static_yaw,
                cameraPitch=self.config.cam_static_pitch,
                cameraTargetPosition=self.config.cam_static_target,
            )

        # Reset the robot and load the object
        self.robot.reset()
        self.obj = load_grasping_object(self.world, self.urdf_obj, 
                                        obj_position=self.obj_init_position, 
                                        obj_orientation=self.obj_init_orientation, 
                                        obj_scale=self.obj_scale)

        # Load the replica scene if applicable
        if self.replica_scene:
            self.scene = load_replica_cad_scene(self.world, 
                                                 scene_path=self.scene_path, 
                                                 scene_config_path=self.scene_config_path, 
                                                 objects_directory=self.objects_directory)

        # Set the object in the parser and load the part hierarchy
        self.parser.set_obj(self.obj)
        self.parser.load_part_hierarchy(self.obj_path, scale=self.obj_scale, num_points=5000)

        # Initialize poses
        self.initial_tcp_pose = self.robot.get_tcp_pose()
        self.initial_obj_pose = self.obj.get_pose()
        self.pre_rotate_pose = self.obj.get_pose()

        # Initialize skill mode parameters
        if self.skill_mode:
            self.cur_skill_idx = 0
            self.cur_skill = self.chain_params[self.cur_skill_idx]["skill_name"]
            self.cur_skill_params = self.chain_params[self.cur_skill_idx]["params"]
            self.cur_target_part = safe_get([self.cur_skill_params[key] for key in self.cur_skill_params.keys() if "part" in key], 0)
            self.state_copy = {
                "cur_skill_idx": self.cur_skill_idx, 
                "cur_skill": self.cur_skill, 
                "cur_skill_params": self.cur_skill_params
            }

        # Perform semantic grounding and prepare the observation
        self.semantic_grounding(resample_spatial=True)
        observation = self._get_observation()
        self.current_state = copy.deepcopy(observation)
        
        # Update object and TCP poses in the current state
        position, orientation = self.world.get_body_pose(self.obj)
        self.current_state["obj_pose"] = Transform(Rotation.from_quat(list(orientation)), list(position))
        
        tcp_pose = self.robot.get_tcp_pose()
        tcp_position = tcp_pose.translation
        tcp_orientation = tcp_pose.rotation.as_quat()
        self.current_state["tcp_pose"] = Transform(Rotation.from_quat(list(tcp_orientation)), list(tcp_position))
        
        self.info = {}
        
        return observation

    def semantic_grounding(self, resample_spatial=True):

        self.parser.update_part_pcds(resample_spatial)

        if self.use_language:
            self.instruction, self.tokenized_instruction = self._get_instruction()

    def step(self, action, gain=0.01, gain_gripper=0.01, gain_near_target=0.01):
        # Execute one time step within the environment
        self.action_list.append(action)
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
        reward = 1.0 # Dummy reward

        resample_spatial = False

        if self.skill_mode:
            done_cur_skill, done = self._check_if_done()
            self.done_cur_skill = done_cur_skill
            cur_skill = copy.deepcopy(self.cur_skill)
            if done_cur_skill:
                self.cur_skill_idx += 1
                if not done:
                    self.cur_skill = self.chain_params[self.cur_skill_idx]["skill_name"]
                    self.cur_skill_params = self.chain_params[self.cur_skill_idx]["params"]
                    self.cur_target_part = safe_get([self.cur_skill_params[key] for key in self.cur_skill_params.keys() if "part" in key], 0)
                    # DEBUG
                    print("check cur_target_part", self.cur_target_part)
                    resample_spatial = True
                    self.initial_obj_pose = self.obj.get_pose()
        else:
            done = self._check_if_done_test_eval()
            if not self.skill_chain in ["8", "10", "12", "13", "14", "15", "16", "17"]:
                resample_spatial = True
        
        self.semantic_grounding(resample_spatial)
        
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
            # self.state_sequence.append(info)

        return observation, reward, done, info

    def planning_step(self, action, mode="direct", max_iteration=100):

        # "direct" no motion planner (tiny motions);
        # "free" for collision-aware motions without grasping; 
        # "holding" for collision-aware motions with grasping

        assert mode in ["direct", "free", "holding"]
        from PartInstruct.PartGym.env.backend.planner.bullet_planner import BulletPlanner, WorldSaver

        saved_world = WorldSaver()
        planning_fn = getattr(BulletPlanner, f"plan_{mode}_motion_fn")(self.robot.panda, [self.floor.uid])
        q_cur = self.robot.get_joint_states(return_all=True)
        
        x, y, z, roll, pitch, yaw = action[:6]
        trgt_pose = Transform(Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False), list([x, y, z]))

        grasp_pose = self.robot.get_tcp_pose()

        all_obs, all_reward, all_done, all_info = [], [], [], []

        try:
            path = planning_fn(self.world, self.obj.uid, q_cur, trgt_pose, grasp_pose=grasp_pose)
        except:
            print("Planning failed!")
            return all_obs, all_reward, all_done, all_info

        saved_world.restore()
        self.robot.set_joint_states(q_cur)

        for j, joint_states in enumerate(path):
            for i in range(self.robot.NUM_DOFS):
                self.world.p.setJointMotorControl2(self.robot.panda, i, self.world.p.POSITION_CONTROL, joint_states[i], 
                                                        positionGain=0.03, force=10 * 240.)
            # set gripper control with a fixed gain
            for i in [9, 10]:
                self.world.p.setJointMotorControl2(self.robot.panda, i, self.world.p.POSITION_CONTROL, joint_states[-1], 
                                                        positionGain=0.001, force=5000)
            q_cur = self.robot.get_joint_states(return_all=True)
            q_pre = np.array([np.inf]*len(q_cur))
            this_it = 0
            while not (np.allclose(np.array(q_cur), np.array(joint_states), rtol=1e-03, atol=1e-02) or \
                    np.allclose(np.array(q_cur), np.array(q_pre), rtol=1e-06, atol=1e-05) or \
                    this_it>max_iteration):
                q_pre = q_cur
                q_cur = self.robot.get_joint_states(return_all=True)
                self.world.step()
                this_it+=1
            
            if j==len(path)-1:
                q_cur = self.robot.get_joint_states(return_all=True)
                q_pre = np.array([np.inf]*len(q_cur))
                this_it = 0
                while not (np.allclose(np.array(q_cur), np.array(joint_states), rtol=1e-03, atol=1e-02) or \
                    this_it>max_iteration):
                    q_pre = q_cur
                    q_cur = self.robot.get_joint_states(return_all=True)
                    self.world.step()
                    this_it+=1
            
            # Execute one time step within the environment
            self.action_list.append(action)
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

            reward = 1.0 # Dummy reward
            resample_spatial = False

            if self.skill_mode:
                done_cur_skill, done = self._check_if_done()
                self.done_cur_skill = done_cur_skill
                cur_skill = copy.deepcopy(self.cur_skill)
                if done_cur_skill:
                    self.cur_skill_idx+=1
                    if not done:
                        self.cur_skill = self.chain_params[self.cur_skill_idx]["skill_name"]
                        self.cur_skill_params = self.chain_params[self.cur_skill_idx]["params"]
                        self.cur_target_part = safe_get([self.cur_skill_params[key] for key in self.cur_skill_params.keys() if "part" in key], 0)
                        # DEBUG
                        print("check cur_target_part", self.cur_target_part)
                        resample_spatial = True
                        self.initial_obj_pose = self.obj.get_pose()

            else:
                done = self._check_if_done_test_eval()
                if not self.skill_chain in ["8", "10", "12", "13", "14", "15", "16", "17"]:
                    resample_spatial = True
            
            self.semantic_grounding(resample_spatial)
            
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
            
            all_obs.append(observation)
            all_reward.append(reward)
            all_done.append(done)
            all_info.append(info)
            
            if (self.skill_mode and done_cur_skill) or done:
                break
        
        return all_obs, all_reward, all_done, all_info

    def clear_buffers(self):
        self.render_sequence_buffer = []
        self.state_sequence_buffer = []
        self.num_steps = self.previous_steps
        if self.skill_mode:
            self.cur_skill_idx = self.state_copy["cur_skill_idx"]
            self.cur_skill = self.state_copy["cur_skill"]
            self.cur_skill_params = self.state_copy["cur_skill_params"]

    def dump_buffers(self):
        self.render_sequence+=self.render_sequence_buffer
        self.state_sequence+=self.state_sequence_buffer
        self.render_sequence_buffer = []
        self.state_sequence_buffer = []
        self.previous_steps = self.num_steps
        if self.skill_mode:
            self.state_copy = {"cur_skill_idx": self.cur_skill_idx, "cur_skill": self.cur_skill, "cur_skill_params": self.cur_skill_params}

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            return np.array([])
        view_matrix = self.world.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.config.cam_static_target,
                                                            distance=self.config.cam_static_dist,
                                                            yaw=self.config.cam_static_yaw,
                                                            pitch=self.config.cam_static_pitch,
                                                            roll=0,
                                                            upAxisIndex=self.config.up_axis_index)
        proj_matrix = self.world.p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.config.render_width) /
                                                        self.config.render_height,
                                                        nearVal=self.config.cam_near,
                                                        farVal=self.config.cam_far)
        (_, _, px, _, _) = self.world.p.getCameraImage(width=self.config.render_width,
                                                height=self.config.render_height,
                                                viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix,
                                                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        self.world.p.configureDebugVisualizer(self.world.p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(np.array(px), (self.config.render_height, self.config.render_width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def render_all(self):
        renders = {}
        # render agent view images
        agent_view_matrix = self.world.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.config.cam_static_target,
                                                            distance=self.config.cam_static_dist,
                                                            yaw=self.config.cam_static_yaw,
                                                            pitch=self.config.cam_static_pitch,
                                                            roll=0,
                                                            upAxisIndex=self.config.up_axis_index)
        agent_proj_matrix = self.world.p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.config.render_width) /
                                                        self.config.render_height,
                                                        nearVal=self.config.cam_near,
                                                        farVal=self.config.cam_far)
        result = self.world.p.getCameraImage(width=self.config.render_width,
                                                height=self.config.render_height,
                                                viewMatrix=agent_view_matrix,
                                                projectionMatrix=agent_proj_matrix,
                                                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(result[2]).reshape(result[1], result[0], -1)[:, :, :3]
        z_buffer = np.array(result[3]).reshape(result[1], result[0])
        segmentation_mask = np.array(result[4]).reshape(result[1], result[0])
        depth = (
            1.0 * self.config.cam_far * self.config.cam_near / (self.config.cam_far - (self.config.cam_far - self.config.cam_near) * z_buffer)
        )
        # ## DEBUG
        # cv2.imwrite("agentview_rgb.png", rgb)
        renders["agentview"] = {
            "rgb": rgb,
            "depth": depth.astype(np.float32),
            "mask": segmentation_mask
        }
        # ## DEBUG
        # color_image = visualize_segmentation_mask(segmentation_mask)
        # cv2.imwrite("instance_mask.png", color_image)
        # seg = renders["agentview"]["mask"]

        if self.use_pcd or self.use_part_pcd_gt:
            table_seg_mask = (renders["agentview"]["mask"] == 1) | (renders["agentview"]["mask"] == 0) | (renders["agentview"]["mask"] == -1)
            depth_filtered = apply_segmentation_mask(depth, ~table_seg_mask)
            valid_pixels = np.where(~table_seg_mask)
            pixels_y_filtered = valid_pixels[0]
            pixels_x_filtered = valid_pixels[1]
            original_indices = pixels_y_filtered * self.config.render_width + pixels_x_filtered
            raw_indices = original_indices
            
            point_cloud, indices = depth_to_pcd_farthest_points(depth_filtered, self.record_camera.intrinsic.K, downsample_size=self.pcd_size, agent_view=True, device=self.device)
            indices = indices.cpu().numpy()
            scene_pcd = point_cloud[indices].squeeze(0)
            renders["agentview"]["pcd"] = scene_pcd

        if self.use_part_pcd_gt or self.use_part_mask_gt:
            if not self.cur_target_part:
                part_pcd = np.zeros((self.pcd_size, 3))
            else:
                part_pcd = self.parser.get_one_part_pcd(self.cur_target_part) # world frame
            dsp_pcd = resample_pcd(part_pcd, self.pcd_size)
            agent_view_matrix_np = np.array(agent_view_matrix).reshape(4, 4, order="F")
            # Convert point cloud to homogeneous coordinates (add 1s to the end)
            pcd_homogeneous = np.hstack((dsp_pcd, np.ones((dsp_pcd.shape[0], 1))))
            # Apply view transformation
            dsp_pcd = pcd_homogeneous @ agent_view_matrix_np.T
            dsp_pcd = dsp_pcd[:, :3]
            dsp_pcd[:, 2] *= -1
            dsp_pcd[:, 1] *= -1

            if self.use_part_pcd_gt:
                # renders["agentview"]["part_pcd"] = dsp_pcd
                part_mask = project_point_cloud_to_mask(part_pcd, self.config.render_width, self.config.render_height, agent_view_matrix, agent_proj_matrix)
                pixels_y, pixels_x = np.where(part_mask == 1)
                pcd_indices = pixels_y * self.config.render_width + pixels_x
                intersection_elements = np.intersect1d(raw_indices, pcd_indices)
                indices_in_raw = np.searchsorted(raw_indices, intersection_elements)
                part_indices = np.intersect1d(indices_in_raw, indices)
                part_channel = np.expand_dims(np.zeros(indices.shape[1]), axis=1)
                boolean_mask = np.isin(indices, part_indices).transpose(1,0)
                part_channel[boolean_mask] = 1
                scene_pcd_with_part = np.hstack((scene_pcd, part_channel))
                # np.save("scene_pcd_filtered.npy", scene_pcd_with_part)
                # print("Vised !!")
                renders["agentview"]["part_pcd"] = scene_pcd_with_part

            if self.use_part_mask_gt:
                # depth_to_pcd() (pixel, pcd_index)
                part_mask = project_point_cloud_to_mask(part_pcd, self.config.render_width, self.config.render_height, agent_view_matrix, agent_proj_matrix)
                renders["agentview"]["part_mask"] = part_mask
                # ## DEBUG
                # color_image = visualize_segmentation_mask(part_mask)
                # cv2.imwrite("part_mask.png", color_image)
                
            # ## DEBUG
            # visualize_and_save_two_point_clouds(dsp_pcd, renders["agentview"]["pcd"], save_path="combined_pcd.ply", visualize=False)
            
        if self.use_wrist_camera:
            wrist_rgb, wrist_depth, wrist_segmentation = self.robot.wrist_camera.render(extrinsic=self.robot.wrist_camera.extrinsic, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
            renders["wrist"] = {
                "rgb": wrist_rgb,
                "depth": wrist_depth,
                "mask": wrist_segmentation
            }

            # Convert extrinsic to view matrix for wrist camera
            gl_view_matrix = self.robot.wrist_camera.extrinsic.as_matrix()
            gl_view_matrix[2, :] *= -1  # flip the Z axis
            wrist_view_matrix = gl_view_matrix.flatten(order="F")
            extrinsic = np.array(wrist_view_matrix).reshape(4, 4, order='F')
            extrinsic[2, :]*= -1
            wrist_proj_matrix = self.robot.wrist_camera.proj_matrix

            if self.use_pcd:
                wrist_pcd = depth_to_pcd_farthest_points(wrist_depth, self.robot.wrist_camera.intrinsic.K)
                renders["wrist"]["pcd"] = wrist_pcd

            if self.use_part_pcd_gt or self.use_part_mask_gt:
                if not self.cur_target_part:
                    wrist_part_pcd = np.zeros((self.pcd_size, 3))
                else:
                    wrist_part_pcd = self.parser.get_one_part_pcd(self.cur_target_part) # world frame
                wrist_dsp_pcd = resample_pcd(wrist_part_pcd, self.pcd_size)

                wrist_extrinsic = self.robot.wrist_camera.extrinsic.as_matrix()
                wrist_view_tranform = Transform.from_matrix(wrist_extrinsic)

                wrist_dsp_pcd = transform_point_cloud(wrist_dsp_pcd, wrist_view_tranform.translation, wrist_view_tranform.rotation.as_quat())
                if self.use_part_pcd_gt:
                    renders["wrist"]["part_pcd"] = wrist_dsp_pcd
                if self.use_part_mask_gt:
                    wrist_part_mask = project_point_cloud_to_mask(wrist_part_pcd, self.config.render_width, self.config.render_height, wrist_view_matrix, wrist_proj_matrix)
                    renders["wrist"]["part_mask"] = wrist_part_mask
                # ## DEBUG
                # color_image = visualize_segmentation_mask(wrist_part_mask)
                # cv2.imwrite("wrist_part_mask.png", color_image)
                
                # ## DEBUG
                # visualize_and_save_two_point_clouds(wrist_dsp_pcd, renders["wrist"]["pcd"], save_path="combined_pcd_wrist.ply", visualize=False)

        self.renders = renders
        renders_save = copy.deepcopy(renders)
        if self.use_pcd:
            del renders_save["agentview"]["pcd"]
            if self.use_wrist_camera:
                del renders_save["wrist"]["pcd"]
        if self.use_part_pcd_gt:
            del renders_save["agentview"]["part_pcd"]
            if self.use_wrist_camera:
                del renders_save["wrist"]["part_pcd"]
        if self.use_part_mask_gt:
            del renders_save["agentview"]["part_mask"]
            if self.use_wrist_camera:
                del renders_save["wrist"]["part_mask"]
        if self.record:
            self.render_sequence_buffer.append(renders_save)
        return renders

    def _take_action(self, action, gain=0.01, gain_gripper=0.001, gain_near_target=0.01):
        self.robot.apply_action(action, gain, gain_gripper, gain_near_target=gain_near_target)
        self.world.step()

    def _get_skill_instruction(self):
        if self.cur_skill == "grasp_obj":
            region_str =""
            instruction = f"Grasp the {self.obj_class} at {region_str}its {self.cur_skill_params['part_grasp']}"
        elif self.cur_skill == "rotate_obj":
            dir_str = mapping_rotate(self.cur_skill_params['dir_rotate'])
            instruction = f"Reorient the {self.cur_skill_params['part_rotate']} of the {self.obj_class} to face {dir_str}"
        elif self.cur_skill == "move_gripper":
            dir_str = mapping_translate(self.cur_skill_params['dir_move'])
            instruction = f"Move {dir_str}"
        elif self.cur_skill == "touch_obj":
            region_str = ""
            instruction = f"Touch the {self.obj_class} at {region_str}its {self.cur_skill_params['part_touch']}"
        elif self.cur_skill == "release_obj":
            instruction = "Release"
        else:
            instruction = ""
        return instruction

    def _get_instruction(self):

        if self.skill_mode:
            if self.cur_skill_idx >= len(self.skill_instructions) or self.skill_instructions is None:
                instruction = self._get_skill_instruction()
            else:
                instruction = self.skill_instructions[self.cur_skill_idx]
        else:
            instruction = self.task_instruction

        assert isinstance(lang_encoder, nn.Module)
        tokenized = lang_encoder.tokenize(instruction)

        return instruction, tokenized

    def _get_observation(self):
        renders = self.render_all()
        agentview_rgb = renders["agentview"]["rgb"]
        agentview_rgb = agentview_rgb.transpose((2, 0, 1))
        # import ipdb; ipdb.set_trace()

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
            agentview_part = renders["agentview"]["part_pcd"]
            observation["agentview_part_pcd"] = agentview_part
        if self.use_part_mask_gt:
            observation["agentview_part_mask"] = renders["agentview"]["part_mask"]
        # Get instruction
        if self.use_language:
            assert self.tokenized_instruction
            observation.update(self.tokenized_instruction)
            # observation["instruction"] = self.instruction

        # Get wrist camera
        if self.use_wrist_camera:
            wrist_rgb = renders["wrist"]["rgb"]
            wrist_rgb = wrist_rgb.transpose((2, 0, 1))
            observation["wrist_rgb"] = wrist_rgb
            if self.use_pcd:
                wrist_pcd = renders["wrist"]["pcd"]
                # wrist_pcd = wrist_pcd.transpose((2, 0, 1))
                observation["wrist_pcd"] = wrist_pcd
            if self.use_part_pcd_gt:
                observation["wrist_part_pcd"] = renders["wrist"]["part_pcd"]
            if self.use_part_mask_gt:
                observation["wrist_part_mask"] = renders["wrist"]["part_mask"]

        return observation

    def _check_if_done(self):
        # Determine if the episode is done
        done_cur_skill = False
        done = False
        from PartInstruct.PartGym.env.backend.planner.bullet_planner import BulletPlanner

        cur_skill_params = self.cur_skill_params.copy()
        effects = getattr(BulletPlanner, f"effects_{self.cur_skill}")(last_gripper_position=self.initial_obj_pose.translation, 
                                                                        distance = self.config.translate_distance,
                                                                        **cur_skill_params)
        # print(f"effects_{self.cur_skill}")
        # print(effects)
        # print("self.cur_skill")
        # print(self.cur_skill)
        # print("cur_skill_params")
        # print(cur_skill_params)
        
        effects_satisfied = True
        for predicate, items in effects.items():
            # print(predicate)
            result = getattr(BulletPlanner, predicate)(self, **items["params"])
            # print(result)
            if result != items["value"]:
                effects_satisfied = False
                break
        # print("effects_satisfied")
        # print(effects_satisfied)
        if effects_satisfied:
            done_cur_skill = True
            if self.cur_skill_idx >= len(self.chain_params)-1:
                done = True

        return done_cur_skill, done

    def _check_if_done_test_eval(self):
        # Determine if the episode is done
        done = False
        from PartInstruct.PartGym.env.backend.planner.bullet_planner import BulletPlanner


        assert self.task_type
        effects = getattr(BulletPlanner, f"effects_type{self.task_type}")(last_gripper_position=self.initial_obj_pose.translation, 
                                                                    distance = self.config.translate_distance,
                                                                    chain_params = self.chain_params)

        
        cnt=0
        if self.two_phases:
            effects_list = copy.deepcopy(effects)
            effects=effects_list[self.task_phase]
            total_predicates = len(effects_list[0].keys())+len(effects_list[1].keys())
            if self.task_phase==1:
                cnt=len(effects_list[0].keys())
        else:
            total_predicates = len(effects.keys())
        
        effects_satisfied = True
        
        for predicate, items in effects.items():
            result = getattr(BulletPlanner, predicate)(self, **items["params"])
            if result != items["value"]:
                effects_satisfied = False
            else:
                cnt+=1
        self.completion_rate = cnt/total_predicates
        if self.two_phases and effects_satisfied and self.task_phase==0:
            self.task_phase+=1
            effects_satisfied = False
        if effects_satisfied:
            done = True

        return done

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def sample_chain_params(self):
        with open(self.dataset_meta_path, 'r') as file:
            dataset_meta = json.load(file)
        if not self.track_samples:
            self.seed()
        chain_params_list = dataset_meta[self.obj_class][self.split][self.skill_chain]
        episode_info = self.np_random.choice(chain_params_list)
        self.obj_id = episode_info["obj_id"]
        self.ep_id = episode_info["ep_id"]
        self.chain_params = episode_info["chain_params"]
        self.task_instruction = episode_info["task_instruction"]
        self.skill_instructions = episode_info["skill_instructions"]
        self.obj_init_position = episode_info["obj_pose"][4:]
        self.obj_init_position[-1] = 0.2
        self.obj_init_orientation = episode_info["obj_pose"][:4]
        self.obj_scale = episode_info["obj_scale"]

    def save_renders(self, video_path, video_only=False):
        assert self.record
        rgbs = [renders["agentview"]["rgb"] for renders in self.render_sequence]
        if not video_only:
            depths = [renders["agentview"]["depth"] for renders in self.render_sequence]
            segmentations = [renders["agentview"]["mask"] for renders in self.render_sequence]

        if self.use_wrist_camera:
            wrist_rgbs = [renders["wrist"]["rgb"] for renders in self.render_sequence]
            if not video_only:
                wrist_depths = [renders["wrist"]["depth"] for renders in self.render_sequence]
                wrist_segmentations = [renders["wrist"]["mask"] for renders in self.render_sequence]

        video_dir, filename = os.path.split(video_path)
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        name, _ = os.path.splitext(filename)
        wrist_name = name+"_wrist"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
        out_rgb = cv2.VideoWriter(os.path.join(video_dir, name+'.mp4'), fourcc, self.control_hz, (self.config.render_width,self.config.render_height))
        
        # Save RGB video
        for frame in rgbs:
            out_rgb.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out_rgb.release()
        print(f"RGB video saved at {os.path.join(video_dir, name + '.mp4')}")

        if self.use_wrist_camera:
            out_wrist_rgb = cv2.VideoWriter(os.path.join(video_dir, wrist_name+'.mp4'), fourcc, self.control_hz, (self.robot.wrist_camera.intrinsic.width,self.robot.wrist_camera.intrinsic.height))
            # Save RGB video
            for frame in wrist_rgbs:
                out_wrist_rgb.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out_wrist_rgb.release()
        if not video_only:
            with ThreadPoolExecutor() as executor:
                depth_dir = os.path.join(video_dir, name + '_depth')
                os.makedirs(depth_dir, exist_ok=True)
                for i, depth_frame in enumerate(depths):
                    depth_image_path = os.path.join(depth_dir, f'depth_{i:04d}.png')
                    executor.submit(save_depth, depth_image_path, depth_frame)

                segmentation_dir = os.path.join(video_dir, name + '_segmentation')
                os.makedirs(segmentation_dir, exist_ok=True)
                for i, segmentation_frame in enumerate(segmentations):
                    segmentation_image_path = os.path.join(segmentation_dir, f'segmentation_{i:04d}.png')
                    executor.submit(save_image, segmentation_image_path, segmentation_frame)

                if self.use_wrist_camera:
                    depth_dir = os.path.join(video_dir, wrist_name + '_depth')
                    os.makedirs(depth_dir, exist_ok=True)
                    for i, depth_frame in enumerate(wrist_depths):
                        depth_image_path = os.path.join(depth_dir, f'depth_{i:04d}.png')
                        executor.submit(save_depth, depth_image_path, depth_frame)

                    segmentation_dir = os.path.join(video_dir, wrist_name + '_segmentation')
                    os.makedirs(segmentation_dir, exist_ok=True)
                    for i, segmentation_frame in enumerate(wrist_segmentations):
                        segmentation_image_path = os.path.join(segmentation_dir, f'segmentation_{i:04d}.png')
                        executor.submit(save_image, segmentation_image_path, segmentation_frame)

    def save_states(self, state_path):
        assert self.record
        with open(state_path, 'w') as json_file:
            json.dump(self.state_sequence, json_file, indent=4)

    def close(self):
        # Clean up PyBullet connection
        self.world.close()

    def get_current_renders(self):
        return self.renders

    def get_task_instrution(self):
        return self.task_instruction
    
    def set_cur_target_part(self, cur_target_part: str):
        self.cur_target_part = cur_target_part