from pathlib import Path
import time
import collections
from typing import List, Dict, Union, Optional, Literal
from numpy.typing import ArrayLike

import numpy as np
import pybullet
import math
import os
import pathlib
import time
import json
import copy

from PartInstruct.PartGym.env.backend.planner.vgn.vgn_detection import VGN
from PartInstruct.PartGym.env.backend.utils.grasp import Grasp
from PartInstruct.PartGym.env.backend.utils.perception import *
from PartInstruct.PartGym.env.backend.utils.vision_utils import * 
from PartInstruct.PartGym.env.backend.utils.transform import *
from omegaconf import OmegaConf
from PartInstruct.PartGym.env.backend.utils.semantic_parser import SpatialSampler, SemanticParser
from typing import Union, List, Dict, Tuple, Optional, Literal

from pybullet_tools.franka_primitives import (
    BodyPose, 
    BodyConf, 
    WorldSaver,
    get_tool_link,
    plan_direct_joint_motion,
    inverse_kinematics,
    get_movable_joints,
    set_joint_positions,
    get_sample_fn,
    plan_joint_motion,
    multiply,
    invert,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
State = collections.namedtuple("State", ["tsdf", "pc"])

root_directory = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(root_directory, "config", "config.yaml")
## Open and read the YAML file
config = OmegaConf.load(config_path)
data_root = config.data_root

# Initialize the output directory and other parameters
output_root = os.path.join(data_root, "demos")

class BulletPlanner:
    def __init__(self, env, planner_model_path: Path=Path(os.path.join(data_root, "models/vgn_conv.pth")), generation_mode: bool=False, keep_all: bool=True):
        """
        Initialize the simulation environment and load an object mesh from PartNet.
        """
        print("BulletPlanner init")
        self.generation_mode = generation_mode # if generation mode, hard stop after env.step() outputs done
        self.env = env
        self.keep_all = keep_all # keep both success and failure videos
        self.checker = OracleChecker(env)
        self.world = self.env.world
        self.robot = self.env.robot
        self.volume_size = 6 * self.robot.finger_depth
        self.record_extrinsic = Transform.look_at(np.array([0.15, -1.5, 1.0]), np.array([0.15, 0.50, 0.1]), np.array([0, 0, 1]))

        intrinsic = CameraIntrinsic(300, 300, 400.0, 400.0, 150.0, 150.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)
        self.record_camera = self.world.add_camera(intrinsic, 0.1, 10.0)
        self.record_camera.set_extrinsic(self.record_extrinsic)
        # ## DEBUG
        # rgb, _, _ = self.record_camera.render(self.record_extrinsic, pybullet.ER_BULLET_HARDWARE_OPENGL)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("test.png", rgb)

        self.grasp_planner = VGN(planner_model_path, rviz=True)
        self.parser = copy.deepcopy(self.env.parser)
        self.parser.world = self.world
    
    def reset(self):
        self.parser = copy.deepcopy(self.env.parser)

    def _plan_grasp(self, n: int, N: Optional[int]=None):
        """
        Plan a grasp using TSDF planner.
        """

        tsdf = TSDFVolume(self.volume_size, 40)
        high_res_tsdf = TSDFVolume(self.volume_size, 120)

        position, _ = self.world.get_body_pose(self.env.obj)

        origin = Transform(Rotation.identity(), np.r_[position])
        r = 1.2 * self.volume_size

        theta = np.pi / 8.0

        N = N if N else n
        phi_list = 2.0 * np.pi * np.arange(n) / N
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

        post_shift = Transform(Rotation.identity(), np.r_[position]-np.r_[self.volume_size / 2, self.volume_size / 2, position[2]])
        # T_volume_world T_world_camera
        timing = 0.0
        for extrinsic in extrinsics:
            depth_img = self.camera.semantic_render(extrinsic, self.env.obj.uid, None)[1]
            # depth_img = self.camera.semantic_render(extrinsic, self.env.obj.uid, self.env.table.uid)[1]
            tic = time.time()
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic*post_shift)
            timing += time.time() - tic
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic*post_shift)

        pc = high_res_tsdf.get_cloud()
        if pc.is_empty():
            return None
        
        state = State(tsdf, pc)
        grasps, scores, _ = self.grasp_planner(state)
        # # print(scores)
        # index = np.argmin(scores)
        # # print(index)
        # grasp, score = grasps[index], scores[index]

        for grasp in grasps:
            grasp.pose = (Transform.from_matrix(post_shift.as_matrix()))*grasp.pose
        pc = pc.transform(post_shift.as_matrix())
        
        return grasps, scores, pc
    
    def visualize_grasps(self, point_cloud: o3d.geometry.PointCloud, grasps: List[Grasp], target: Optional[np.ndarray]=None):
        """
        Visualize the point cloud and the grasps.
        
        Parameters:
            point_cloud (o3d.geometry.PointCloud): The object point cloud.
            grasps (list[Grasp]): A list of Grasp objects.
            target (np.ndarray): The center of the sphere to visualize.
        """
        # Create a visualizer object
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Create a sphere mesh to represent the point
        if target is not None:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere.paint_uniform_color([0, 0, 1])
            sphere.translate(target)  
            vis.add_geometry(sphere)

            # num_points = np.asarray(point_cloud.points).shape[0]
            # white_colors = np.ones((num_points, 3))
            # point_cloud.colors = o3d.utility.Vector3dVector(white_colors)

        # num_points = np.asarray(point_cloud.points).shape[0]
        # white_colors = np.ones((num_points, 3))
        # point_cloud.colors = o3d.utility.Vector3dVector(white_colors)

        # Add the point cloud
        vis.add_geometry(point_cloud)
        # Define the gripper dimensions
        gripper_length = self.robot.finger_depth 

        def create_gripper_lines(transform: Transform, width: float):
            # Points in the gripper frame
            half_width = width / 2
            points = [
                [0, 0, 0], # origin
                [0, -half_width, 0], [0, half_width, 0],  # Base of the gripper
                [0, 0, -gripper_length/2], # handle tip
                [0, -half_width, gripper_length], [0, half_width, gripper_length]  # Tips of the gripper fingers
            ]
            points = np.dot(transform.as_matrix()[:3, :3], np.array(points).T).T
            points += transform.translation
            # Lines defining the gripper edges
            lines = [
                [0, 3], [1, 2],  
                [1, 4], [2, 5]  
            ]
            # Colors for each line
            colors = [[1, 0, 0] for _ in range(len(lines))]  # Start with all lines green
            # colors[0] = [1, 0, 0]  # Make the handle line red
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            return line_set

        # Add each grasp to the visualizer
        for grasp in grasps:
            gripper_lines = create_gripper_lines(grasp.pose, grasp.width)
            vis.add_geometry(gripper_lines)

        # Run the visualizer
        vis.run()
        vis.destroy_window()

    #########################
    #  Robot Move Wrappers  #
    #########################

    def pose_tcp(self, target: Transform, gain: float=0.01, gain_near_target: float=0.01, norm_threshold: float=0.01, close_threshold: float=0.01, max_iteration: int=1000, allow_contact: bool=False, check_collision: bool=False, check_obj_collision: bool=False, grasping: bool=False, timeout: int=30):
        # T_world_gripper = target * self.robot.T_tcp_gripper
        
        tcp_position = target.translation
        tcp_orientation = target.rotation.as_euler('xyz', degrees=False)

        x = tcp_position[0]
        y = tcp_position[1]
        z = tcp_position[2]
        roll = tcp_orientation[0]
        pitch = tcp_orientation[1]
        yaw = tcp_orientation[2]
        current_gripper_state = self.robot.get_gripper_state()
        gripper_state = min([self.robot.GRIPPER_CLOSED_JOINT_POS, self.robot.GRIPPER_OPEN_JOINT_POS], key=lambda x: abs(x - current_gripper_state))

        action = [x, y, z, roll, pitch, yaw, gripper_state]

        it = 0
        ret = False
        env_output = (None, None, None, None)

        start_time = time.time()
        previous_tcp_pose = None
        
        try:
            while True:
                if timeout is not None and (time.time() - start_time) > timeout:
                    print(f"Operation timed out after {timeout} seconds.")
                    break
                
                observation, reward, done, info = self.env.step(action, gain, gain_near_target=gain_near_target)
                # update parser
                self.parser.update_part_pcds()
                if self.generation_mode and done:
                    ret = True
                    break
                
                if grasping and (not self.robot.detect_grasp_contact()):
                    break

                if not allow_contact and self.robot.detect_grasp_contact():
                    break

                if check_collision and (self.robot.check_self_collision() or self.robot.check_collision([self.env.table.uid])):
                    break

                if check_obj_collision and self.world.check_body_collision(self.env.table, self.env.obj):
                    break

                current_tcp_pose = self.robot.get_tcp_pose()

                if previous_tcp_pose is not None and pose_distance(current_tcp_pose, previous_tcp_pose) < close_threshold:
                    print("Current pose between two iterations is too close. Breaking the loop.")
                    ret = True
                    break

                previous_tcp_pose = current_tcp_pose

                if pose_distance(current_tcp_pose, target) < norm_threshold:
                    ret = True
                    break

                it += 1

                if it >= max_iteration:
                    print(f"Reached maximum iterations of {max_iteration}.")
                    break
        except KeyboardInterrupt:
            print("KeyboardInterrupt has been caught")

        env_output = (observation, reward, done, info)
        return ret, env_output

    def switch_gripper(self, state: Literal[0, 1], threshold: float=0.009, max_iteration: int=100, abort_on_contact: bool=True):
        if state == self.robot.OPEN:
            finger_target = self.robot.GRIPPER_OPEN_JOINT_POS
        elif state == self.robot.CLOSED:
            finger_target = self.robot.GRIPPER_CLOSED_JOINT_POS

        current_tcp_pose = self.robot.get_tcp_pose()
        tcp_position = current_tcp_pose.translation
        tcp_orientation = current_tcp_pose.rotation.as_euler('xyz', degrees=False)

        x = tcp_position[0] 
        y = tcp_position[1] 
        z = tcp_position[2] 
        roll = tcp_orientation[0] 
        pitch = tcp_orientation[1] 
        yaw = tcp_orientation[2] 
        gripper_state = finger_target

        action = [x, y, z, roll, pitch, yaw, gripper_state]

        it = 0
        ret = False
        last_dist = np.inf

        try:
            while True:
                all_close = True
                observation, reward, done, info = self.env.step(action, gain=0.0001, gain_gripper=0.01)
                current_joint_position = self.robot.get_gripper_state()
                # update parser
                self.parser.update_part_pcds()

                if self.generation_mode and done:
                    ret = True
                    break
                
                current_dist = np.abs(finger_target- current_joint_position)

                if current_dist > threshold:
                    all_close = False
                
                del_dist = np.abs(last_dist-current_dist)
                
                if (abort_on_contact and self.robot.detect_grasp_contact()) \
                        or all_close \
                        or del_dist<0.1*threshold:
                    ret = True
                    for _ in range(20):
                        observation, reward, done, info = self.env.step(action, gain=0.0001, gain_gripper=0.1)
                    break

                last_dist = current_dist

                it+=1

                if it >= max_iteration:
                    print(f"Reached maximum iterations of {max_iteration}.")
                    break

        except KeyboardInterrupt:
            print("KeyboardInterrupt has been caught")
        
        env_output = (observation, reward, done, info)
        # print("After gripper closed")
        # print(info["Current Skill"])
        # print(info["Current Skill Success"])

        return ret, env_output
    
    ##################
    #  Action Utils  #
    ##################

    def plan_grasps(self, obj, part_grasp="", region_on_part="", bbox_extension_ratio=1.0):
        assert obj in self.parser.part_pcds_t0.keys(), "Invalid object name. Make sure to load part hierarchy."
        assert part_grasp in self.parser.part_pcds_t0.keys() or part_grasp=="", "Invalid part entity. Make sure to load part hierarchy."
        assert region_on_part in self.parser.spatial_sampler.locative_nouns or region_on_part=="", "Invalid region on part. Available locative nouns: front, back, top, bottom, left, right, middle."

        # language grounding
        grounded_pcd = self.parser.part_pcds[self.parser.obj_name] if part_grasp=="" else self.parser.part_pcds[part_grasp]
        grounded_pcd = grounded_pcd if region_on_part=="" else self.parser.spatial_sampler.sample_query(grounded_pcd, region_on_part)

        # Compute the bounding box of the grounded_pcd with an extension
        min_point = np.min(grounded_pcd, axis=0)
        max_point = np.max(grounded_pcd, axis=0)
        bbox_diagonal = np.linalg.norm(max_point - min_point)
        bbox_extension = bbox_extension_ratio * bbox_diagonal

        # Apply the adaptive extension
        min_point -= bbox_extension
        max_point += bbox_extension

        # get target point
        target_point =  np.mean(grounded_pcd, axis=0)
        # target_point = target_point+np.array([0, 0, 0.02])
        # target_point = target_point

        # grasp planning
        grasps, scores, pc = self._plan_grasp(n=5)
        T_grasp = Transform(Rotation.identity(), list(target_point))

        # Filter grasps to ensure the TCP position is within the expanded bounding box
        filtered_grasps = []
        filtered_scores = []
        # print("All grasps")
        # print(len(grasps))
        for grasp, score in zip(grasps, scores):
            tcp_position = grasp.pose.translation
            if np.all(tcp_position >= min_point) and np.all(tcp_position <= max_point):
                filtered_grasps.append(grasp)
                filtered_scores.append(score)
        # print("Filtered grasps")
        # print(len(filtered_grasps))
        return T_grasp, filtered_grasps, filtered_scores, pc
        
    def execute_grasp(self, grasp: Grasp, allow_contact: bool=False, retreat_height: float=0.25, part_grasp=''):
        self.checker.reset_skill("grasp_obj", {
            "part_grasp": part_grasp,
        })
        self.switch_gripper(self.robot.OPEN)
        T_world_grasp = grasp.pose
        depth = self.robot.finger_depth+self.robot.finger_depth/10
        openning_half_width = self.robot.get_gripper_state()
        left_tip_tcp = np.array([0.0, openning_half_width, depth])
        right_tip_tcp = np.array([0.0, -openning_half_width, depth])
        left_tip_world = T_world_grasp*Transform(Rotation.identity(), left_tip_tcp)
        right_tip_world = T_world_grasp*Transform(Rotation.identity(), right_tip_tcp)
        # self.world.draw_frame_axes(left_tip_world.translation, left_tip_world.rotation.as_matrix())
        # self.world.draw_frame_axes(right_tip_world.translation, right_tip_world.rotation.as_matrix())
        left_tip_world_position = left_tip_world.translation
        right_tip_world_position = right_tip_world.translation
        finger_z = min(left_tip_world_position[2], right_tip_world_position[2])
        table_z = self.env.table.get_pose().translation[2]
        if finger_z<table_z:
            T_world_grasp.translation[2] = T_world_grasp.translation[2]+(table_z-finger_z)
        # self.world.draw_frame_axes(T_world_grasp.translation, T_world_grasp.rotation.as_matrix())
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, retreat_height])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -retreat_height])
            T_world_retreat = T_world_grasp * T_grasp_retreat
        # print(T_world_pregrasp)
        result = False
        ret, env_output = self.pose_tcp(T_world_pregrasp, gain=0.005, gain_near_target=0.2, norm_threshold=0.05, close_threshold=0.001, max_iteration=150, check_collision=True)
        _, _, done, info = env_output
        if ret==False:
            return result, T_world_pregrasp, T_world_grasp, T_world_retreat, env_output

        if self.robot.detect_grasp_contact() and not allow_contact:
            return result, T_world_pregrasp, T_world_grasp, T_world_retreat, env_output
        else:
            # # DEBUG
            # self.world.draw_frame_axes(T_world_pregrasp.translation, T_world_pregrasp.rotation.as_matrix())
            # self.world.draw_frame_axes(T_world_grasp.translation, T_world_grasp.rotation.as_matrix())
            ret, env_output = self.pose_tcp(T_world_grasp, norm_threshold=0.003, max_iteration=120, allow_contact=True, close_threshold=0.0005)
            _, _, done, info = env_output
            if ret==False:
                return result, T_world_pregrasp, T_world_grasp, T_world_retreat, env_output
            if self.robot.detect_grasp_contact() and not allow_contact:
                pass
            else:
                ret, env_output = self.switch_gripper(self.robot.CLOSED, abort_on_contact=False)
                _, _, done, info = env_output
                if self.generation_mode and (done or self.checker.is_skill_done(self.parser)):
                    result = True
                elif not self.generation_mode and (self.robot.detect_grasp_contact() and not self.robot.check_self_collision()):
                    result = True
                else:
                    result = False

        return result, T_world_pregrasp, T_world_grasp, T_world_retreat, env_output
    
    def target_grasp(self, target_pose: Transform, grasps: List[Grasp], scores: List[float], thred=0.5, max_attempts=1, score_weight=0.5, retreat_height=0.25, pc=None, visualize=False, part_grasp=None):
        # best_grasp = None
        # best_evaluation = float('-inf')
        current_tcp_pose = self.robot.get_tcp_pose()
        ret = False
        executed_grasp = None
        T_world_pregrasp = None
        T_world_retreat = None
        current_obj_pose = self.env.obj.get_pose()
        current_joint_states = self.robot.get_joint_states()
        current_gripper_state = self.robot.get_gripper_state()
        current_joint_states = current_joint_states+2*[current_gripper_state]
        # self.world.draw_frame_axes(current_obj_pose.translation, current_obj_pose.rotation.as_matrix())
        aabb_min, aabb_max, max_distance = self.env.obj.get_bbox()

        distances = [pose_distance(grasp.pose, target_pose, orientation_weight=0.0) for grasp in grasps]
        # print(distances)
        inv_norm_distance = [1 - (d / max_distance) for d in distances]
        # print(inv_norm_distance)
        grasp_evaluations = []
        sorted_grasps = []

        for i, (norm_distance, grasp) in enumerate(zip(inv_norm_distance, grasps)):
            # Score is already normalized to [0, 1]
            norm_score = scores[i]
            # print(norm_score)
            # print(score_weight) 
            evaluation = score_weight*norm_score + (1-score_weight)*norm_distance
            # print(evaluation)
            if evaluation > thred:
                grasp_evaluations.append((grasp, evaluation))
        
        # print(len(grasp_evaluations)) 
        sorted_grasps = sorted(grasp_evaluations, key=lambda x: x[1], reverse=True)
        length = len(sorted_grasps)
        # print(length)
        sorted_grasps = [grasp for grasp, _ in sorted_grasps][:int(min(length, max_attempts))]
        
        try:
            for i, grasp in enumerate(sorted_grasps):
                # if self.state==0 and (i==0 or i==1 or i==2):
                #     continue
                # if self.state==1 and (i==0):
                #     continue
                if visualize: 
                    self.visualize_grasps(pc, [grasp], target=target_pose.translation)
                    # self.world.draw_frame_axes(grasp.pose.translation, grasp.pose.rotation.as_matrix())
                ret, T_world_pregrasp, _, T_world_retreat, env_output = self.execute_grasp(grasp, allow_contact=True, retreat_height=retreat_height, part_grasp=part_grasp)
                # print(ret)
                _, _, done, info = env_output
                # print("self.env.cur_skill_idx")
                # print(self.env.cur_skill_idx)
                if self.generation_mode and (done or ret):
                    executed_grasp = grasp
                    self.env.dump_buffers()
                    break
                elif not self.generation_mode and ret:
                    executed_grasp = grasp
                    self.env.dump_buffers()
                    break
                else:
                    if not self.keep_all:
                        self.env.clear_buffers()
                        self.env.obj.set_pose(current_obj_pose)
                        self.robot.reset(current_joint_states)
                    else:
                        self.env.dump_buffers()
        
        except KeyboardInterrupt:
            print(print("KeyboardInterrupt has been caught"))
        
        # time.sleep(self.world.dt*100)

        print("Target grasping completed.")

        return ret, current_obj_pose, current_tcp_pose, executed_grasp, T_world_pregrasp, T_world_retreat

    def target_rotate(self, rel_rotation, grasping=True, part_rotate=None, dir_rotate=None):
        """
        Rotate the grasped object. rotation is the desired object rotation.
        """
        self.checker.reset_skill("rotate_obj", {
            "part_rotate": part_rotate,
            "dir_rotate": dir_rotate,
            "grasping": grasping
        })
        current_tcp_pose = self.robot.get_tcp_pose()
        current_obj_pose = self.env.obj.get_pose()

        T_world_tcp_c = current_tcp_pose

        rotation_world_tcp_t = rel_rotation*T_world_tcp_c.rotation
        T_world_tcp_t = Transform(rotation_world_tcp_t, T_world_tcp_c.translation)
        
        target_pose = None
        ret = False
        ret, env_output = self.pose_tcp(T_world_tcp_t, gain=0.005, allow_contact=True, norm_threshold=0.2, close_threshold=0.01, max_iteration=80, grasping=grasping)
        # self.world.draw_frame_axes(T_world_tcp_t.translation, T_world_tcp_t.rotation.as_matrix())
        # ret, optimized_position, env_output = self.free_orient_tcp(T_world_tcp_t.rotation.as_quat(), norm_threshold=0.5, max_iteration=80)
        _, _, done, info = env_output
        if self.generation_mode and (done or self.checker.is_skill_done(self.parser)):
            ret = True
            self.env.dump_buffers()
            target_pose = T_world_tcp_t
            # target_pose = Transform(T_world_tcp_t.rotation, np.array(optimized_position))
        elif not self.generation_mode and ret:
            ret = True
            self.env.dump_buffers()
            target_pose = T_world_tcp_t
            # target_pose = Transform(T_world_tcp_t.rotation, np.array(optimized_position))
        else:
            ret = False
            if not self.keep_all:
                self.env.clear_buffers()
            else:
                self.env.dump_buffers()

        print("Target rotation completed.")

        return ret, current_obj_pose, current_tcp_pose, target_pose

    ################
    #  Predicates  #
    ################
    @staticmethod
    def predicate_on_table(env, parser: Optional[SemanticParser]=None):
        return env.world.check_body_collision(env.table, env.obj)
    
    @staticmethod
    def predicate_gripper_open(env, parser: Optional[SemanticParser]=None, threshold=0.005):
        current_gripper_state = env.robot.get_gripper_state()
        if abs(current_gripper_state-env.robot.GRIPPER_OPEN_JOINT_POS) < threshold:
            return True
        return False
    
    @staticmethod
    def predicate_min_distance(env, parser: Optional[SemanticParser]=None, min_distance=0.05):
        current_obj_pose = env.obj.get_pose()
        current_tcp_pose = env.robot.get_tcp_pose()
        distance = np.linalg.norm(current_obj_pose.translation - current_tcp_pose.translation)
        if distance>min_distance:
            return True
        return False
    
    @staticmethod
    def predicate_touching(env, parser: Optional[SemanticParser]=None, part_name=""):
        ## check gripper closeness
        # enable collision check between the two fingers
        env.world.p.setCollisionFilterPair(env.robot.panda, env.robot.panda, env.robot.LEFTFINGERIDX, env.robot.RIGHTFINGERIDX, 1)
        # check collision between two fingers
        env.world.step()
        ret_closeness = env.world.check_link_contact(env.robot.panda, env.robot.panda, env.robot.LEFTFINGERIDX, env.robot.RIGHTFINGERIDX)
        # disable collision check between the two fingers
        env.world.p.setCollisionFilterPair(env.robot.panda, env.robot.panda, env.robot.LEFTFINGERIDX, env.robot.RIGHTFINGERIDX, 0)
        ## check collison between two fingers and obj part
        if parser is None:
            parser=env.parser
        parser.update_part_pcds()
        grounded_pcd = parser.part_pcds[parser.obj_name] if part_name=="" else parser.part_pcds[part_name]
        ret_collision_gripper = env.world.check_point_cloud_gripper_collision(env.robot, grounded_pcd)
        ret = (ret_closeness and ret_collision_gripper)
        # print("ret_closeness")
        # print(ret_closeness)
        # print("ret_collision_gripper")
        # print(ret_collision_gripper)
        return ret

    @staticmethod
    def predicate_grasping(env, parser: Optional[SemanticParser]=None, part_name=""):
        ## check collison between two fingers and obj part
        if parser is None:
            parser=env.parser
        parser.update_part_pcds()
        grounded_pcd = parser.part_pcds[parser.obj_name] if part_name=="" else parser.part_pcds[part_name]
        ret_collision_left = env.world.check_obj_gripper_collision(env.robot.panda, env.robot.LEFTFINGERIDX, grounded_pcd, 0.003, 20.0)
        ret_collision_right = env.world.check_obj_gripper_collision(env.robot.panda, env.robot.RIGHTFINGERIDX, grounded_pcd, 0.003, 20.0)
        ret = (ret_collision_left or ret_collision_right)
        return ret

    @staticmethod
    def predicate_at_position(env, parser: Optional[SemanticParser]=None, is_obj: bool=True, position: ArrayLike=[0,0,0], position_thred=0.02):
        
        if is_obj:
            current_position = env.obj.get_pose().translation
        else:
            current_position = env.robot.get_tcp_pose().translation   
        distance = np.linalg.norm(current_position-np.array(position))
        ret = False
        if distance<position_thred:
            ret = True
        return ret

    @staticmethod
    def predicate_facing_direction(env, part_name: str, parser: Optional[SemanticParser]=None, direction: ArrayLike=[1,0,0], direction_thred=0.5):
        position = env.obj.get_pose().translation
        if parser is None:
            parser=env.parser
        parser.update_part_pcds()
        grounded_pcd = parser.part_pcds[part_name]
        grounded_center =  np.mean(grounded_pcd, axis=0)
        current_direction = grounded_center-np.array(position)
        _, diff = rotation_quaternion(current_direction, direction)
        ret = False
        if diff<direction_thred:
            ret = True
        return ret
    
    @staticmethod
    def predicate_facing_opposite_direction(env, part_name: str, parser: Optional[SemanticParser]=None, direction: ArrayLike=[1,0,0], direction_thred=0.5):
        position = env.obj.get_pose().translation
        if parser is None:
            parser=env.parser
        parser.update_part_pcds()
        grounded_pcd = parser.part_pcds[part_name]
        grounded_center =  np.mean(grounded_pcd, axis=0)
        current_direction = grounded_center-np.array(position)
        _, diff = rotation_quaternion(current_direction, direction)
        ret = False
        if diff<direction_thred:
            ret = True
        return ret

    ####################
    #  Pre-conditions  #
    ####################
    @staticmethod
    def preconditions_grasp_obj(env):
        # check if the object is on the table
        ret_on_table = BulletPlanner.predicate_on_table(env)
        # check if the object is not being grasped
        ret_grasping = BulletPlanner.predicate_grasping(env)
        # check if the object is not being touched
        ret_touching = BulletPlanner.predicate_touching(env)
        ret = ret_on_table and (not ret_grasping) and (not ret_touching)
        return ret
    
    @staticmethod
    def preconditions_move_gripper(env, grasping: bool=False, touching: bool=False, put_down: bool=False):
        if grasping:
            assert not touching, "Grasping and touching cannot be true at the same time."
            # check if the object is being grasped
            ret = BulletPlanner.predicate_grasping(env)
            if put_down:
                # check if the object is on the table
                ret = ret and (not BulletPlanner.predicate_on_table(env))
        elif touching:
            # check if the object is being touched
            ret = BulletPlanner.predicate_touching(env)
        else:
            assert not put_down, "Invalid object for put down action."
            ret = True
        return ret

    @staticmethod
    def preconditions_rotate_obj(env):
        # check if the object is being grasped
        ret = BulletPlanner.predicate_grasping(env)
        return ret

    @staticmethod
    def preconditions_touch_obj(env):
        # check if the object is on the table
        ret_on_table = BulletPlanner.predicate_on_table(env)
        # check if the object is not being grasped
        ret_grasping = BulletPlanner.predicate_grasping(env)
        # check if the object is not being touched
        ret_touching = BulletPlanner.predicate_touching(env)
        ret = ret_on_table and (not ret_touching) and (not ret_grasping)
        return ret

    @staticmethod
    def preconditions_release_obj(env):
        # # check if the object is on the table
        # ret_on_table = BulletPlanner.predicate_on_table(env)
        # check if the object is being grasped
        # ret_grasping = BulletPlanner.predicate_grasping(env)
        # check if the object is being touched
        # ret_touching = BulletPlanner.predicate_touching(env)
        # print("ret_on_table")
        # print(ret_on_table)
        # print("ret_grasping")
        # print(ret_grasping)
        # print("ret_touching")
        # print(ret_touching)
        # ret = ret_on_table and (ret_grasping or ret_touching)
        # ret = (ret_grasping or ret_touching)
        ret = True # can perform release at any time
        return ret

    #############
    #  Effects  #
    #############
    @staticmethod
    def effects_type1(chain_params: list, **kwargs):
        part_grasp = chain_params[0]["params"]["part_grasp"]
        effects = {}
        # the object is being grasped
        effects = BulletPlanner.effects_grasp_obj(part_grasp)
        return effects

    @staticmethod
    def effects_type2(chain_params: list, **kwargs):
        part_touch = chain_params[0]["params"]["part_touch"]
        effects = {}
        # the object is being touched
        effects = BulletPlanner.effects_touch_obj(part_touch)
        return effects
    
    @staticmethod
    def effects_type3(chain_params: list, last_gripper_position: ArrayLike, distance: float, position_thred: float=0.05, **kwargs):
        part_grasp = chain_params[0]["params"]["part_grasp"]
        args_move = chain_params[1]["params"]
        dir_move = args_move["dir_move"]
        put_down = args_move["put_down"]
        touching = args_move["touching"]
        # the object is being touched
        effects = {}
        effects = BulletPlanner.effects_grasp_obj(part_grasp)
        if put_down:
            assert dir_move=="bottom", "Invalid direction for moving the gripper down."
            # the object is on the table
            effects["predicate_on_table"] = {"params": {}, "value": True}
        if touching:
            # the object is still being touched
            effects["predicate_touching"] = {"params": {"part_name": ""}, "value": True}
        if isinstance(dir_move, str):
            dir_move = np.array(SpatialSampler.gaussian_param_mappings[dir_move]['mu'])
        if not put_down:
            goal_position = np.array(last_gripper_position)+np.array(dir_move)*distance
            effects["predicate_at_position"] = {"params": {"position": goal_position,
                                                           "position_thred": position_thred}, 
                                                "value": True}
        return effects

    @staticmethod
    def effects_type4(chain_params: list, last_gripper_position: ArrayLike, distance: float, position_thred: float=0.05, **kwargs):
        part_touch = chain_params[0]["params"]["part_touch"]
        args_move = chain_params[1]["params"]
        dir_move = args_move["dir_move"]
        # the object is being touched
        effects = {}
        effects = BulletPlanner.effects_touch_obj(part_touch)
        if isinstance(dir_move, str):
            dir_move = np.array(SpatialSampler.gaussian_param_mappings[dir_move]['mu'])
        goal_position = np.array(last_gripper_position)+np.array(dir_move)*distance
        effects["predicate_at_position"] = {"params": {"position": goal_position,
                                                       "position_thred": position_thred}, 
                                            "value": True}
        effects["predicate_on_table"] = {"params": {}, "value": True}
        return effects
    
    @staticmethod # 2p
    def effects_type5(chain_params: list, last_gripper_position: ArrayLike, distance: float, **kwargs):
        # the object is being touched
        effects_list = []
        effects = {}
        effects = BulletPlanner.effects_type3(chain_params, last_gripper_position=last_gripper_position, distance=distance, position_thred=0.03)
        effects_list.append(effects)
        effects = {}
        effects = BulletPlanner.effects_release_obj()
        effects_list.append(effects)
        return effects_list
    
    @staticmethod # 2p
    def effects_type6(chain_params: list, last_gripper_position: ArrayLike, distance: float, **kwargs):
        effects_list = []
        effects = {}
        effects = BulletPlanner.effects_type4(chain_params, last_gripper_position=last_gripper_position, distance=distance, position_thred=0.03)
        effects_list.append(effects)
        effects = {}
        effects = BulletPlanner.effects_release_obj()
        effects_list.append(effects)
        return effects_list
    
    @staticmethod
    def effects_type7(chain_params: list, last_gripper_position: ArrayLike, distance: float, **kwargs):
        part_grasp = chain_params[0]["params"]["part_grasp"]
        args_move_1 = chain_params[1]["params"]
        dir_move_1 = args_move_1["dir_move"]
        args_move_2 = chain_params[2]["params"]
        dir_move_2 = args_move_2["dir_move"]
        effects = {}
        effects = BulletPlanner.effects_grasp_obj(part_grasp)
        if isinstance(dir_move_1, str):
            dir_move_1 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_1]['mu'])
        if isinstance(dir_move_2, str):
            dir_move_2 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_2]['mu'])

        goal_position = np.array(last_gripper_position)+np.array(dir_move_1)*distance+np.array(dir_move_2)*distance
        effects["predicate_at_position"] = {"params": {"position": goal_position}, 
                                                "value": True}
        return effects
    
    @staticmethod
    def effects_type8(chain_params: list, **kwargs):
        part_rotate = chain_params[2]["params"]["part_rotate"]
        dir_rotate = chain_params[2]["params"]["dir_rotate"]
        effects = {}
        effects = BulletPlanner.effects_grasp_obj()
        effects.update(BulletPlanner.effects_rotate_obj(part_rotate, dir_rotate))
        effects["predicate_on_table"] = {"params": {}, "value": False}
        return effects

    @staticmethod # 2p
    def effects_type9(chain_params: list, last_gripper_position: ArrayLike, distance: float, **kwargs):
        effects_list = []
        part_grasp = chain_params[0]["params"]["part_grasp"]
        args_move_1 = chain_params[1]["params"]
        dir_move_1 = args_move_1["dir_move"]
        args_move_2 = chain_params[2]["params"]
        dir_move_2 = args_move_2["dir_move"]
        args_move_3= chain_params[3]["params"]
        dir_move_3 = args_move_3["dir_move"]
        if isinstance(dir_move_1, str):
            dir_move_1 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_1]['mu'])
        if isinstance(dir_move_2, str):
            dir_move_2 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_2]['mu'])
        if isinstance(dir_move_3, str):
            dir_move_3 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_3]['mu'])
        
        effects = {}
        effects = BulletPlanner.effects_grasp_obj(part_grasp)

        goal_position = np.array(last_gripper_position)+np.array(dir_move_1)*distance+np.array(dir_move_2)*distance
        effects["predicate_at_position"] = {"params": {"position": goal_position,
                                                       "position_thred": 0.08}, 
                                            "value": True}
        effects_list.append(effects)
        
        effects = {}
        effects = BulletPlanner.effects_grasp_obj()
        goal_position = np.array(last_gripper_position)+np.array(dir_move_1)*distance+np.array(dir_move_2)*distance+np.array(dir_move_3)*distance
        effects["predicate_at_position"] = {"params": {"position": goal_position,
                                                       "position_thred": 0.08}, 
                                            "value": True}
        effects_list.append(effects)

        return effects_list
    
    @staticmethod
    def effects_type10(chain_params: list, last_gripper_position: ArrayLike, distance: float, **kwargs):
        part_rotate = chain_params[3]["params"]["part_rotate"]
        dir_rotate = chain_params[3]["params"]["dir_rotate"]

        args_move_1 = chain_params[1]["params"]
        dir_move_1 = args_move_1["dir_move"]
        args_move_2 = chain_params[2]["params"]
        dir_move_2 = args_move_2["dir_move"]

        if isinstance(dir_move_1, str):
            dir_move_1 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_1]['mu'])
        if isinstance(dir_move_2, str):
            dir_move_2 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_2]['mu'])

        effects = {}
        effects = BulletPlanner.effects_grasp_obj()
        effects.update(BulletPlanner.effects_rotate_obj(part_rotate, dir_rotate))

        goal_position = np.array(last_gripper_position)+np.array(dir_move_1)*distance+np.array(dir_move_2)*distance
        effects["predicate_at_position"] = {"params": {"position": goal_position,
                                                       "position_thred": 0.08}, 
                                                "value": True}

        return effects

    @staticmethod # 2p
    def effects_type11(chain_params: list, last_gripper_position: ArrayLike, distance: float, **kwargs):
        # the object is being touched
        effects_list = []
        effects = {}
        effects = BulletPlanner.effects_type8(chain_params, last_gripper_position=last_gripper_position, distance=distance)
        effects_list.append(effects)
        effects = {}
        goal_position = np.array(last_gripper_position)
        effects["predicate_at_position"] = {"params": {"position": goal_position,
                                                       "position_thred": 0.08}, 
                                                "value": True}
        effects_list.append(effects)
        return effects_list
    
    @staticmethod # 2p
    def effects_type12(chain_params: list, last_gripper_position: ArrayLike, distance: float, **kwargs):
        effects_list = []
        part_grasp = chain_params[0]["params"]["part_grasp"]
        args_move_1 = chain_params[1]["params"]
        dir_move_1 = args_move_1["dir_move"]
        args_move_2 = chain_params[2]["params"]
        dir_move_2 = args_move_2["dir_move"]

        part_rotate = chain_params[3]["params"]["part_rotate"]
        dir_rotate = chain_params[3]["params"]["dir_rotate"]
        args_move_3 = chain_params[4]["params"]
        dir_move_3 = args_move_3["dir_move"]

        if isinstance(dir_move_1, str):
            dir_move_1 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_1]['mu'])
        if isinstance(dir_move_2, str):
            dir_move_2 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_2]['mu'])
        if isinstance(dir_move_3, str):
            dir_move_3 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_3]['mu'])
        
        effects = {}
        effects = BulletPlanner.effects_grasp_obj(part_grasp)

        goal_position = np.array(last_gripper_position)+np.array(dir_move_1)*distance+np.array(dir_move_2)*distance
        effects["predicate_at_position"] = {"params": {"position": goal_position,
                                                       "position_thred": 0.05}, 
                                            "value": True}
        effects_list.append(effects)

        effects = {}
        effects = BulletPlanner.effects_grasp_obj()
        goal_position = np.array(last_gripper_position)+np.array(dir_move_1)
        effects["predicate_at_position"] = {"params": {"position": goal_position,
                                                       "position_thred": 0.08}, 
                                            "value": True}
        effects.update(BulletPlanner.effects_rotate_obj(part_rotate, dir_rotate))
        effects_list.append(effects)

        return effects_list
    
    @staticmethod # 2p
    def effects_type13(chain_params: list, last_gripper_position: ArrayLike, distance: float, **kwargs):
        effects_list = []
        part_rotate = chain_params[1]["params"]["part_rotate"]
        dir_rotate = chain_params[1]["params"]["dir_rotate"]
        args_move = chain_params[-1]["params"]
        dir_move = args_move["dir_move"]

        if isinstance(dir_rotate, str):
            dir_rotate = np.array(SpatialSampler.gaussian_param_mappings[dir_rotate]['mu'])
        if isinstance(dir_move, str):
            dir_move = np.array(SpatialSampler.gaussian_param_mappings[dir_move]['mu'])
        
        effects = {}
        effects = BulletPlanner.effects_rotate_obj(part_rotate, dir_rotate)
        effects["predicate_on_table"] = {"params": {}, "value": True}

        effects_list.append(effects)

        effects = {}
        goal_position = np.array(last_gripper_position)+np.array(dir_move)
        effects["predicate_at_position"] = {"params": {"position": goal_position,
                                                       "position_thred": 0.08}, 
                                            "value": True}
        effects["predicate_on_table"] = {"params": {}, "value": True}
        effects_list.append(effects)

        return effects_list

    @staticmethod # 2p
    def effects_type14(chain_params: list, last_gripper_position: ArrayLike, distance: float, **kwargs):
        effects_list = []
        part_rotate = chain_params[-1]["params"]["part_rotate"]
        dir_rotate = chain_params[-1]["params"]["dir_rotate"]

        if isinstance(dir_rotate, str):
            dir_rotate = np.array(SpatialSampler.gaussian_param_mappings[dir_rotate]['mu'])

        effects = {}
        effects = BulletPlanner.effects_type4(chain_params, last_gripper_position, distance)
        effects["predicate_on_table"] = {"params": {}, "value": True}

        effects_list.append(effects)

        effects = {}
        effects = BulletPlanner.effects_rotate_obj(part_rotate, dir_rotate)
        effects["predicate_on_table"] = {"params": {}, "value": True}
        effects_list.append(effects)

        return effects_list

    @staticmethod
    def effects_type15(chain_params: list, **kwargs):
        part_rotate = "left"
        dir_rotate = "right"

        if isinstance(dir_rotate, str):
            dir_rotate = np.array(SpatialSampler.gaussian_param_mappings[dir_rotate]['mu'])

        effects = {}
        effects = BulletPlanner.effects_rotate_obj(part_rotate, dir_rotate)

        effects["predicate_on_table"] = {"params": {}, "value": True}

        return effects

    @staticmethod # 2p
    def effects_type16(chain_params: list, last_gripper_position: ArrayLike, distance: float, **kwargs):
        effects_list = []
        args_move_1 = chain_params[1]["params"]
        dir_move_1 = args_move_1["dir_move"]
        args_move_2 = chain_params[4]["params"]
        dir_move_2 = args_move_2["dir_move"]
        part_rotate = chain_params[5]["params"]["part_rotate"]
        dir_rotate = chain_params[5]["params"]["dir_rotate"]

        if isinstance(dir_rotate, str):
            dir_rotate = np.array(SpatialSampler.gaussian_param_mappings[dir_rotate]['mu'])
        if isinstance(dir_move_1, str):
            dir_move_1 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_1]['mu'])
        if isinstance(dir_move_2, str):
            dir_move_2 = np.array(SpatialSampler.gaussian_param_mappings[dir_move_2]['mu'])

        effects = {}
        effects = BulletPlanner.effects_rotate_obj(part_rotate, dir_rotate)
    
        goal_position = np.array(last_gripper_position)+np.array(dir_move_1)+np.array(dir_move_2)
        effects["predicate_at_position"] = {"params": {"position": goal_position,
                                                       "position_thred": 0.05}, 
                                            "value": True}
        effects["predicate_on_table"] = {"params": {}, "value": False}


        effects_list.append(effects)

        effects = {}
    
        goal_position = np.array(last_gripper_position)+np.array(dir_move_1)
        effects["predicate_at_position"] = {"params": {"position": goal_position,
                                                       "position_thred": 0.08}, 
                                            "value": True}

        effects_list.append(effects)

        return effects_list

    @staticmethod
    def effects_grasp_obj(part_grasp="", **kwargs):
        effects = {}
        # the object is being grasped
        effects["predicate_grasping"] = {"params": {"part_name": part_grasp}, "value": True}
        return effects

    @staticmethod
    def effects_move_gripper(grasping: bool, touching: bool, dir_move: Union[str, ArrayLike], 
                             distance: float, put_down: bool, 
                             last_gripper_position: ArrayLike, **kwargs):
        effects = {}
        if grasping:
            # the object is still being grasped
            effects["predicate_grasping"] = {"params": {"part_name": ""}, "value": True}
            if put_down:
                assert dir_move=="bottom", "Invalid direction for moving the gripper down."
                # the object is on the table
                effects["predicate_on_table"] = {"params": {}, "value": True}
        if touching:
            # the object is still being touched
            effects["predicate_touching"] = {"params": {"part_name": ""}, "value": True}
        
        # the object is at the target position
        if isinstance(dir_move, str):
            dir_move = np.array(SpatialSampler.gaussian_param_mappings[dir_move]['mu'])
        if not put_down:
            effects["predicate_at_position"] = {"params": {"position": np.array(last_gripper_position)+np.array(dir_move)*distance,
                                                           "position_thred": 0.05}, 
                                                "value": True}
        return effects

    @staticmethod
    def effects_rotate_obj(part_rotate: str, dir_rotate: Union[str, ArrayLike], **kwargs):
        # the object is being grasped
        effects = {}
        effects["predicate_grasping"] = {"params": {"part_name": ""}, "value": True}
        # the object is facing the target direction
        if isinstance(dir_rotate, str):
            dir_rotate = np.array(SpatialSampler.gaussian_param_mappings[dir_rotate]['mu'])
        effects["predicate_facing_direction"] = {"params": {"part_name": part_rotate, 
                                                             "direction": dir_rotate}, 
                                                  "value": True}
        return effects

    @staticmethod
    def effects_touch_obj(part_touch: str, **kwargs):
        # the object is being touched
        effects = {}
        effects["predicate_touching"] = {"params": {"part_name": part_touch}, "value": True}
        return effects

    @staticmethod
    def effects_release_obj(**kwargs):
        effects = {}
        # the object is not being grasped
        effects["predicate_grasping"] = {"params": {"part_name": ""}, "value": False}
        # the object is not being touched
        effects["predicate_touching"] = {"params": {"part_name": ""}, "value": False}
        # the gripper is open
        effects["predicate_gripper_open"] = {"params": {}, "value": True}
        # the gripper is away from the object
        effects["predicate_min_distance"] = {"params": {}, "value": True}
        return effects

    #####################
    #  Skill Functions  #
    #####################

    def grasp_obj(self, obj, part_grasp="", region_on_part="", visualize=False, retreat_height=0.25):
        """
        Grasp the object by part and region on part.
        
        Parameters:
            obj (Str): The object to be grasped.
            part_grasp (Str): The part of the object to be located as the first-level grasping position refinement. 
                Can be a part name of a locative noun. An empty string means the entire object.
            region_on_part (Str): The relative direction on the part to be grasped as the second-level grasping position refinement. 
                Can be a locative noun. Available locative nouns: front, back, top, bottom, left, right, middle.  
        """
        # update pcds
        self.parser.update_part_pcds(resample_spatial=True)
        # check preconditions
        if not self.preconditions_grasp_obj(self.env):
            print("Preconditions of grasp_obj not satisfied.")
            return False
        
        T_grasp, grasps, scores, pc = self.plan_grasps(obj, part_grasp, region_on_part)
        ret, _, _, _, _, _ = self.target_grasp(T_grasp, grasps, scores, pc=pc, visualize=visualize, retreat_height=retreat_height, part_grasp=part_grasp)
        return ret

    def move_gripper(self, dir_move: Union[str, ArrayLike], distance=0.1, grasping=False, touching=False, put_down=False, **kwargs):
        """
        Translate the grasped object towards a directional reference by a given distance.
        
        Parameters:
            dir_move (Str, Arraylike): The directional reference as an obj name or a locative noun, or the coordinates of a point.
            distance (Float): The distance to be traveled.
        """
        self.checker.reset_skill("move_gripper", {
            "dir_move": dir_move,
            "distance": distance,
            "grasping": grasping,
            "touching": touching,
            "put_down": put_down
        })
        # check preconditions
        if not self.preconditions_move_gripper(self.env, grasping, touching, put_down):
            print("Preconditions of move_gripper not satisfied.")
            return False
        init_obj_pose = self.env.obj.get_pose()
        init_tcp_pose = self.robot.get_tcp_pose()
        position = init_tcp_pose.translation
        orientation = init_tcp_pose.rotation
        ret = False
        direction = None
        target_pose = None
        if isinstance(dir_move, str):
            assert dir_move in self.parser.spatial_sampler.locative_nouns and dir_move!="middle", "Invalid direction. Available locative nouns: front, back, top, bottom, left, right."
            direction = np.array(self.parser.spatial_sampler.gaussian_param_mappings[dir_move]['mu'])
            direction = direction/np.linalg.norm(direction)
        else:
            direction = np.array(dir_move)-np.array(position)
            direction = direction/np.linalg.norm(direction)
        translation = direction*distance
        target_position = np.array(position)+translation
        T_world_tcp_t = Transform(orientation, target_position)
        if touching:
            gain_near_target = 0.05
        else:
            gain_near_target = 0.2
        ret, env_output = self.pose_tcp(T_world_tcp_t, max_iteration=100, gain=0.001, gain_near_target=gain_near_target, close_threshold=0.0005, allow_contact=True, check_collision=True, check_obj_collision=False, grasping=grasping, timeout=15)
        _, _, done, info = env_output
        if self.generation_mode and (ret or done or self.checker.is_skill_done(self.parser)):
            ret = True
            self.env.dump_buffers()
            target_pose = T_world_tcp_t
        elif not self.generation_mode and ret:
            ret = True
            self.env.dump_buffers()
            target_pose = T_world_tcp_t
        else:
            ret = False
            if not self.keep_all:
                self.env.clear_buffers()
            else:
                self.env.dump_buffers()

        print("Target translation completed.")

        return ret

    def rotate_obj(self, part_rotate, dir_rotate, grasping=True):
        """
        Rotate the grasped object such that a given part is facing a directional reference.
        
        Parameters:
            part_rotate (Str): The part to be reoriented. Either be a part name or a locative noun.
            dir_rotate (Str, Arraylike): The directional reference as an obj name or a locative noun, or coordinates of a point.
        """
        assert part_rotate in self.parser.part_pcds_t0.keys(), "Invalid part entity. Make sure to load part hierarchy."
        # check preconditions
        if not self.preconditions_rotate_obj(self.env):
            print("Preconditions of rotate_obj not satisfied.")
            return False
        # update pcds
        self.parser.update_part_pcds(resample_spatial=True)
        ret = False
        position, orientation = self.world.get_body_pose(self.env.obj)
        direction = None
        if isinstance(dir_rotate, str):
            assert dir_rotate in self.parser.spatial_sampler.locative_nouns and dir_rotate!="middle", "Invalid direction. Available locative nouns: front, back, top, bottom, left, right."
            direction = np.array(self.parser.spatial_sampler.gaussian_param_mappings[dir_rotate]['mu'])
        else:
            direction = np.array(dir_rotate)-np.array(position)
            direction = direction/np.linalg.norm(direction)
        
        grounded_pcd = self.parser.part_pcds[part_rotate]
        grounded_center =  np.mean(grounded_pcd, axis=0)

        rel_quaternion, _ = rotation_quaternion(grounded_center-np.array(position), direction)
        rel_rotation = Rotation(rel_quaternion)

        ret, _, _, _ = self.target_rotate(rel_rotation, grasping, part_rotate=part_rotate, dir_rotate=dir_rotate)
        return ret

    def touch_obj(self, obj, part_touch="", region_on_part="", bbox_extension_ratio=0.1):
        """
        Touch the object such on a part.
        
        Parameters:
            obj (Str): The object to be touched.
            part_touch (Str): The part of the object to be approached and touched. Either be a part name or a locative noun.
            region_on_part (Str): The relative direction on the part to be grasped as the second-level grasping position refinement. 
                Can be a locative noun. Available locative nouns: front, back, top, bottom, left, right, middle.  
        """
        self.checker.reset_skill("touch_obj", {
            "part_touch": part_touch,
        })
        assert obj in self.parser.part_pcds_t0.keys(), "Invalid object name. Make sure to load part hierarchy."
        assert part_touch in self.parser.part_pcds_t0.keys(), "Invalid part entity. Make sure to load part hierarchy."
        # check preconditions
        if not self.preconditions_touch_obj(self.env):
            print("Preconditions of touch_obj not satisfied.")
            return False
        # Switch gripper to close
        self.switch_gripper(self.robot.CLOSED)

        init_obj_pose = self.env.obj.get_pose()
        init_tcp_pose = self.robot.get_tcp_pose()
        # update pcds
        self.parser.update_part_pcds(resample_spatial=True)
        ret = False

        # language grounding
        obj_pcd = self.parser.part_pcds[self.parser.obj_name]
        grounded_pcd = self.parser.part_pcds[self.parser.obj_name] if part_touch=="" else self.parser.part_pcds[part_touch]
        grounded_pcd = grounded_pcd if region_on_part=="" else self.parser.spatial_sampler.sample_query(grounded_pcd, region_on_part)
        # Compute the bounding box of the grounded_pcd with an extension
        min_point = np.min(grounded_pcd, axis=0)
        max_point = np.max(grounded_pcd, axis=0)
        bbox_diagonal = np.linalg.norm(max_point - min_point)
        bbox_extension = bbox_extension_ratio * bbox_diagonal

        # Apply the adaptive extension
        min_point -= bbox_extension
        max_point += bbox_extension

        grounded_center =  np.mean(grounded_pcd, axis=0)
        obj_center = np.mean(obj_pcd, axis=0)

        # Define directions and corresponding vectors
        directions = {
            'left': np.array([0, 1, 0], dtype=float),
            'right': np.array([0, -1, 0], dtype=float),
            'front': np.array([-1, 0, 0], dtype=float),
            'back': np.array([1, 0, 0], dtype=float),
            'top': np.array([0, 0, 1], dtype=float),
        }
        
        free_directions = []

        obj_min_point = np.min(obj_pcd, axis=0)
        obj_max_point = np.max(obj_pcd, axis=0)

        for direction, vector in directions.items():
            ext_vector_obj = [np.dot(vector, obj_max_point), np.dot(vector, obj_min_point)][np.argmax([np.dot(vector, obj_max_point), np.dot(vector, obj_min_point)])]
            ext_vector_region = [np.dot(vector, max_point), np.dot(vector, min_point)][np.argmax([np.dot(vector, max_point), np.dot(vector, min_point)])]
            if ext_vector_obj<=ext_vector_region:
                free_directions.append(direction)
        pretouch_pose = None
        touch_pose = None
        if len(free_directions) > 0:
            # Calculate the closest direction to the vector from object center to grounded region center
            target_vector = grounded_center - obj_center
            best_direction = min(free_directions, key=lambda dir: np.dot(target_vector, -directions[dir]))
            # Find the outermost point in the grounded region along the best direction
            best_vector = directions[best_direction]
            dot_products = np.dot(grounded_pcd, best_vector)
            touch_point = grounded_pcd[np.argmax(dot_products)]
            # Determine the TCP orientation
            z_axis_tcp = -best_vector  # TCP z-axis is opposite to the best vector

            if best_direction != 'top':
                x_axis_world = np.array([0, 0, 1], dtype=float)
                x_axis_tcp = -x_axis_world
            else:
                possible_axes = [np.array([1, 0, 0], dtype=float), np.array([-1, 0, 0], dtype=float), np.array([0, 1, 0], dtype=float), np.array([0, -1, 0], dtype=float)]
                closest_axis = min(possible_axes, key=lambda axis: np.linalg.norm(axis - z_axis_tcp))
                x_axis_tcp = -closest_axis

            # Normalize the axes to form a proper orthogonal basis
            x_axis_tcp /= np.linalg.norm(x_axis_tcp)
            z_axis_tcp /= np.linalg.norm(z_axis_tcp)
            y_axis_tcp = np.cross(z_axis_tcp, x_axis_tcp)
            # The orientation matrix
            tcp_orientation_matrix = np.column_stack((x_axis_tcp, y_axis_tcp, z_axis_tcp))
            tcp_orientation = Rotation.from_matrix(tcp_orientation_matrix)
            # touch_point_tcp = touch_point - self.robot.finger_depth/2*z_axis_tcp
            touch_point_tcp = touch_point - (self.robot.finger_depth-self.robot.finger_depth/8)*z_axis_tcp
            if best_direction=="top":
                top_offset = np.array([0.0, 0.0, 0.0])
            else:
                top_offset = np.array([0.0, 0.0, 0.1])
            pretouch_pose = Transform(tcp_orientation, touch_point_tcp-0.05*z_axis_tcp+top_offset)
            # self.world.draw_frame_axes(pretouch_pose.translation, pretouch_pose.rotation.as_matrix())
            touch_pose = Transform(tcp_orientation, touch_point_tcp)
            # print("Before")
            # print(pretouch_pose.translation)
            self.pose_tcp(pretouch_pose, gain=0.005, max_iteration=120, norm_threshold=0.01, close_threshold=0.0001, allow_contact=False, check_obj_collision=False, gain_near_target=0.2)
            # tcp_pose = self.robot.get_tcp_pose()
            # self.world.draw_frame_axes(touch_pose.translation, touch_pose.rotation.as_matrix())
            # print("Executed")
            # print(tcp_pose.translation)
            ret_planner, env_output = self.pose_tcp(touch_pose, gain=0.005, norm_threshold=0.005, max_iteration=120, close_threshold=0.0001, allow_contact=False, check_collision=True, check_obj_collision=False, gain_near_target=0.2)
            # tcp_pose = self.robot.get_tcp_pose()
            # self.world.draw_frame_axes(tcp_pose.translation, tcp_pose.rotation.as_matrix())
            _, _, done, info = env_output
            if self.generation_mode and (done or self.checker.is_skill_done(self.parser)):
                ret = True
                self.env.dump_buffers()
            elif not self.generation_mode and ret_planner:
                ret = True
                self.env.dump_buffers()
            else:
                ret = False
                if not self.keep_all:
                    self.env.clear_buffers()
                else:
                    self.env.dump_buffers()
        else:
            ret = False

        return ret
    
    def release_obj(self):
        """
        Release the gripper and move away from the object.
        """
        self.checker.reset_skill("release_obj", {})
        # check preconditions
        if not self.preconditions_release_obj(self.env):
            print("Preconditions of release_obj not satisfied.")
            return False
        # Switch gripper to open
        self.switch_gripper(self.robot.OPEN, abort_on_contact=False)
        init_obj_pose = self.env.obj.get_pose()
        init_tcp_pose = self.robot.get_tcp_pose()
        init_tcp_position = init_tcp_pose.translation
        init_tcp_orientation = init_tcp_pose.rotation.as_matrix()
        # update pcds
        self.parser.update_part_pcds(resample_spatial=True)
        ret = False

        # Define the distance to move along the -z axis of the TCP coordinate
        release_distance_tcp_z = self.env.config.release_distance_tcp_z
        # Calculate the target position along the -z axis of the TCP coordinate
        target_tcp_position = init_tcp_position + release_distance_tcp_z * init_tcp_orientation[:, 2]

        # Move the robot to the target position along the -z axis of the TCP coordinate
        new_tcp_pose = Transform(init_tcp_pose.rotation, target_tcp_position)
        # # Calculate the angle between the TCP z-axis and the world z-axis
        # angle_between_z_axes = np.arccos(np.dot(tcp_z_axis_world, world_z_axis))

        # # If the angle is not zero (not parallel), move a little along the +z axis of the world frame
        # if angle_between_z_axes > 1e-3:
        ret_planner, env_output = self.pose_tcp(new_tcp_pose, gain=0.005, gain_near_target=0.1, norm_threshold=0.002, close_threshold=0.0001, max_iteration=120)
        # Get the new TCP pose after the movement
        new_tcp_pose = self.robot.get_tcp_pose()
        new_tcp_position = new_tcp_pose.translation

        # Check if the z-axis of the TCP is not parallel to the z-axis of the world frame
        tcp_z_axis_world = init_tcp_orientation[:, 2]
        world_z_axis = np.array([0, 0, 1])

        # If the angle is not zero (not parallel), move a little along the +z axis of the world frame
        # if angle_between_z_axes > 1e-3:  # Allowing for a small numerical tolerance
        release_distance_world_z = self.env.config.release_distance_world_z
        target_world_position = new_tcp_position + release_distance_world_z * world_z_axis

        # Move the robot to the target position along the +z axis of the world frame
        new_tcp_pose = Transform(new_tcp_pose.rotation, target_world_position)
        ret_planner, env_output = self.pose_tcp(new_tcp_pose, gain=0.005, norm_threshold=0.01, close_threshold=0.0001, max_iteration=120)

        _, _, done, info = env_output
        # print(done)
        # print(info)
        if self.generation_mode and (done or self.checker.is_skill_done(self.parser)):
            ret = True
            self.env.dump_buffers()
        elif not self.generation_mode and ret_planner:
            ret = True
            self.env.dump_buffers()
        else:
            ret = False
            if not self.keep_all:
                self.env.clear_buffers()
            else:
                self.env.dump_buffers()
        
        return ret
    
    #####################
    #  Motion Planners  #
    #####################
    @staticmethod
    def plan_direct_motion_fn(robot, fixed=[], num_attempts=50, **kwargs):
        movable_joints = get_movable_joints(robot)
        sample_fn = get_sample_fn(robot, movable_joints)
        def fn(world, body, q_cur, trgt_pose, **kwargs):
            # saved_world = WorldSaver()
            trgt_pose = Grasp(trgt_pose).to_planning_grasp(body, robot, get_tool_link(robot)).grasp_pose
            # obstacles = fixed
            obstacles = fixed
            for i in range(num_attempts):
                set_joint_positions(robot, movable_joints, sample_fn())
                conf = BodyConf(robot, q_cur)
                q_trgt = inverse_kinematics(robot, get_tool_link(robot), trgt_pose)
                if q_trgt is not None:
                    q_trgt_list = list(q_trgt)
                    q_trgt_list[-2:] = q_cur[-2:]
                    q_trgt = tuple(q_trgt_list)
                print("Attempt {}: q_trgt: {}".format(i, q_trgt))
                if q_trgt is None:
                    print("IK failed for target pose on attempt", i)
                    continue
                conf.assign()
                path = plan_direct_joint_motion(robot, conf.joints, q_trgt, obstacles=obstacles)
                print("path", path)
                if path is None:
                    print('Direct motion failed!')
                    continue
                return path
            return None
        return fn

    @staticmethod
    def plan_free_motion_fn(robot, fixed=[], num_attempts=50, self_collisions=True, **kwargs):
        movable_joints = get_movable_joints(robot)
        sample_fn = get_sample_fn(robot, movable_joints)
        def fn(world, body, q_cur, trgt_pose, **kwargs):
            trgt_pose = Grasp(trgt_pose).to_planning_grasp(body, robot, get_tool_link(robot)).grasp_pose
            # obstacles = fixed
            obstacles = fixed + [body]
            for i in range(num_attempts):
                set_joint_positions(robot, movable_joints, sample_fn())
                conf = BodyConf(robot, q_cur)
                q_trgt = inverse_kinematics(robot, get_tool_link(robot), trgt_pose)
                if q_trgt is not None:
                    q_trgt_list = list(q_trgt)
                    q_trgt_list[-2:] = q_cur[-2:]
                    q_trgt = tuple(q_trgt_list)
                print("Attempt {}: q_trgt: {}".format(i, q_trgt))
                if q_trgt is None:
                    print("IK failed for target pose on attempt", i)
                    continue
                conf.assign()
                path = plan_joint_motion(
                    robot,
                    conf.joints,
                    q_trgt,
                    obstacles=obstacles,
                    self_collisions=self_collisions
                )
                print("path", path)
                if path is None:
                    print('Free-motion planning failed for Franka!')
                    continue
                return path
            return None
        return fn

    @staticmethod
    def plan_holding_motion_fn(robot, fixed=[], num_attempts=50, self_collisions=False, **kwargs):
        movable_joints = get_movable_joints(robot)
        sample_fn = get_sample_fn(robot, movable_joints)
        def fn(world, body, q_cur, trgt_pose, grasp_pose=None):
            world.p.setCollisionFilterPair(robot, body, linkIndexA=-1, linkIndexB=-1, enableCollision=0)
            trgt_pose = Grasp(trgt_pose).to_planning_grasp(body, robot, get_tool_link(robot)).grasp_pose
            obstacles = fixed
            grasp = Grasp(grasp_pose).to_planning_grasp(body, robot, get_tool_link(robot))
            grasp_body = copy.deepcopy(grasp)
            pose0 = BodyPose(body).pose
            grasp_body.grasp_pose = multiply(invert(grasp_body.grasp_pose), pose0)
            grasp_body.approach_pose = multiply(invert(grasp_body.approach_pose), pose0)
            for i in range(num_attempts):
                set_joint_positions(robot, movable_joints, sample_fn())
                conf = BodyConf(robot, q_cur)
                q_trgt = inverse_kinematics(robot, get_tool_link(robot), trgt_pose)
                if q_trgt is not None:
                    q_trgt_list = list(q_trgt)
                    q_trgt_list[-2:] = (0.0, 0.0)
                    q_trgt = tuple(q_trgt_list)
                print("Attempt {}: q_trgt: {}".format(i, q_trgt))
                if q_trgt is None:
                    print("IK failed for target pose on attempt", i)
                    continue
                conf.assign()
                attachment = grasp_body.attachment()
                path = plan_joint_motion(
                    robot,
                    conf.joints,
                    q_trgt,
                    obstacles=obstacles,
                    attachments=[attachment],
                    self_collisions=self_collisions
                )
                print("path", path)
                if path is None:
                    continue
                return path
            return None
        return fn

class OracleChecker:
    ALL_SKILLS = ["grasp_obj", "move_gripper", "rotate_obj", "touch_obj", "release_obj"]

    def __init__(self, env):
        self.env = env
        self.current_skill = None
        self.chain_params = None

    def reset_skill(self, skill_name: str, params: dict):
        assert skill_name in self.ALL_SKILLS
        self.current_skill = skill_name
        self.params = params
        self.initial_obj_position = self.env.obj.get_pose()
    
    def get_current_skill(self):
        if self.current_skill is None:
            raise RuntimeError("Skill not set.")
        return self.current_skill

    def is_skill_done(self, parser: Optional[SemanticParser]=None):
        if parser is None:
            parser = self.env.parser
        if "distance" not in self.params:
            self.params.update({
                "distance": self.env.config.translate_distance
            })
        effects = getattr(BulletPlanner, f"effects_{self.current_skill}")(last_gripper_position=self.initial_obj_position.translation, 
                                                                        **self.params)

        effects_satisfied = True
        
        for predicate, items in effects.items():
            result = getattr(BulletPlanner, predicate)(self.env, parser=parser, **items["params"])
            if result != items["value"]:
                effects_satisfied = False
                break
        
        return effects_satisfied
