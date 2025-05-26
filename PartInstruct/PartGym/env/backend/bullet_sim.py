import time

import numpy as np
import pybullet
import cv2
import os
from pybullet_utils import bullet_client
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

from PartInstruct.PartGym.env.backend.utils.transform import Rotation, Transform

class Client(bullet_client.BulletClient):
    def __init__(self, connection_mode=None, options=""):
        self._shapes = {}
        if connection_mode is None:
            self._client = pybullet.connect(pybullet.SHARED_MEMORY)
            if self._client >= 0:
                return
            else:
                connection_mode = pybullet.DIRECT
        self._client = pybullet.connect(connection_mode, options=options)
        

class BtWorld(object):
    """Interface to a PyBullet physics server."""

    def __init__(self, gui=True, control_hz=20.0):

        self.fps = 240.0
        self.control_hz = control_hz
        self.n_steps_per_control = int(self.fps / self.control_hz)
        self.gui = gui
        self.dt = 1.0 / self.fps
        self.solver_iterations = 150

        connection_mode = pybullet.GUI if gui else pybullet.DIRECT
        self.p = bullet_client.BulletClient(connection_mode)

        self.recording = False
        self.recording_camera_list = []

        self.reset()

    def set_gravity(self, gravity):
        self.p.setGravity(*gravity)

    def load_urdf(self, urdf_path, pose, scale=1.0, flags=None, useFixedBase=False):
        if flags is not None:
            body = Body.from_urdf(self.p, urdf_path, pose, scale, flags, useFixedBase=useFixedBase)
        else:
            body = Body.from_urdf(self.p, urdf_path, pose, scale, useFixedBase=useFixedBase)
        self.bodies[body.uid] = body
        return body

    def remove_body(self, body):
        self.p.removeBody(body.uid)
        del self.bodies[body.uid]

    def check_body_collision(self, body1, body2, margin=0.001):
        closest_points = self.p.getClosestPoints(body1.uid, body2.uid, margin)
        return len(closest_points) > 0
    
    def check_object_link_collision(self, objectUid, bodyUid, linkIndex):
        contacts = self.p.getContactPoints(bodyA=objectUid, bodyB=bodyUid)
        for contact in contacts:
            if contact[4] == linkIndex:
                return True  # Collision detected
        return False  # No collision detected

    def check_link_contact(self, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB):
        contacts = self.p.getContactPoints(bodyA=bodyUniqueIdA, bodyB=bodyUniqueIdB)
        for contact in contacts:
            if contact[3] == linkIndexA and contact[4] == linkIndexB:
                return True  # Contact detected
        return False  # No contact detected
    
    def check_obj_gripper_collision(self, bodyUid, bodyLinkUid, point_cloud, threshold_distance=0.01, force_threshold=20.0):
        contacts = self.get_link_contacts(bodyA=bodyUid, linkIndexA=bodyLinkUid)
        ret_force = False
        for contact in contacts:
            contact_normal_force = np.array(contact[9])
            if np.linalg.norm(contact_normal_force) > force_threshold:
                ret_force = True
                break
        for contact in contacts:
            contact_point = np.array(contact[5])
            distances = np.linalg.norm(point_cloud - contact_point, axis=1)
            # print(np.min(distances))
            if ret_force and (np.min(distances) < threshold_distance):
                return True
        return False

    def check_point_cloud_gripper_collision(self, robot, point_cloud, threshold_distance=0.01):
        fingertip_pose: Transform = robot.get_fingertip_pose()
        # self.draw_frame_axes(fingertip_pose.translation, fingertip_pose.rotation.as_matrix())
        distances = np.linalg.norm(point_cloud - fingertip_pose.translation, axis=1)
        if np.min(distances) < threshold_distance:
            return True  # A point in the point cloud is close enough to a contact point
        return False  # No point in the point cloud is close enough to a contact point
    
    def add_constraint(self, *argv, **kwargs):
        """See `Constraint` below."""
        constraint = Constraint(self.p, *argv, **kwargs)
        return constraint

    def add_camera(self, intrinsic, near, far):
        camera = Camera(self.p, intrinsic, near, far, control_hz=self.control_hz)
        return camera

    def start_recording(self, camera_list, save_dir_list=None, name_list=None, save_depth_list=None, save_segmentation_list=None):
        self.recording = True
        self.recording_camera_list = camera_list
        if save_dir_list is None:
            save_dir_list = ["."]*len(camera_list)
        if name_list is None:
            name_list = list(map(str, range(len(camera_list))))
        if save_depth_list is None:
            save_depth_list = [False]*len(camera_list)
        if save_segmentation_list is None:
            save_segmentation_list = [False]*len(camera_list)
        assert len(camera_list)==len(save_depth_list)==len(name_list)==len(save_depth_list)==len(save_segmentation_list)

        for cam, save_dir, name, save_depth, save_seg in zip(camera_list, save_dir_list, name_list, save_depth_list, save_segmentation_list):
            cam.start_recording(save_dir, name, save_depth, save_seg)

    def stop_recording(self):
        self.recording = False    
        for cam in self.recording_camera_list:
            cam.stop_recording()
        self.recording_camera_list = []

    def get_contacts(self, bodyA):
        points = self.p.getContactPoints(bodyA=bodyA.uid)
        contacts = []
        for point in points:
            contact = Contact(
                bodyA=self.bodies[point[1]],
                bodyB=self.bodies[point[2]],
                point=point[5],
                normal=point[7],
                depth=point[8],
                force=point[9],
            )
            contacts.append(contact)
        return contacts
    
    def get_link_contacts(self, bodyA, linkIndexA):
        points = self.p.getContactPoints(bodyA=bodyA, linkIndexA=linkIndexA)
        return points
    
    def get_body_pose(self, obj):
        position, orientation = self.p.getBasePositionAndOrientation(obj.uid)
        return position, orientation

    def reset(self):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt, numSolverIterations=self.solver_iterations
        )
        self.bodies = {}
        self.sim_time = 0.0

    def step(self):
        for _ in range(self.n_steps_per_control):
            self.p.stepSimulation()
            self.sim_time += self.dt

        if self.recording:
            # Use ThreadPoolExecutor to process each camera's recording in parallel
            with ThreadPoolExecutor() as executor:
                # Create a list of futures
                futures = [executor.submit(cam.buffer_recording) for cam in self.recording_camera_list]
                # Optionally, wait for all futures to complete if needed
                for future in futures:
                    future.result() 

    def save_state(self):
        return self.p.saveState()

    def restore_state(self, state_uid):
        self.p.restoreState(stateId=state_uid)

    def close(self):
        self.p.disconnect()

    # def __del__(self):
    #     pybullet.disconnect(self.p._client)

    def draw_frame_axes(self, frame_pos, frame_orn, axis_length=0.1, line_width=1.5):
        # Convert quaternion to rotation matrix to get axes directions
        rot_matrix = np.array(frame_orn).reshape(3,3)

        frame_pos = np.array(frame_pos)

        # Define axis directions in the frame's local coordinates
        x_axis = rot_matrix[:, 0] * axis_length
        y_axis = rot_matrix[:, 1] * axis_length
        z_axis = rot_matrix[:, 2] * axis_length

        # Draw X, Y, Z axes
        self.p.addUserDebugLine(frame_pos, frame_pos + x_axis, [1, 0, 0], line_width)
        self.p.addUserDebugLine(frame_pos, frame_pos + y_axis, [0, 1, 0], line_width)
        self.p.addUserDebugLine(frame_pos, frame_pos + z_axis, [0, 0, 1], line_width)


class Body(object):
    """Interface to a multibody simulated in PyBullet.

    Attributes:
        uid: The unique id of the body within the physics server.
        name: The name of the body.
        joints: A dict mapping joint names to Joint objects.
        links: A dict mapping link names to Link objects.
    """

    def __init__(self, physics_client, body_uid, pose):
        self.p = physics_client
        self.uid = body_uid
        self.name = self.p.getBodyInfo(self.uid)[1].decode("utf-8")
        self.joints, self.links = {}, {}
        self.init_pose = pose
        for i in range(self.p.getNumJoints(self.uid)):
            joint_info = self.p.getJointInfo(self.uid, i)
            joint_name = joint_info[1].decode("utf8")
            self.joints[joint_name] = Joint(self.p, self.uid, i)
            link_name = joint_info[12].decode("utf8")
            self.links[link_name] = Link(self.p, self.uid, i)

    @classmethod
    def from_urdf(cls, physics_client, urdf_path, pose, scale, flags=None, useFixedBase=False):
        if flags is not None:
            body_uid = physics_client.loadURDF(
                str(urdf_path),
                pose.translation,
                pose.rotation.as_quat(),
                globalScaling=scale,
                flags=flags,
                useFixedBase=useFixedBase,
            )
        else:
            body_uid = physics_client.loadURDF(
                str(urdf_path),
                pose.translation,
                pose.rotation.as_quat(),
                globalScaling=scale,
                useFixedBase=useFixedBase,
            )
        return cls(physics_client, body_uid, pose)
    
    def reset(self):
        self.set_pose(self.init_pose)

    def get_pose(self):
        pos, ori = self.p.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(ori), np.asarray(pos))

    def set_pose(self, pose):
        self.p.resetBasePositionAndOrientation(
            self.uid, pose.translation, pose.rotation.as_quat()
        )

    def get_velocity(self):
        linear, angular = self.p.getBaseVelocity(self.uid)
        return linear, angular
    
    def get_bbox(self):
        aabb = self.p.getAABB(self.uid)
        # print(aabb)
        # Extract the min and max corners
        aabb_min = np.array(aabb[0])
        # print(aabb_min)
        aabb_max = np.array(aabb[1])
        # print(aabb_max)

        # Calculate the length of the diagonal
        diagonal_length = np.linalg.norm(aabb_max - aabb_min)
        return aabb_min, aabb_max, diagonal_length


class Link(object):
    """Interface to a link simulated in Pybullet.

    Attributes:
        link_index: The index of the joint.
    """

    def __init__(self, physics_client, body_uid, link_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.link_index = link_index

    def get_pose(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos, ori = link_state[0], link_state[1]
        return Transform(Rotation.from_quat(ori), pos)


class Joint(object):
    """Interface to a joint simulated in PyBullet.

    Attributes:
        joint_index: The index of the joint.
        lower_limit: Lower position limit of the joint.
        upper_limit: Upper position limit of the joint.
        effort: The maximum joint effort.
    """

    def __init__(self, physics_client, body_uid, joint_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.joint_index = joint_index

        joint_info = self.p.getJointInfo(body_uid, joint_index)
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.effort = joint_info[10]

    def get_position(self):
        joint_state = self.p.getJointState(self.body_uid, self.joint_index)
        return joint_state[0]

    def set_position(self, position, kinematics=False):
        if kinematics:
            self.p.resetJointState(self.body_uid, self.joint_index, position)
        self.p.setJointMotorControl2(
            self.body_uid,
            self.joint_index,
            pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.effort,
        )


class Constraint(object):
    """Interface to a constraint in PyBullet.

    Attributes:
        uid: The unique id of the constraint within the physics server.
    """

    def __init__(
        self,
        physics_client,
        parent,
        parent_link,
        child,
        child_link,
        joint_type,
        joint_axis,
        parent_frame,
        child_frame,
    ):
        """
        Create a new constraint between links of bodies.

        Args:
            parent:
            parent_link: None for the base.
            child: None for a fixed frame in world coordinates.

        """
        self.p = physics_client
        parent_body_uid = parent.uid
        parent_link_index = parent_link.link_index if parent_link else -1
        child_body_uid = child.uid if child else -1
        child_link_index = child_link.link_index if child_link else -1

        self.uid = self.p.createConstraint(
            parentBodyUniqueId=parent_body_uid,
            parentLinkIndex=parent_link_index,
            childBodyUniqueId=child_body_uid,
            childLinkIndex=child_link_index,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=parent_frame.translation,
            parentFrameOrientation=parent_frame.rotation.as_quat(),
            childFramePosition=child_frame.translation,
            childFrameOrientation=child_frame.rotation.as_quat(),
        )

    def change(self, **kwargs):
        self.p.changeConstraint(self.uid, **kwargs)


class Contact(object):
    """Contact point between two multibodies.

    Attributes:
        point: Contact point.
        normal: Normal vector from ... to ...
        depth: Penetration depth
        force: Contact force acting on body ...
    """

    def __init__(self, bodyA, bodyB, point, normal, depth, force):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.point = point
        self.normal = normal
        self.depth = depth
        self.force = force


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, physics_client, intrinsic, near, far, extrinsic=None, control_hz=20.0):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.p = physics_client
        self.recording_step = 0
        self.control_hz = control_hz

    def set_extrinsic(self, extrinsic):
        self.extrinsic = extrinsic

    def render(self, extrinsic, renderer=pybullet.ER_TINY_RENDERER):
        """Render synthetic RGB, depth images, and segmentation masks.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=renderer,
        )

        # rgb, z_buffer = result[2][:, :, :3], result[3]
        rgb = np.array(result[2]).reshape(result[1], result[0], -1)[:, :, :3]
        z_buffer = np.array(result[3]).reshape(result[1], result[0])
        segmentation_mask = np.array(result[4]).reshape(result[1], result[0])

        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )
        return rgb, depth.astype(np.float32), segmentation_mask
    
    def semantic_render(self, extrinsic, obj_uid, surface_uid=None):
        """Render synthetic RGB and masked depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )

        # rgb, z_buffer = result[2][:, :, :3], result[3]
        rgb = np.array(result[2]).reshape(result[1], result[0], -1)[:, :, :3]
        z_buffer = np.array(result[3]).reshape(result[1], result[0])
        segmentation_mask_buffer = np.array(result[4]).reshape(result[1], result[0])
        if surface_uid is not None:
            object_mask = (segmentation_mask_buffer == obj_uid)
            surface_mask = (segmentation_mask_buffer == surface_uid)
            union_mask = np.logical_or(object_mask, surface_mask)
        else:
            union_mask = segmentation_mask_buffer == obj_uid
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )
        masked_depth = np.where(union_mask, depth, 0)
        return rgb, masked_depth.astype(np.float32)
    
    def start_recording(self, save_dir, name="video", save_depth=False, save_segmentation=False):
        self.record_name = name
        self.save_dir = save_dir
        self.save_depth = save_depth
        self.save_segmentation = save_segmentation
        self.frames_rgb = []
        self.frames_depth = []
        self.frames_segmentation = []
        self.recording_step = 0
        if self.save_depth:
            self.depth_dir = os.path.join(self.save_dir, self.record_name + '_depth')
            if os.path.exists(self.depth_dir):
                shutil.rmtree(self.depth_dir)
            os.makedirs(self.depth_dir, exist_ok=True)
        if self.save_segmentation:
            self.segmentation_dir = os.path.join(self.save_dir, self.record_name + '_segmentation')
            if os.path.exists(self.segmentation_dir):
                shutil.rmtree(self.segmentation_dir)
            os.makedirs(self.segmentation_dir, exist_ok=True)

    def buffer_recording(self, cam_extrinsic=None):
        if cam_extrinsic is None:
            cam_extrinsic = self.extrinsic
        assert cam_extrinsic is not None
        rgb, depth, segmentation = self.render(cam_extrinsic, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL) 
        self.frames_rgb.append(rgb)
        if self.save_depth:
            # depth_image_path = os.path.join(self.depth_dir, f'depth_{self.recording_step:04d}.png')
            # save_image(depth_image_path, depth)
            self.frames_depth.append(depth)
        if self.save_segmentation:
            # segmentation_image_path = os.path.join(self.segmentation_dir, f'segmentation_{self.recording_step:04d}.png')
            # save_image(segmentation_image_path, segmentation)
            self.frames_segmentation.append(segmentation)
        self.recording_step+=1

    def stop_recording(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
        out_rgb = cv2.VideoWriter(os.path.join(self.save_dir, self.record_name+'.mp4'), fourcc, self.control_hz, (self.intrinsic.width,self.intrinsic.height))
        # Save RGB video
        for frame in self.frames_rgb:
            out_rgb.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out_rgb.release()

        with ThreadPoolExecutor() as executor:
            if self.save_depth:
                depth_dir = os.path.join(self.save_dir, self.record_name + '_depth')
                for i, depth_frame in enumerate(self.frames_depth):
                    depth_image_path = os.path.join(depth_dir, f'depth_{i:04d}.png')
                    executor.submit(save_image, depth_image_path, depth_frame)

            if self.save_segmentation:
                segmentation_dir = os.path.join(self.save_dir, self.record_name + '_segmentation')
                for i, segmentation_frame in enumerate(self.frames_segmentation):
                    segmentation_image_path = os.path.join(segmentation_dir, f'segmentation_{i:04d}.png')
                    executor.submit(save_image, segmentation_image_path, segmentation_frame)

        # Clearing frames after saving
        self.frames_rgb = []
        self.frames_depth = []
        self.frames_segmentation = []


def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho

def _build_intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([
                [fx, 0, px],
                [0, fy, py],
                [0, 0, 1]
            ])

def _build_intrinsic_from_proj_matrix(proj_matrix, img_width, img_height):
    # Reshape the projection matrix to 4x4
    proj_matrix = np.array(proj_matrix).reshape(4, 4, order="F")

    # Elements of the projection matrix
    a = proj_matrix[0, 0]
    c = proj_matrix[1, 1]
    b = proj_matrix[0, 2]
    d = proj_matrix[1, 2]

    # Compute focal lengths in pixel units
    f_x = a * img_width / 2.0
    f_y = c * img_height / 2.0

    # Compute the principal point coordinates
    c_x = (b + 1) * img_width / 2.0
    c_y = (d + 1) * img_height / 2.0

    # Construct the intrinsic matrix
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])
    
    return K

def save_image(image_path, image_data):
    cv2.imwrite(image_path, image_data)

def _save_npz(file_path, data):
    np.savez_compressed(file_path, data)

def save_depth(file_path, data, depth_scale=1000.0):
    data = data * depth_scale
    data = np.nan_to_num(data)
    data = data.astype(np.uint16)
    cv2.imwrite(file_path, data)

def _euler_angles_to_rotation_matrix(yaw, pitch, roll=0):
    # Assuming angles are in degrees and converting them to radians
    yaw, pitch, roll = np.radians([yaw, pitch, roll])
    
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
                    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
                    
    # Combining the rotations around each axis
    R = R_z @ R_y @ R_x
    
    return R

def _build_extrinsic_matrix(camera_target_position, camera_distance, camera_yaw, camera_pitch):
    # Convert yaw and pitch from degrees to radians
    yaw = np.radians(camera_yaw)
    pitch = np.radians(camera_pitch)
    
    # Extract target position components
    Tx, Ty, Tz = camera_target_position
    
    # Compute the view matrix elements using the given formula
    view_matrix = np.array([
        [-np.sin(yaw), np.cos(yaw), 0, Tx * np.sin(yaw) - Ty * np.cos(yaw)],
        [-np.sin(pitch) * np.cos(yaw), -np.sin(pitch) * np.sin(yaw), np.cos(pitch), np.sin(pitch) * (Tx * np.cos(yaw) + Ty * np.sin(yaw)) - Tz * np.cos(pitch)],
        [-np.cos(pitch) * np.cos(yaw), -np.cos(pitch) * np.sin(yaw), -np.sin(pitch), np.cos(pitch) * (Tx * np.cos(yaw) + Ty * np.sin(yaw)) + Tz * np.sin(pitch)],
        [0, 0, 0, 1]
    ])
    
    return view_matrix

def _look_at_to_extrinsic(eye, target, up):
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)
    
    forward = eye - target
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    
    true_up = np.cross(forward, right)
    
    R = np.vstack([right, true_up, forward])
    t = -R @ eye
    
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t
    
    return extrinsics

def _remove_items_with_string(root_dir, search_str):
    """
    Removes all files and folders under the root directory that contain the search_str in their name.

    Args:
    - root_dir (str): The root directory to search in.
    - search_str (str): The string to search for in file and folder names.
    """
    root = Path(root_dir)
    
    if not root.exists():
        print(f"The specified root directory {root_dir} does not exist.")
        return
    
    for item in root.rglob('*'):  # Recursively search for all items under root
        if search_str in item.name:
            try:
                if item.is_dir():
                    shutil.rmtree(item)  # Remove the directory and all its contents
                    print(f"Removed directory: {item}")
                else:
                    item.unlink()  # Remove the file
                    print(f"Removed file: {item}")
            except Exception as e:
                print(f"Error removing {item}: {e}")