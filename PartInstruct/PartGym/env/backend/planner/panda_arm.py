from pathlib import Path
import numpy as np
import pybullet
import math
import os
from PartInstruct.PartGym.env.backend.utils.transform import *
from PartInstruct.PartGym.env.backend.utils.perception import *

class PandaArm(object):
  
    FLANGEIDX = 7
    LEFTFINGERIDX = 9
    RIGHTFINGERIDX = 10
    ENDEFFECTORIDX = 11

    NUM_DOFS = 7

    # restposes 
    # HOME_JOINT_POS=[0.920, 0.086, 0.413, -2.163, -0.051, 2.242, -0.164, 0.04, 0.04] 
    # HOME_JOINT_POS=[0.920, 0.286, 0.413, -1.763, -0.051, 2.242, -0.164, 0.04, 0.04] 
    # HOME_JOINT_POS=[0.920, 0.286, 0.413, -1.963, -0.051, 2.242, -0.164, 0.04, 0.04] 
    # HOME_JOINT_POS=[0.920, 0.286, 0.713, -1.963, -0.051, 2.042, -0.164, 0.04, 0.04] 
    HOME_JOINT_POS=[0.920, 0.286, 0.713, -1.963, -0.051, 2.042, -0.164, 0.04, 0.04] 

    # BASE_ORN = [0.0, 0.0, 0.707, 0.707] # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
    # BASE_ORN = [0.0, 0.0, 0.38268, 0.92388] # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
    BASE_ORN = [0.0, 0.0, 0.0, 1.0] # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])

    # lower limits
    LL = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    CLL = list(np.array(LL)*0.8)
    # upper limits 
    UL = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
    CUL = list(np.array(UL)*0.8)
    # joint ranges 
    JR = [5.7946, 3.5256, 5.7946, 3.002, 5.7946, 3.77, 5.7946]
    JC = list((np.array(LL) + np.array(UL)) / 2.0)

    DH = [
        [0, 0, 0.333, 'theta1'],
        [0, -np.pi/2, 0, 'theta2'],
        [0, np.pi/2, 0.316, 'theta3'],
        [0.0825, np.pi/2, 0, 'theta4'],
        [-0.0825, -np.pi/2, 0.384, 'theta5'],
        [0, np.pi/2, 0, 'theta6'],
        [0.088, np.pi/2, 0, 'theta7'],
        [0, 0, 0.107, 0] 
    ]

    GRIPPER_OPEN_JOINT_POS = 0.04
    GRIPPER_CLOSED_JOINT_POS = 0.00

    OPEN = 1
    CLOSED = 0
  
    def __init__(self, world, offset, urdf_path="data/urdfs/robots/franka_panda/panda.urdf"):
        self.world = world
        self.bullet_client = world.p
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)

        # gripper related 
        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_gripper_tcp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        self.T_flange_gripper = Transform(Rotation.identity(), [0.0, 0.0, 0.0])
        self.T_tcp_gripper = self.T_gripper_tcp.inverse()
        
        self.T_tcp_camera = Transform(Rotation.from_matrix(rotation_matrix_z(90)), [0.05, 0, -0.07])
        # wrist camera
        intrinsic = CameraIntrinsic(300, 300, 200.0, 200.0, 150.0, 150.0)
        self.wrist_camera = self.world.add_camera(intrinsic, 0.001, 10.0)

        self.control_dt = 1./240.
        self.gripper_height = 0.2
        self.flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | self.bullet_client.URDF_USE_SELF_COLLISION
        # print(self.BASE_ORN)
        self.panda = self.bullet_client.loadURDF(urdf_path, self.offset, self.BASE_ORN, useFixedBase=True, flags=self.flags)
        self.reset()
        self.update_wrist_camera_extrinsic()

        # ## DEBUG
        # rgb, _, _ = self.wrist_camera.render(self.wrist_camera.extrinsic, pybullet.ER_BULLET_HARDWARE_OPENGL)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("test.png", rgb)

    def update_wrist_camera_extrinsic(self):
        results = self.world.p.getLinkState(self.panda, self.ENDEFFECTORIDX)
        T_world_gripper = Transform(Rotation(list(results[1])), list(results[0]))
        T_world_camera = T_world_gripper*self.T_gripper_tcp*self.T_tcp_camera
        # self.world.draw_frame_axes(T_world_camera.translation, T_world_camera.rotation.as_matrix())
        extrinsic = T_world_camera.inverse()
        self.wrist_camera.set_extrinsic(extrinsic)

    def forward_kinematics(self, joint_values):
        # Initialize the final transformation matrix as identity
        T = np.eye(4)
        
        # Replace 'theta#' placeholders with actual joint values
        for i in range(len(joint_values)):
            self.DH[i][3] = joint_values[i]
        
        # Compute the overall transformation matrix
        for i, (a, alpha, d, theta) in enumerate(self.DH):
            T = T @ dh_transform(a, alpha, d, theta)
        FK = Transform.from_matrix(T)
        FK = FK*self.T_flange_gripper*self.T_gripper_tcp
        return FK
    
    def reset(self, reset_joint_states=None):
        if reset_joint_states is None:
            reset_joint_states = self.HOME_JOINT_POS
        index = 0
        # create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(self.panda,
                        9,
                        self.panda,
                        10,
                        jointType=self.bullet_client.JOINT_GEAR,
                        jointAxis=[1, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
    
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            #print("info=",info)
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, reset_joint_states[index]) 
                index=index+1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, reset_joint_states[index]) 
                index=index+1
        self.t = 0.

        # Disable collision check between the two fingers
        self.world.p.setCollisionFilterPair(self.panda, self.panda, self.LEFTFINGERIDX, self.RIGHTFINGERIDX, 0)

        # Calculate a fixed transformation
        results = self.world.p.getLinkState(self.panda, self.FLANGEIDX)
        T_world_flange = Transform(Rotation(list(results[1])), list(results[0]))

        results = self.world.p.getLinkState(self.panda, self.ENDEFFECTORIDX)
        T_world_gripper = Transform(Rotation(list(results[1])), list(results[0]))

        self.T_flange_gripper = T_world_flange.inverse()*T_world_gripper

        # Reset the control to be at the reset joint position
        for i in range(self.NUM_DOFS):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, reset_joint_states[i], positionGain=0.03, force=10 * 240.)
            self.update_wrist_camera_extrinsic()
            self.world.step()

        for i in [9,10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, self.GRIPPER_OPEN_JOINT_POS, force=5000)
            self.world.step()
    
    def check_collision(self, uid_list: list, start_link_name="panda_link1", end_link_name="panda_rightfinger"):
        # Get the indices of the start and end links
        num_joints = self.world.p.getNumJoints(self.panda)
        start_link_index = None
        end_link_index = None

        for joint_index in range(num_joints):
            joint_info = self.world.p.getJointInfo(self.panda, joint_index)
            link_name = joint_info[12].decode("utf-8")
            
            if link_name == start_link_name:
                start_link_index = joint_index
            elif link_name == end_link_name:
                end_link_index = joint_index
                break

        if start_link_index is None or end_link_index is None:
            print("Start or end link not found.")
            return False

        # Check for collisions between the specified range of links and other objects
        for link_index in range(start_link_index, end_link_index + 1):
            for body_uid in uid_list:
                contact_points = self.world.p.getContactPoints(bodyA=self.panda, linkIndexA=link_index, bodyB=body_uid)
                if contact_points:
                    return True  # Collision detected
        
        return False  # No collisions detected

    def check_self_collision(self, start_link_name="panda_link1", end_link_name="panda_link8"):
        # Get the indices of the start and end links
        num_joints = self.world.p.getNumJoints(self.panda)
        start_link_index = None
        end_link_index = None
        
        for joint_index in range(num_joints):
            joint_info = self.world.p.getJointInfo(self.panda, joint_index)
            link_name = joint_info[12].decode("utf-8")
            
            if link_name == start_link_name:
                start_link_index = joint_index
            elif link_name == end_link_name:
                end_link_index = joint_index
                break

        if start_link_index is None or end_link_index is None:
            print("Start or end link not found.")
            return False
        
        # Check for collisions between the specified range of links
        for i in range(start_link_index, end_link_index + 1):
            for j in range(i + 1, end_link_index + 1):
                contact_points = self.world.p.getContactPoints(bodyA=self.panda, linkIndexA=i, bodyB=self.panda, linkIndexB=j)
                if contact_points:
                    return True  # Collision detected
        
        return False  # No collisions detected

    def detect_grasp_contact(self, thred=2):
        points_left = self.world.get_link_contacts(self.panda, self.LEFTFINGERIDX)
        points_right = self.world.get_link_contacts(self.panda, self.RIGHTFINGERIDX)
        if len(points_left)>thred or len(points_right)>thred:
            return True
        else:
            return False

    def set_target(self, tcp_pose, finger_target, gain=0.01, gain_gripper=0.001, threshold=0.05, gain_near_target=0.01):
        # compute the desired TCP pose
        T_world_gripper = tcp_pose * self.T_tcp_gripper
        target_pos = list(T_world_gripper.translation)
        target_orn = list(T_world_gripper.rotation.as_quat())
        # get the current TCP pose
        current_gripper_pose = self.get_tcp_pose() * self.T_tcp_gripper
        current_pos = list(current_gripper_pose.translation)
        # calculate the distance between current and target TCP pose
        distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
        # adjust the gain based on the distance to the target
        if distance_to_target < threshold:
            gain = gain_near_target  # Higher gain near the target for more aggressive behavior
        else:
            gain = gain  # Use base gain when far from the target
        # calculate the inverse kinematics for the desired target pose
        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, self.ENDEFFECTORIDX, target_pos, target_orn, 
                                                                self.LL, self.UL, self.JR, self.HOME_JOINT_POS, maxNumIterations=20)
        # set joint motor control with dynamic position gain
        for i in range(self.NUM_DOFS):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i], 
                                                    positionGain=gain, force=10 * 240.)

        # set gripper control with a fixed gain
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, finger_target, 
                                                    positionGain=gain_gripper, force=5000)
            
    def apply_action(self, action, gain=0.01, gain_gripper=0.001, gain_near_target=0.01):
        x, y, z, roll, pitch, yaw = action[:6]
        target_tcp_pose = Transform(Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False), list([x, y, z]))
        finger_target = action[6]
        self.update_wrist_camera_extrinsic()
        self.set_target(target_tcp_pose, finger_target, gain, gain_gripper, gain_near_target)

    def get_tcp_pose(self):
        results = self.world.p.getLinkState(self.panda, self.ENDEFFECTORIDX)
        current_tcp_pose = Transform(Rotation(list(results[1])), list(results[0]))*self.T_gripper_tcp
        return current_tcp_pose
    
    def get_fingertip_pose(self):
        tcp_pose = self.get_tcp_pose()
        z_axis_tcp = tcp_pose.rotation.as_matrix()[:, 2]
        fingertip_position = tcp_pose.translation + (self.finger_depth+self.finger_depth/8)*z_axis_tcp
        fingertip_pose = Transform(tcp_pose.rotation, fingertip_position)
        return fingertip_pose
    
    def get_finger_velocity(self):
        link_state = self.world.p.getLinkState(self.panda, self.LEFTFINGERIDX, computeLinkVelocity=1)
        linear_velocity_left = link_state[6]
        angular_velocity_left = link_state[7]
        link_state = self.world.p.getLinkState(self.panda, self.RIGHTFINGERIDX, computeLinkVelocity=1)
        linear_velocity_right = link_state[6]
        angular_velocity_right = link_state[7]
        left_velocity = [linear_velocity_left, angular_velocity_left]
        right_velocity = [linear_velocity_right, angular_velocity_right]
        return left_velocity, right_velocity

    def get_tcp_velocity(self):
        link_state = self.world.p.getLinkState(self.panda, self.ENDEFFECTORIDX, computeLinkVelocity=1)
        linear_velocity_ee = link_state[6]
        angular_velocity_ee = link_state[7]
        # print("linear_velocity_ee")
        # print(linear_velocity_ee)
        # print("angular_velocity_ee")
        # print(angular_velocity_ee)
        R = self.T_gripper_tcp.rotation.as_matrix()
        t = self.T_gripper_tcp.translation
        linear_velocity = np.dot(R, np.array(linear_velocity_ee)) + np.cross(np.array(angular_velocity_ee), t)
        angular_velocity = np.dot(R, angular_velocity_ee)
        return linear_velocity, angular_velocity
    
    def get_joint_states(self):
        joint_states_result = list(self.world.p.getJointStates(self.panda, range(self.NUM_DOFS)))
        current_joint_states = [joint_states[0] for joint_states in joint_states_result]
        return current_joint_states
    
    def get_gripper_state(self):
        joint_states_result = self.bullet_client.getJointState(self.panda, self.LEFTFINGERIDX)
        gripper_state = joint_states_result[0]
        return gripper_state