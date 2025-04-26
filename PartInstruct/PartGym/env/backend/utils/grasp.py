import enum
from itertools import count
from PartInstruct.PartGym.env.backend.utils.transform import *
from PartInstruct.PartGym.env.backend.utils.planning_utils import Pose, Attachment

class Label(enum.IntEnum):
    FAILURE = 0  # grasp execution failed due to collision or slippage
    SUCCESS = 1  # object was successfully removed


class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    
    TODO(mbreyer): clarify definition of grasp frame
    """
    def __init__(self, pose, width=0.04):
        self.pose = pose
        self.width = width
        self.T_gripper_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.05])
        self.T_tcp_gripper = self.T_gripper_tcp.inverse()
    
    def to_planning_grasp(self, body, robot, link):
        pose_gripper = self.pose * self.T_tcp_gripper
        grasp_pose = Pose(point=pose_gripper.translation, euler=pose_gripper.rotation.as_euler('xyz', degrees=False))
        grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
        pregrasp = self.pose * grasp_pregrasp
        pregrasp_gripper = pregrasp * self.T_tcp_gripper
        approach_pose = Pose(point=pregrasp_gripper.translation, euler=pregrasp_gripper.rotation.as_euler('xyz', degrees=False))
        return BodyGrasp(body, grasp_pose, approach_pose, robot, link)

class BodyGrasp(object):
    num = count()
    def __init__(self, body, grasp_pose, approach_pose, robot, link):
        self.body = body
        self.grasp_pose = grasp_pose
        self.approach_pose = approach_pose
        self.robot = robot
        self.link = link
        self.index = next(self.num)
    @property
    def value(self):
        return self.grasp_pose
    @property
    def approach(self):
        return self.approach_pose
    #def constraint(self):
    #    grasp_constraint()
    def attachment(self):
        return Attachment(self.robot, self.link, self.grasp_pose, self.body)
    def assign(self):
        return self.attachment().assign()
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'g{}'.format(index)

def to_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation /= voxel_size
    width = grasp.width / voxel_size
    return Grasp(pose, width)


def from_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation *= voxel_size
    width = grasp.width * voxel_size
    return Grasp(pose, width)
