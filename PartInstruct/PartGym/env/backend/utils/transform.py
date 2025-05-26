import numpy as np
import scipy.spatial.transform


class Rotation(scipy.spatial.transform.Rotation):
    @classmethod
    def identity(cls):
        return cls.from_quat([0.0, 0.0, 0.0, 1.0])


class Transform(object):
    """Rigid spatial transform between coordinate systems in 3D space.

    Attributes:
        rotation (scipy.spatial.transform.Rotation)
        translation (np.ndarray)
    """

    def __init__(self, rotation, translation):
        assert isinstance(rotation, scipy.spatial.transform.Rotation)
        assert isinstance(translation, (np.ndarray, list))

        self.rotation = rotation
        self.translation = np.asarray(translation, np.double)

    def as_matrix(self):
        """Represent as a 4x4 matrix."""
        return np.vstack(
            (np.c_[self.rotation.as_matrix(), self.translation], [0.0, 0.0, 0.0, 1.0])
        )

    def to_dict(self):
        """Serialize Transform object into a dictionary."""
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    def to_list(self):
        return np.r_[self.rotation.as_quat(), self.translation]

    def __mul__(self, other):
        """Compose this transform with another."""
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def transform_point(self, point):
        return self.rotation.apply(point) + self.translation

    def transform_vector(self, vector):
        return self.rotation.apply(vector)

    def inverse(self):
        """Compute the inverse of this transform."""
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        """Initialize from a 4x4 matrix."""
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_dict(cls, dictionary):
        rotation = Rotation.from_quat(dictionary["rotation"])
        translation = np.asarray(dictionary["translation"])
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, list):
        rotation = Rotation.from_quat(list[:4])
        translation = list[4:]
        return cls(rotation, translation)

    @classmethod
    def identity(cls):
        """Initialize with the identity transformation."""
        rotation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    @classmethod
    def look_at(cls, eye, center, up):
        """Initialize with a LookAt matrix.

        Returns:
            T_eye_ref, the transform from camera to the reference frame, w.r.t.
            which the input arguments were defined.
        """
        eye = np.asarray(eye)
        center = np.asarray(center)

        forward = center - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up = np.asarray(up) / np.linalg.norm(up)
        up = np.cross(right, forward)

        m = np.eye(4, 4)
        m[:3, 0] = right
        m[:3, 1] = -up
        m[:3, 2] = forward
        m[:3, 3] = eye

        return cls.from_matrix(m).inverse()


def quaternion_distance(q1, q2):
    """
    Calculate the distance between two quaternions.
    
    Args:
        q1 (numpy.ndarray): The first quaternion (4 elements).
        q2 (numpy.ndarray): The second quaternion (4 elements).
        
    Returns:
        float: The distance between the two quaternions in radians.
    """
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate the dot product between the quaternions
    dot_product = np.dot(q1, q2)
    
    # Clamp the dot product to the range [-1, 1]
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate the angle between the quaternions
    angle = 2 * np.arccos(dot_product)
    
    return angle

def pose_distance(pose1, pose2, position_weight=1.0, orientation_weight=1.0):
    """
    Calculate a weighted distance between two poses.

    Parameters:
        pose1 (Transform): The first pose.
        pose2 (Transform): The second pose.
        position_weight (float): The weight for position distance.
        orientation_weight (float): The weight for orientation distance.

    Returns:
        float: The weighted distance between the two poses.
    """
    # Calculate position distance
    pos_distance = np.linalg.norm(pose1.translation - pose2.translation)

    # Calculate orientation distance
    quat1 = pose1.rotation.as_quat()
    quat2 = pose2.rotation.as_quat()
    dot_product = np.clip(np.dot(quat1, quat2), -1.0, 1.0)
    # Adjust for the shortest path
    if dot_product < 0.0:
        dot_product = -dot_product
    angle_distance = 2 * np.arccos(dot_product)  # Angle in radians

    # Weighted sum of distances
    distance = position_weight * pos_distance + orientation_weight * angle_distance
    return distance

def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def random_object_position(robot_base_position, obj_z, min_distance=0.55, max_distance=0.65):
    # Randomly generate angle in radians for the direction from the robot base to the object
    angle = np.random.uniform(np.pi-np.pi/10, np.pi*3/2-np.pi/3-np.pi/8)

    # Randomly choose a distance within the specified range
    distance = np.random.uniform(min_distance, max_distance)

    # Calculate the object's position
    object_x = robot_base_position[0] + distance * np.cos(angle)
    object_y = robot_base_position[1] + distance * np.sin(angle)
    object_z = obj_z  # Assuming the object is on the same height as the robot base

    return [object_x, object_y, object_z]

def rotation_matrix_x(angle_degrees):
    """Returns the rotation matrix for a rotation around the x-axis by `angle_degrees`."""
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([
        [1, 0, 0],
        [0, cos_angle, -sin_angle],
        [0, sin_angle, cos_angle]
    ])

def rotation_matrix_y(angle_degrees):
    """Returns the rotation matrix for a rotation around the y-axis by `angle_degrees`."""
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ])

def rotation_matrix_z(angle_degrees):
    """Returns the rotation matrix for a rotation around the z-axis by `angle_degrees`."""
    angle_radians = np.radians(angle_degrees)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])

def find_retreat_pose(T_world_grasp):
    approach = T_world_grasp.rotation.as_matrix()[:, 2]
    angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
    if angle > np.pi / 3.0:
        T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.35])
        T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
    else:
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.35])
        T_world_retreat = T_world_grasp * T_grasp_retreat
    
    return T_world_retreat